"""T1m steps 2-3 driver: quantize -> fixed-point reference -> C tables -> gcc bit-exact check
-> SDCC builds (board UART firmware + ucsim firmware) -> ucsim bit-exact check + cycle count.

Per trained model  results/T1m/models/<name>_t<seed>.npz  writes
  results/T1m/build/<name>_t<seed>/{model.h, uart_35M.ihx/.hex/.bin, sim.ihx, *.mem, *.map}
  results/T1m/quant/<name>_t<seed>.json
and finally results/T1m/results_all.csv (every seed) and results/T1m/results.csv
(per model: mean over seeds; build / cycle columns from seed 1).

Nothing here touches the board (flashing: step4_flash_and_measure.sh).
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess

import numpy as np

from common import EXP, MODELS, ensure, load_fmnist_u8
from gen_c import emit
from quant import float_forward, quantize, ref_forward

C_DIR = os.path.dirname(os.path.abspath(__file__)) + "/c"
BUILD = f"{EXP}/build"
QDIR = f"{EXP}/quant"
S51 = os.environ.get("S51", "")  # path to ucsim s51 (sdcc-ucsim 4.2.0 extracted locally)
FLASH_MAX = 65024  # 63.5 KB code flash (stcgal report)
XRAM_MAX = 8192
F_OSC = 35000000
SDCC = ["sdcc", "-mmcs51", "--model-large", "--opt-code-speed"]


def sdcc_build(bdir, mains, out, defines, xram_size, extra=()):
    rels = []
    for src in ["infer", "cyc"] + list(extra) + mains:
        rel = f"{bdir}/{out}_{src}.rel"
        subprocess.run(SDCC + [f"-I{bdir}", f"-I{C_DIR}"] + defines + ["-c", f"{C_DIR}/{src}.c", "-o", rel],
                       check=True, capture_output=True, text=True)
        rels.append(rel)
    r = subprocess.run(SDCC + ["--code-size", "65536", "--xram-size", str(xram_size), "--iram-size", "256"]
                       + rels + ["-o", f"{bdir}/{out}.ihx"], capture_output=True, text=True)
    log = r.stdout + r.stderr
    with open(f"{bdir}/{out}.link.log", "w") as f:
        f.write(log)
    mem = open(f"{bdir}/{out}.mem").read() if os.path.exists(f"{bdir}/{out}.mem") else ""
    m_rom = re.search(r"ROM/EPROM/FLASH\s+0x\w+\s+0x\w+\s+(\d+)", mem)
    m_x = re.search(r"EXTERNAL RAM\s+0x\w+\s+0x\w+\s+(\d+)", mem)
    info = dict(ok=(r.returncode == 0 and m_rom is not None and "rror" not in log),
                flash=int(m_rom.group(1)) if m_rom else None,
                xdata=int(m_x.group(1)) if m_x else 0, link_log=log.strip()[-400:])
    if info["ok"]:
        with open(f"{bdir}/{out}.hex", "w") as fh:
            subprocess.run(["packihx", f"{bdir}/{out}.ihx"], check=True, stdout=fh, stderr=subprocess.DEVNULL)
        subprocess.run(["makebin", "-s", "65536", "-p", f"{bdir}/{out}.ihx", f"{bdir}/{out}.bin"], check=True)
    return info


def map_addr(bdir, sym):
    m = re.search(r"\b([0-9A-F]{8})\s+" + re.escape(sym) + r"\b", open(f"{bdir}/sim.map").read())
    return int(m.group(1), 16)


def ucsim_run(bdir, img):
    """One inference in s51 (-t 52). Returns (class, logits[10], timer0 cycles, overhead, stats)."""
    done, res, im = map_addr(bdir, "_sim_done"), map_addr(bdir, "_res"), map_addr(bdir, "_img")
    mark = map_addr(bdir, "_sim_mark")
    cmds = [f"set memory xram 0x{im + i:x} " + " ".join(str(int(v)) for v in img[i:i + 16])
            for i in range(0, 784, 16)]
    # stop at sim_mark (end of inference) for the instruction statistics, then run to sim_done
    cmds += [f"break 0x{mark:x}", f"break 0x{done:x}", "run", "state", "run", "state",
             f"dx 0x{res:x} 0x{res + 63:x}", "quit"]
    out = subprocess.run([S51, "-t", "52", f"{bdir}/sim.ihx"], input="\n".join(cmds) + "\n",
                         capture_output=True, text=True, timeout=3600).stdout
    b = []
    for line in out.splitlines():
        m = re.match(r"0x([0-9a-f]{4,})\s+((?:[0-9a-f]{2} ){1,8})", line)
        if m and res <= int(m.group(1), 16) < res + 64:
            b += [int(v, 16) for v in m.group(2).split()]
    b = np.array(b[:64], np.uint8)
    assert b[49] == 0xA5, out[-2000:]
    logits = b[1:41].copy().view("<i4").astype(np.int64)
    cyc = int(b[41:45].copy().view("<u4")[0])
    cyc0 = int(b[45:49].copy().view("<u4")[0])
    st = dict(inst=int(re.search(r"Inst= (\d+)", out).group(1)),
              fetch=int(re.search(r"Fetch= (\d+)", out).group(1)),
              clks=int(re.search(r"\((\d+) clks\)", out).group(1)))
    st["gen_cycles"] = int(b[50:54].copy().view("<u4")[0]) - cyc0  # PRNG models only (else garbage)
    return int(b[0]), logits, cyc, cyc0, st


def process(tag, data, n_sim, do_sim, do_sdcc=True):
    m = dict(np.load(f"{MODELS}/{tag}.npz"))
    meta = json.load(open(f"{MODELS}/{tag}.json"))
    xtr, ytr = data["train"]
    xte, yte = data["test"]
    vi = m["val_idx"]
    q = quantize(m, xtr[vi], ytr[vi])
    L = ref_forward(q, xte)
    acc_int8 = float((L.argmax(1) == yte).mean())
    acc_fp32_np = float((float_forward(m, xte).argmax(1) == yte).mean())
    bdir = ensure(f"{BUILD}/{tag}")
    mask_source = meta.get("mask_source") or ("official" if meta["kind"] == "dann" else "")
    prng = mask_source == "prng"
    extra = ["prng_rf"] if prng else []
    tables = emit(q, meta["kind"], meta["cfg"], f"{bdir}/model.h", tag, M1=m.get("M1"),
                  mask_source=mask_source, model_seed=meta["seed"])
    # --- gcc build of the same infer.c, all 10 000 test images, bit-exact vs reference
    subprocess.run(["gcc", "-O2", "-Wall", "-Wno-unused-function", f"-I{bdir}", f"-I{C_DIR}",
                    f"{C_DIR}/infer.c", f"{C_DIR}/pc_main.c"] + [f"{C_DIR}/{e}.c" for e in extra]
                   + ["-o", f"{bdir}/pc_infer"], check=True)
    xte.tofile(f"{bdir}/test_x.u8")
    subprocess.run([f"{bdir}/pc_infer", f"{bdir}/test_x.u8", f"{bdir}/pc_out.bin"], check=True)
    raw = np.fromfile(f"{bdir}/pc_out.bin", np.uint8).reshape(-1, 41)
    pc_logits = raw[:, :40].copy().view("<i4").astype(np.int64)
    pc_match = int((pc_logits == L).all(1).sum())
    pc_cls = int((raw[:, 40] == L.argmax(1)).sum())
    os.remove(f"{bdir}/test_x.u8")
    os.remove(f"{bdir}/pc_out.bin")
    # --- SDCC: board firmware (STC8, UART, 35 MHz) and ucsim firmware
    if do_sdcc:
        uart = sdcc_build(bdir, ["uart_main"], "uart_35M", ["-DSTC8", f"-DF_OSC={F_OSC}UL"], XRAM_MAX, extra)
        sim = sdcc_build(bdir, ["sim_main"], "sim", [], 0x1B00, extra)
    else:  # SDCC is slow on large tables; footprint does not depend on the seed -> seed 1 only
        uart = sim = dict(ok=False, flash=None, xdata=None, link_log="not built (seed != 1)")
    fits = bool(uart["ok"] and uart["flash"] <= FLASH_MAX and uart["xdata"] <= XRAM_MAX)
    r = dict(tag=tag, model=meta["name"], kind=meta["kind"], mask_source=mask_source, seed=meta["seed"],
             params=meta["params"],
             acc_fp32=meta["acc_fp32"], acc_fp32_numpy=acc_fp32_np, acc_int8=acc_int8,
             cal_acc_int8=q["cal_acc"], sh1=q["sh1"], sh2=q["sh2"], acc_bound=q["acc_bound"],
             table_bytes=tables["total"], tables=tables,
             flash_bytes=uart["flash"], xdata_bytes=uart["xdata"], fits=fits,
             uart_build_ok=uart["ok"], uart_link_log=None if uart["ok"] else uart["link_log"],
             sim_build_ok=sim["ok"], sim_flash=sim["flash"],
             pc_bitexact_logits=f"{pc_match}/{len(xte)}", pc_bitexact_class=f"{pc_cls}/{len(xte)}")
    if do_sim and sim["ok"] and S51:
        cycs, insts, fetch, gens, n_ok, n_cls = [], [], [], [], 0, 0
        for i in range(n_sim):
            c, lg, cyc, cyc0, st = ucsim_run(bdir, xte[i])
            n_ok += int((lg == L[i]).all())
            n_cls += int(c == L[i].argmax())
            cycs.append(cyc - cyc0)
            insts.append(st["inst"])
            fetch.append(st["fetch"])
            gens.append(st["gen_cycles"])
        r.update(ucsim_n=n_sim, ucsim_bitexact_logits=f"{n_ok}/{n_sim}", ucsim_bitexact_class=f"{n_cls}/{n_sim}",
                 cycles_ucsim_mc=float(np.mean(cycs)), cycles_ucsim_mc_min=int(min(cycs)),
                 cycles_ucsim_mc_max=int(max(cycs)), instr_ucsim=float(np.mean(insts)),
                 fetch_bytes_ucsim=float(np.mean(fetch)))
        if prng:  # index regeneration alone: all H1 rf_gen calls (+ loop), timed separately
            r.update(gen_cycles_ucsim_mc=float(np.mean(gens)))
    with open(f"{ensure(QDIR)}/{tag}.json", "w") as f:
        json.dump(r, f, indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "tables"}), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--n_sim", type=int, default=20)
    ap.add_argument("--sim_seeds", type=int, nargs="*", default=[1])
    ap.add_argument("--sdcc_seeds", type=int, nargs="*", default=[1])
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--prefix", default="", help="only models whose name starts with this")
    a = ap.parse_args()
    data = load_fmnist_u8()
    tags = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{MODELS}/*.npz"))
    for tag in tags:
        name, seed = tag.rsplit("_t", 1)
        if a.only and name not in a.only and tag not in a.only:
            continue
        if a.prefix and not name.startswith(a.prefix):
            continue
        if a.skip_done and os.path.exists(f"{QDIR}/{tag}.json"):
            continue
        process(tag, data, a.n_sim, int(seed) in a.sim_seeds, int(seed) in a.sdcc_seeds)


if __name__ == "__main__":
    main()

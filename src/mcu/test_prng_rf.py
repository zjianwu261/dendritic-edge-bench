"""T1m step 5 unit test: rf_gen is bit-exact between Python (prng_rf.py), gcc (prng_rf.c) and
SDCC 4.2.0 / ucsim s51 (prng_rf.c on the 8051). Also measures rf_gen cycles in ucsim.

  python test_prng_rf.py [--n_pc 4096] [--n_sim 64]     (needs env S51 for the ucsim part)
Writes results/T1m/prng_rf_unittest.json.
"""
import argparse
import json
import os
import re
import subprocess

import numpy as np

from common import EXP, ensure
from prng_rf import rf_c, rf_gen, seed_of

C_DIR = os.path.dirname(os.path.abspath(__file__)) + "/c"
SDCC = ["sdcc", "-mmcs51", "--model-large", "--opt-code-speed"]


def py_dump(n, C):
    out = []
    for i in range(n):
        b, off = rf_gen(seed_of(i, C))
        out += [b & 0xFF, b >> 8] + off
    return np.array(out, np.uint8)


def pc_dump(n, C, bdir):
    exe = f"{bdir}/rf_test_pc"
    subprocess.run(["gcc", "-O2", "-Wall", f"-I{C_DIR}", f"-DRF_C={C}u", f"-DN_RF={n}",
                    f"{C_DIR}/prng_rf.c", f"{C_DIR}/rf_test_main.c", "-o", exe], check=True)
    subprocess.run([exe, f"{bdir}/rf_pc.bin"], check=True)
    return np.fromfile(f"{bdir}/rf_pc.bin", np.uint8)


def sim_dump(n, C, bdir, s51):
    rels = []
    for src in ("prng_rf", "cyc", "rf_test_main"):
        rel = f"{bdir}/rft_{src}.rel"
        subprocess.run(SDCC + [f"-I{C_DIR}", f"-DRF_C={C}u", f"-DN_RF={n}", "-c", f"{C_DIR}/{src}.c", "-o", rel],
                       check=True, capture_output=True, text=True)
        rels.append(rel)
    subprocess.run(SDCC + ["--xram-size", "8192"] + rels + ["-o", f"{bdir}/rft.ihx"], check=True,
                   capture_output=True, text=True)
    mp = open(f"{bdir}/rft.map").read()
    done = int(re.search(r"\b([0-9A-F]{8})\s+_sim_done\b", mp).group(1), 16)
    cmds = [f"break 0x{done:x}", "run", f"dx 0x0400 0x{0x400 + 18 * n - 1:x}", "dx 0x0300 0x030f", "quit"]
    out = subprocess.run([s51, "-t", "52", f"{bdir}/rft.ihx"], input="\n".join(cmds) + "\n",
                         capture_output=True, text=True, timeout=600).stdout
    mem = {}
    for line in out.splitlines():
        m = re.match(r"0x([0-9a-f]{4,})\s+((?:[0-9a-f]{2} ){1,8})", line)
        if m:
            a = int(m.group(1), 16)
            for k, v in enumerate(m.group(2).split()):
                mem.setdefault(a + k, int(v, 16))
    dump = np.array([mem[0x400 + i] for i in range(18 * n)], np.uint8)
    res = [mem[0x300 + i] for i in range(9)]
    assert res[8] == 0xA5
    cy = int.from_bytes(bytes(res[0:4]), "little")
    cy0 = int.from_bytes(bytes(res[4:8]), "little")
    return dump, (cy - cy0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pc", type=int, default=4096)
    ap.add_argument("--n_sim", type=int, default=64)
    a = ap.parse_args()
    s51 = os.environ.get("S51", "")
    bdir = ensure(f"{EXP}/build/prng_rf_unittest")
    rep = {}
    for seed in (1, 2, 3):
        C = rf_c(seed)
        py = py_dump(a.n_pc, C)
        pc = pc_dump(a.n_pc, C, bdir)
        r = dict(rf_c=f"0x{C:04x}", n_pc=a.n_pc, pc_equal=bool(np.array_equal(py, pc)),
                 first3_py=[rf_gen(seed_of(i, C)) for i in range(3)])
        if s51:
            sd, cyc = sim_dump(a.n_sim, C, bdir, s51)
            r.update(n_sim=a.n_sim, sim_equal=bool(np.array_equal(py[:18 * a.n_sim], sd)),
                     rf_gen_machine_cycles_mean=cyc)
        rep[f"seed{seed}"] = r
        print(seed, json.dumps(r), flush=True)
        assert r["pc_equal"] and r.get("sim_equal", True), r
    with open(f"{EXP}/prng_rf_unittest.json", "w") as f:
        json.dump(rep, f, indent=1)
    print("ALL BIT-EXACT")


if __name__ == "__main__":
    main()

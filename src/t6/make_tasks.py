"""Append T6 training tasks to the task file ($DEB_QUEUE/tasks.txt, run by src/tools/run_tasks.py).
One task = one configuration x 3 seeds, run sequentially (seeds already finished are skipped).
Usage: python make_tasks.py {g1|g2|g2base|g3} [--dry]
"""
import os
import sys

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TASKS = os.path.join(os.environ.get("DEB_QUEUE", f"{ROOT}/work/queue"), "tasks.txt")
PY = os.environ.get("PYTHON", sys.executable)
SEEDS = (1, 2, 3)


def seeds_cmd(base):
    return " && ".join(f"{PY} {base} --seed {s}" for s in SEEDS)


def g1():
    out = []
    # MLP baselines first (cheap, needed for latency matching), then dANN-G, then 2:4 variants
    for h in (32, 64, 128, 256, 512, 1024, 2048):
        out.append((f"g1_mlp_h{h}", 1100, seeds_cmd(f"dann_g.py --arch mlp --h {h}")))
    for m in (8, 16, 32, 64):
        for s in (64, 128, 256):
            for sg in (1, 4):
                if (m // sg) % 4:
                    continue
                out.append((f"g1_dannG_m{m}_s{s}_sg{sg}", 1100,
                            seeds_cmd(f"dann_g.py --arch dannG --p 4 --m {m} --s {s} --sg {sg}")))
    for m in (16, 32, 64):
        out.append((f"g1_dannG_m{m}_s128_sg4_sp24", 1100,
                    seeds_cmd(f"dann_g.py --arch dannG --p 4 --m {m} --s 128 --sg 4 --sp24")))
    for D in (512, 1024, 2048, 4096):
        for s in (128, 256):
            out.append((f"g1_dann24_D{D}_s{s}", 1300, seeds_cmd(f"dann_g.py --arch dann24 --D {D} --s {s}")))
    return out


def g2():
    """DH-SSM sweep, one task per (cfg, seed), priority order; 60 epochs, cosine, dropout .1 (pilot recipe)."""
    out = []
    pri = [(2, 128, 4), (2, 128, 1), (1, 64, 8), (1, 64, 1), (2, 64, 4), (2, 256, 4), (1, 128, 4),
           (2, 128, 8), (2, 128, 2), (1, 128, 1), (2, 64, 1), (2, 256, 1), (1, 256, 4), (1, 64, 4)]
    for s in SEEDS:
        for L, n, nb in pri:
            for act in ("spk", "cont"):
                C = n * nb
                micro, mem = (4, 2400) if C <= 512 else (10, 2400)
                out.append((f"g2_dhssm_L{L}_n{n}_b{nb}_{act}_t{s}", mem,
                            f"{PY} dh_ssm.py --L {L} --n {n} --nb {nb} --act {act} --epochs 60 --micro {micro} --seed {s}"))
    return out


def g2base():
    """GRU baselines (ONNX GRU op at deployment), two regularisation settings, selected on val."""
    out = []
    for s in SEEDS:
        for h in (32, 64, 128, 256):
            for do, wd, tag in ((0.0, 0.0, "_r0"), (0.3, 1e-2, "_r1")):
                out.append((f"g2_gru_h{h}{tag}_t{s}", 2400,
                            f"{PY} dh_ssm.py --arch gru --h {h} --lr 3e-3 --dropout {do} --wd {wd} --epochs 60 "
                            f"--micro {4 if h < 128 else 10} --tag {tag} --seed {s}"))
    return out


def g3():
    out = []
    for h in (256, 512, 1024, 2048):
        for ff in ("rand", "24"):
            out.append((f"g3_adc_h{h}_ff{ff}", 1500 if h >= 2048 else 1200,
                        seeds_cmd(f"ad_cache.py --arch ad --h {h} --ff {ff}")))
    for h in (128, 256, 512, 1024, 2048, 4096):
        out.append((f"g3_mlp_h{h}", 1500 if h >= 2048 else 1200, seeds_cmd(f"ad_cache.py --arch mlp --h {h}")))
    return out


def g3ctx():
    """Dense G3 baseline with task context (one-hot of the nearest-prototype task), width sweep."""
    return [(f"g3_mlpctx_h{h}", 1500 if h >= 2048 else 1200, seeds_cmd(f"ad_cache.py --arch mlpctx --h {h}"))
            for h in (256, 512, 1024, 2048, 4096)]


def g2v():
    """Speaker-disjoint validation split, prio 40 (< t4a_*).
    Equal budgets: DH-SSM 8 configs (fixed recipe), non-dendritic diagonal SSM 4, LRU 4, GRU 8, LSTM 4.
    Every config x 3 seeds, 60 epochs, AdamW wd 1e-2, cosine; selection on the speaker val split."""
    E = "--epochs 60 --split speaker"
    cfgs = []
    for L, n, nb, act in [(2, 128, 4, "cont"), (2, 128, 4, "spk"), (2, 256, 4, "cont"), (2, 256, 4, "spk"),
                          (2, 64, 4, "cont"), (2, 128, 8, "cont"), (1, 64, 8, "spk"), (1, 64, 8, "cont")]:
        cfgs.append((f"dhssm_L{L}_n{n}_b{nb}_{act}", n * nb,
                     f"dh_ssm.py --arch dhssm --L {L} --n {n} --nb {nb} --act {act} --dropout 0.1 {E}"))
    for arch, L, n, nb, act in [("dssmU", 2, 128, 4, "cont"), ("dssmU", 2, 128, 4, "spk"),
                                ("dssmU", 2, 256, 4, "cont"), ("dhssm", 2, 512, 1, "cont")]:
        cfgs.append((f"{arch}_L{L}_n{n}_b{nb}_{act}_nondend", n * nb,
                     f"dh_ssm.py --arch {arch} --L {L} --n {n} --nb {nb} --act {act} --dropout 0.1 {E}"))
    for H, N in ((128, 256), (64, 128)):
        for lr in ("1e-3", "3e-3"):
            cfgs.append((f"lru_H{H}_N{N}_lr{lr}", 2 * N,
                         f"dh_ssm.py --arch lru --L 2 --H {H} --N {N} --lr {lr} --dropout 0.1 {E}"))
    for h in (128, 256, 512, 768):
        for lr in ("1e-3", "3e-3"):
            cfgs.append((f"gru_h{h}_lr{lr}", h, f"dh_ssm.py --arch gru --h {h} --lr {lr} --dropout 0.3 --wd 1e-2 {E}"))
    for h in (256, 512):
        for lr in ("1e-3", "3e-3"):
            cfgs.append((f"lstm_h{h}_lr{lr}", h, f"dh_ssm.py --arch lstm --h {h} --lr {lr} --dropout 0.3 --wd 1e-2 {E}"))
    out = []
    for s in SEEDS:
        for tid, C, cmd in cfgs:
            micro = 4 if C <= 512 else 10
            out.append((f"g2v_{tid}_t{s}", 2400, f"prio=40 | {PY} {cmd} --micro {micro} --seed {s}"))
    return out


def g1r():
    """G1 MLP with regularisation tuned on the 10 % val split (8 configs) + small CNN (3 widths), prio 45."""
    out = []
    for h in (256, 1024):
        for do in (0.2, 0.5):
            for wd in ("0", "5e-4"):
                out.append((f"g1_mlpreg_h{h}_do{do}_wd{wd}", 1100,
                            "prio=45 | " + seeds_cmd(f"dann_g.py --arch mlpreg --h {h} --dropout {do} --wd {wd}")))
    for c1, c2 in ((8, 16), (16, 32), (32, 64)):
        out.append((f"g1_cnn_c{c1}_{c2}", 1100, "prio=45 | " + seeds_cmd(f"dann_g.py --arch cnn --c1 {c1} --c2 {c2}")))
    return out


def main():
    which = sys.argv[1]
    tasks = globals()[which]()
    existing = set()
    if os.path.exists(TASKS):
        existing = {ln.split("|")[0].strip() for ln in open(TASKS) if "|" in ln and not ln.startswith("#")}
    new = [t for t in tasks if t[0] not in existing]
    if "--dry" in sys.argv:
        for t in new:
            print(" | ".join(map(str, t)))
        return
    with open(TASKS, "a") as f:
        for tid, mem, cmd in new:
            f.write(f"{tid} | {mem} | {cmd}\n")
    print(f"appended {len(new)} tasks ({len(tasks) - len(new)} already present)")


if __name__ == "__main__":
    main()

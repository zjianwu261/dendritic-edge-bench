"""C-c multi-seed timing / energy on Orin #1 (2026-10-26). Workstation side; Orin side = ms_orin/ms_orin.sh.

Sessions (each = one ms_orin.sh batch, rounds interleaved across engines, NRUNS=3 runs per engine, nocublas builds):
  A : seed 1 (the original T1 models): b0 = the original 10-01/10-02 T1 engine (trt_v07, hard link, not rebuilt)
      + b1..b3 fresh trtexec builds of the same ONNX.
  B : seeds 2, 3: b1..b3 fresh builds of the seed-2/3 ONNX; seed 1: b0 and b1 engines of session A re-timed
      (session anchor: per-seed ratios in B are all taken in the same session).
  E : energy (src/bench/energy_one.sh, T1 protocol: jtop VDD_IN 100 ms, idle 12 s / >= 30 s inference / idle 14 s,
      net of idle), 3 reps per engine, interleaved; seed 1 = b0 (the engine of results/T1/energy_orin1.csv),
      seeds 2/3 = b1.
Rules: Orin #1 only (ssh host ORIN_HOST), flock /tmp/dn_orin_bench.lock (inside ms_orin.sh / energy_one.sh),
REQUIRE_LOCK_HZ=624750000 (run skipped unless GPU devfreq min=cur=max=624.75 MHz), stop at start if not locked,
jetson_clocks / nvpmodel never set.
Usage: python ms_orin.py start A|B|C | wait A|B|C|EB|EC | pull <same> | start_energy B|C | status
"""
import json
import os
import subprocess
import sys
import time
import hashlib

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = f"{ROOT}/results/robustness_B/Cc_multiseed"
T1M = f"{ROOT}/results/T1/models"
ORIN = os.environ.get("ORIN_HOST", "orin")
RD = "dendritic-edge-bench/T1/ms_cc"
G = "/sys/devices/gpu.0/devfreq/17000000.ga10b"
LOCK_HZ = 624750000
NRUNS = 3
MODELS = ["dann_lrf_d32_s128", "mlp_h85_match_d32_s128", "vann_same_d32_s128", "mlp_h1024_fmnist",
          "dhsrnn_n64_b8_val_seqloop", "lif_srnn_n64_val_seqloop", "gru_pm_val_seqloop", "gru_acc_val_seqloop",
          "gru_pm_val_seq", "gru_acc_val_seq",
          "dhsrnn_n64_b8_step", "lif_srnn_n64_step", "gru_h23_step", "gru_acc_val_step",
          "active_dend_k4", "active_dend_k8", "mlp_h3592_match_k4", "mlp_h4965_match_k8",
          "mlp_h3941_matchstored_k4", "mlp_h5230_matchstored_k8", "active_dend_k4_cached", "active_dend_k8_cached"]


G2M = [m for m in MODELS if "srnn" in m or m.startswith("gru")]
# 2026-10-26: G1/G3 seeds finish ~1 day before the SHD SNNs (CPU training), so the seed-2/3 session is split:
# B = G1 + G3 models, C = G2 models; same layout (seeds 2/3 b1..b3, seed-1 anchors b0/b1 re-timed from workA).
SESSION_MODELS = {"A": MODELS, "B": [m for m in MODELS if m not in G2M], "C": G2M}


def onnx_path(seed, m):
    return f"{T1M}/{m}.onnx" if seed == 1 else f"{OUT}/models/s{seed}/{m}.onnx"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ssh(cmd, t=60):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", ORIN, cmd],
                          capture_output=True, text=True, timeout=t)


def lock_state():
    r = ssh(f"cat {G}/min_freq {G}/cur_freq {G}/max_freq")
    try:
        return tuple(int(v) for v in r.stdout.split()[:3])
    except ValueError:
        return None


def T():
    return time.strftime("%F %T")


def push_common():
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/bin ~/{RD}/onnx/s1 ~/{RD}/onnx/s2 ~/{RD}/onnx/s3"], check=True)
    subprocess.run(["rsync", "-a", f"{ROOT}/src/robustness_B/ms_orin/ms_orin.sh", f"{ORIN}:{RD}/bin/"], check=True)


def start(session):
    st = lock_state()
    print(T(), "GPU devfreq min/cur/max", st, flush=True)
    if st != (LOCK_HZ,) * 3:
        raise SystemExit(f"GPU NOT LOCKED {st}: stop (jetson_clocks / nvpmodel are not run by this script)")
    W = f"work{session}"
    if ssh(f"test -e ~/{RD}/{W} && echo yes").stdout.strip() == "yes":
        raise SystemExit(f"~/{RD}/{W} already exists on the Orin; refusing to restart")
    push_common()
    os.makedirs(f"{OUT}/raw_orin1_{session}", exist_ok=True)
    lines, info, prep = [], {}, []
    seeds = [1] if session == "A" else [1, 2, 3]
    for s in seeds:
        mods = SESSION_MODELS[session]
        files = [onnx_path(s, m) for m in mods]
        miss = [f for f in files if not os.path.exists(f)]
        if miss:
            raise SystemExit(f"missing ONNX: {miss}")
        subprocess.run(["rsync", "-a"] + files + [f"{ORIN}:{RD}/onnx/s{s}/"], check=True)
        for m, f in zip(mods, files):
            h = sha(f)
            info[f"s{s}_{m}"] = {"seed": s, "model": m, "onnx": f, "onnx_sha256": h}
            if session == "A":
                ks = [0, 1, 2, 3]
                prep.append(f"ln ~/dendritic-edge-bench/T1/trt_v07/{m}_fp16_nocublas_{h[:12]}.engine "
                            f"{W}/engines/s1b0_{m}_fp16_nocublas_{h[:12]}.engine")
            elif s == 1:
                ks = [0, 1]
                for k in ks:
                    prep.append(f"ln workA/engines/s1b{k}_{m}_fp16_nocublas_{h[:12]}.engine "
                                f"{W}/engines/s1b{k}_{m}_fp16_nocublas_{h[:12]}.engine")
            else:
                ks = [1, 2, 3]
            info[f"s{s}_{m}"]["builds"] = ks
    # manifest: build-index outer, then seed, then model (fresh builds spread over the session)
    for k in range(0, 4):
        for s in seeds:
            for m in SESSION_MODELS[session]:
                if k in info[f"s{s}_{m}"]["builds"]:
                    lines.append(f"$HOME/{RD}/onnx/s{s}/{m}.onnx s{s}b{k}_{m} nocublas")
    raw = f"{OUT}/raw_orin1_{session}"
    open(f"{raw}/manifest.txt", "w").write("\n".join(lines) + "\n")
    json.dump(info, open(f"{raw}/onnx_info.json", "w"), indent=1)
    print(len(lines), "engines,", len(lines) * NRUNS, "runs", flush=True)
    # sha check on the Orin
    for v in info.values():
        r = ssh(f"sha256sum ~/{RD}/onnx/s{v['seed']}/{v['model']}.onnx")
        if not r.stdout.startswith(v["onnx_sha256"]):
            raise SystemExit(f"sha mismatch on Orin {v}")
    r = subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/{W}/engines && cd ~/{RD} && " + " && ".join(prep)],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise SystemExit(f"engine link failed: {r.stderr}")
    subprocess.run(["rsync", "-a", f"{raw}/manifest.txt", f"{ORIN}:{RD}/{W}/"], check=True)
    cmd = (f"cd ~/{RD} && sed \"s#^\\$HOME#$HOME#\" {W}/manifest.txt > {W}/manifest_abs.txt && "
           f"REQUIRE_LOCK_HZ={LOCK_HZ} setsid nohup bash bin/ms_orin.sh batch $HOME/{RD}/{W}/manifest_abs.txt $HOME/{RD}/{W} {NRUNS} "
           f"> {W}/batch.out 2>&1 < /dev/null & echo started")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60)
    print(T(), f"session {session} started (REQUIRE_LOCK_HZ set)", flush=True)


def start_energy(session="B"):
    st = lock_state()
    print(T(), "GPU devfreq min/cur/max", st, flush=True)
    if st != (LOCK_HZ,) * 3:
        raise SystemExit(f"GPU NOT LOCKED {st}")
    E = f"energy{session}"
    if ssh(f"test -e ~/{RD}/{E} && echo yes").stdout.strip() == "yes":
        raise SystemExit("energy dir exists; refusing to restart")
    info = json.load(open(f"{OUT}/raw_orin1_{session}/onnx_info.json"))
    lines = []
    for s in (1, 2, 3):
        k = 0 if s == 1 else 1
        for m in SESSION_MODELS[session]:
            h = info[f"s{s}_{m}"]["onnx_sha256"][:12]
            lines.append(f"s{s}b{k}_{m} $HOME/{RD}/work{session}/engines/s{s}b{k}_{m}_fp16_nocublas_{h}.engine")
    raw = f"{OUT}/raw_orin1_E{session}"
    os.makedirs(raw, exist_ok=True)
    open(f"{raw}/energy_list.txt", "w").write("\n".join(lines) + "\n")
    push_common()
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/{E}"], check=True)
    subprocess.run(["rsync", "-a", f"{raw}/energy_list.txt", f"{ORIN}:{RD}/{E}/"], check=True)
    cmd = (f"cd ~/{RD} && sed \"s# \\$HOME# $HOME#\" {E}/energy_list.txt > {E}/energy_list_abs.txt && "
           f"REQUIRE_LOCK_HZ={LOCK_HZ} setsid nohup bash bin/ms_orin.sh energy $HOME/{RD}/{E}/energy_list_abs.txt "
           f"$HOME/{RD}/{E}/out 3 > {E}/energy.out 2>&1 < /dev/null & echo started")
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60)
    print(T(), "energy started", len(lines), "engines x 3 reps", flush=True)


def pull(session):
    raw = f"{OUT}/raw_orin1_{session}"
    os.makedirs(raw, exist_ok=True)
    src = f"{RD}/energy{session[1:]}/" if session.startswith("E") else f"{RD}/work{session}/"
    subprocess.run(["rsync", "-a", "--exclude", "*.engine", f"{ORIN}:{src}", f"{raw}/"], check=True)


def wait(session, poll_s=300, max_h=14):
    t_end, lost = time.time() + float(max_h) * 3600, 0
    if session.startswith("E"):
        prog, done_pat = f"~/{RD}/energy{session[1:]}/out/progress.txt", "energy done"
    else:
        prog, done_pat = f"~/{RD}/work{session}/progress.txt", "batch done"
    while True:
        try:
            done = ssh(f"grep -c '{done_pat}' {prog} || true").stdout.strip()
            st = lock_state()
            if st != (LOCK_HZ,) * 3:
                lost += 1
                print(T(), "WARNING LOCK LOST", st, flush=True)
            if done not in ("", "0"):
                break
            print(T(), "running:", ssh(f"tail -1 {prog}").stdout.strip(), "| lock", st, flush=True)
        except subprocess.TimeoutExpired:
            print(T(), "ssh timeout", flush=True)
        if time.time() > t_end:
            raise SystemExit("not done in time")
        time.sleep(int(poll_s))
    pull(session)
    print(T(), f"session {session} done and pulled; lock-lost polls {lost}", flush=True)


def status():
    for s in ("A", "B", "C"):
        print(s, ssh(f"tail -2 ~/{RD}/work{s}/progress.txt 2>/dev/null; wc -l < ~/{RD}/work{s}/results.jsonl 2>/dev/null").stdout)
    for e in ("B", "C"):
        print("E" + e, ssh(f"tail -2 ~/{RD}/energy{e}/out/progress.txt 2>/dev/null").stdout)
    print("lock", lock_state(), ssh("df -h ~ | tail -1").stdout)


if __name__ == "__main__":
    f = {"start": start, "wait": wait, "pull": pull, "start_energy": start_energy, "status": status}[sys.argv[1]]
    f(*sys.argv[2:])

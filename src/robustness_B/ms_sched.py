"""Memory-aware CPU scheduler for the C-c multi-seed retraining (replaces ms_run_all.sh's fixed xargs -P after the
workstation ran out of RAM + swap with 10 parallel jobs next to other GPU jobs, 2026-10-26 08:45).
A job (ms_train.py --cpu <job> <seed_index>) is started only if
  MemAvailable - need(job) >= HEADROOM_MB (default 5000; the other jobs and their loader workers keep growing)
  and fewer than MAXP of our jobs run; jobs already running (pgrep) or with a .done marker are skipped.
Order = longest first. After all training jobs: ms_train.py export 2 / 3.
Usage: python ms_sched.py [MAXP=6] [THREADS=2]
"""
import os
import subprocess
import sys
import time

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PY = os.environ.get("PYTHON", sys.executable)
L = f"{ROOT}/results/robustness_B/Cc_multiseed/logs"
HEADROOM_MB = int(os.environ.get("MS_HEADROOM_MB", 5000))
NEED = {"g2v_dhsrnn": 3500, "g2o_dhsrnn": 3500, "g2v_gruacc": 2200, "g2v_lif": 1800, "g2o_lif": 2200, "g2o_gru23": 1800,
        "g3_mlp5230": 2000, "g3_mlp3941": 1800, "g2v_grupm": 1200, "g3_mlp4965": 1800, "g3_mlp3592": 1500,
        "g3_ad8": 2000, "g3_ad4": 1500, "g1_dann": 1200, "g1_vann": 1200, "g1_mlp1024": 1000, "g1_mlp85": 800}
ORDER = list(NEED)


def avail_mb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) // 1024
    return 0


def running(job, si):
    return subprocess.run(["pgrep", "-f", f"ms_train.py --cpu {job} {si}$"], capture_output=True).returncode == 0


def log(msg):
    with open(f"{L}/driver.log", "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S") + " [sched] " + msg + "\n")


def main():
    maxp = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    th = sys.argv[2] if len(sys.argv) > 2 else "2"
    procs = {}
    pending = [(j, si) for j in ORDER for si in (2, 3)]
    log(f"start maxp={maxp} threads={th} headroom={HEADROOM_MB}")
    last_wait = 0
    while True:
        for key, p in list(procs.items()):
            rc = p.poll()
            if rc is not None:
                j, si = key
                log(f"END {j} s{si} rc={rc}")
                if rc == 0:
                    open(f"{L}/{j}_s{si}.done", "w").write(time.strftime("%FT%T") + "\n")
                del procs[key]
        pending = [(j, si) for j, si in pending if not os.path.exists(f"{L}/{j}_s{si}.done") and (j, si) not in procs]
        ours = sum(1 for j, si in pending if running(j, si)) + len(procs)
        if not pending and not procs:
            break
        for j, si in pending:
            if running(j, si):
                continue
            if ours >= maxp:
                break
            am = avail_mb()
            if am - NEED[j] < HEADROOM_MB:
                if time.time() - last_wait > 1800:
                    log(f"waiting for memory: next {j} s{si} needs {NEED[j]}, MemAvailable {am}")
                    last_wait = time.time()
                break
            env = dict(os.environ, OMP_NUM_THREADS=th, MKL_NUM_THREADS=th)
            out = open(f"{L}/{j}_s{si}.log", "w")
            procs[(j, si)] = subprocess.Popen(["nice", "-n", "10", PY, "-u", f"{ROOT}/src/robustness_B/ms_train.py", "--cpu", j, str(si)],
                                              stdout=out, stderr=subprocess.STDOUT, env=env)
            log(f"START {j} s{si} (MemAvailable {am})")
            ours += 1
            time.sleep(90)  # let it allocate before the next memory check
            break
        # jobs that are pending but running elsewhere (older driver): wait for their .done
        if all(running(j, si) for j, si in pending) and not procs and pending:
            pass
        time.sleep(60)
    for si in (2, 3):
        r = subprocess.run([PY, "-u", f"{ROOT}/src/robustness_B/ms_train.py", "--cpu", "export", str(si)],
                           stdout=open(f"{L}/export_s{si}.log", "w"), stderr=subprocess.STDOUT)
        log(f"export s{si} rc={r.returncode}")
    log("ALL DONE")


if __name__ == "__main__":
    main()

"""Race-free successor of ms_sched.py (2026-10-26 18:30). ms_sched.py decided "finished" from the child's return code
and "start" from (no .done and no pgrep match); with two scheduler instances, one restarted g2v_lif s2 in the ~2 s between
the first run's exit and its .done marker (the rerun overwrote the best-val checkpoint; it is let run to completion, the
first run's metadata is kept as models/s2/shd_val/lif_srnn_n64_s1.firstrun_json_backup).
Here completion is read from the job log: a job is done iff its process is not running and its log contains the final
metadata line ('"test_acc"'); a job is started only if it is neither done nor running. Single instance (pid file).
Usage: python ms_sched2.py [MAXP=6] [THREADS=2]
"""
import os
import subprocess
import sys
import time

from ms_sched import HEADROOM_MB, L, NEED, ORDER, PY, ROOT, avail_mb, log, running

PID = f"{L}/ms_sched2.pid"


def log_done(j, si):
    p = f"{L}/{j}_s{si}.log"
    return os.path.exists(p) and '"test_acc"' in open(p, errors="replace").read()


def main():
    if os.path.exists(PID):
        try:
            os.kill(int(open(PID).read()), 0)
            raise SystemExit("another ms_sched2 is running")
        except (ProcessLookupError, ValueError):
            pass
    open(PID, "w").write(str(os.getpid()))
    maxp = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    th = sys.argv[2] if len(sys.argv) > 2 else "2"
    log(f"sched2 start maxp={maxp} threads={th} headroom={HEADROOM_MB}")
    procs, last_wait = {}, 0
    jobs = [(j, si) for j in ORDER for si in (2, 3)]
    while True:
        for k, p in list(procs.items()):
            if p.poll() is not None:
                log(f"sched2 END {k[0]} s{k[1]} rc={p.returncode}")
                del procs[k]
        state = {}
        for j, si in jobs:
            if os.path.exists(f"{L}/{j}_s{si}.done"):
                state[(j, si)] = "done"
            elif running(j, si):
                state[(j, si)] = "run"
            elif log_done(j, si):
                open(f"{L}/{j}_s{si}.done", "w").write(time.strftime("%FT%T") + " (sched2: from log)\n")
                log(f"sched2 DONE {j} s{si} (log has final metadata)")
                state[(j, si)] = "done"
            else:
                state[(j, si)] = "todo"
        if all(v == "done" for v in state.values()):
            break
        nrun = sum(1 for v in state.values() if v == "run")
        for (j, si), v in state.items():
            if v != "todo" or nrun >= maxp:
                continue
            am = avail_mb()
            if am - NEED[j] < HEADROOM_MB:
                if time.time() - last_wait > 1800:
                    log(f"sched2 waiting for memory: next {j} s{si} needs {NEED[j]}, MemAvailable {am}")
                    last_wait = time.time()
                break
            env = dict(os.environ, OMP_NUM_THREADS=th, MKL_NUM_THREADS=th)
            procs[(j, si)] = subprocess.Popen(["nice", "-n", "10", PY, "-u", f"{ROOT}/src/robustness_B/ms_train.py", "--cpu", j, str(si)],
                                              stdout=open(f"{L}/{j}_s{si}.log", "w"), stderr=subprocess.STDOUT, env=env)
            log(f"sched2 START {j} s{si} (MemAvailable {am})")
            time.sleep(90)
            break
        time.sleep(60)
    for si in (2, 3):
        r = subprocess.run([PY, "-u", f"{ROOT}/src/robustness_B/ms_train.py", "--cpu", "export", str(si)],
                           stdout=open(f"{L}/export_s{si}.log", "w"), stderr=subprocess.STDOUT)
        log(f"sched2 export s{si} rc={r.returncode}")
    log("sched2 ALL DONE")
    os.remove(PID)


if __name__ == "__main__":
    main()

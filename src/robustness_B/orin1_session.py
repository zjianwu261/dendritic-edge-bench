"""2026-10-30 Orin #1 locked session driver (workstation side): E-13 timing (already running, orin1_oos_time.py push at
17:28) -> BS-1 G2 rebuilds -> BS-1 G2 FP16 seeds 2-3 -> BS-1 G1 energy -> A-2 (src/t4a/orin_run7.sh).
Each stage starts only after the previous one reports done; before each start the GPU devfreq lock (624.75 MHz,
min = cur = max) is read (never set) and disk is logged.  A stage that fails is logged and the next independent stage
still runs.  After the E-13 raw data are pulled and complete, the E-13 engines are deleted from the Orin (disk).
  python orin1_session.py [from_stage]     stages: e13 g2b g2s energy a2

Release notes. Device-side session driver (needs the Orin #1; not runnable from the release alone). Paths are
results/<exp>/ (the research repository used experiments/<exp>/); ROOT = this checkout. Stage a2 (src/t4a/orin_run7.sh,
Paper A) is not part of this release.
"""
import os
import subprocess
import sys
import time

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PY = "~/anaconda3/envs/dendritic/bin/python"
sys.path.insert(0, f"{ROOT}/src/costmodel")
import orin1_oos_time as OT  # noqa: E402

RD, LOCK_HZ = OT.RD, OT.LOCK_HZ
T = lambda: time.strftime("%F %T")  # noqa: E731


def log(*a):
    print(T(), *a, flush=True)


def sh(cmd, t=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)


def wait_for(path, token, poll=120, max_h=8):
    t_end = time.time() + max_h * 3600
    while time.time() < t_end:
        try:
            txt = OT.ssh(f"cat {path} 2>/dev/null").stdout
            st = OT.lock_state()
            if st != (LOCK_HZ,) * 3:
                log("WARNING lock state", st)
            if token in txt:
                log("done:", path)
                return True
            last = txt.strip().splitlines()[-1] if txt.strip() else ""
            log("waiting", path, "|", last[:120], "| lock", st)
        except subprocess.TimeoutExpired:
            log("ssh timeout")
        time.sleep(poll)
    log("TIMEOUT waiting for", path)
    return False


def precheck():
    st = OT.lock_state()
    df = OT.ssh("df -BM / | tail -1; grep MemAvailable /proc/meminfo").stdout.strip().replace("\n", " | ")
    log("precheck lock", st, "|", df)
    return st == (LOCK_HZ,) * 3


def stage_e13():
    if not wait_for(f"~/{RD}/work_e13/progress.txt", "batch done"):
        return False
    r = sh(f"cd {ROOT}/src/costmodel && {PY} orin1_oos_time.py pull")
    log("pull e13", r.returncode, r.stdout[-300:], r.stderr[-300:])
    n = sum(1 for _ in open(f"{ROOT}/results/costmodel/orin1_oos/raw/results.jsonl"))
    log("e13 results.jsonl lines", n, "(expected 486)")
    remote_n = OT.ssh(f"wc -l < ~/{RD}/work_e13/results.jsonl").stdout.strip()
    if r.returncode == 0 and str(n) == remote_n:
        d = OT.ssh(f"du -sh ~/{RD}/work_e13/engines; rm -f ~/{RD}/work_e13/engines/*.engine; df -h / | tail -1").stdout
        log("e13 engines deleted after a complete pull:", d.replace("\n", " | "))
    else:
        log("e13 pull incomplete; engines kept")
    return True


def stage_g2b():
    if not precheck():
        log("NOT LOCKED: g2b not started"); return False
    r = sh(f"cd {ROOT}/src/robustness_B && {PY} bs1_orin.py g2_start")
    log("g2_start", r.returncode, r.stdout[-400:], r.stderr[-400:])
    if r.returncode != 0:
        return False
    ok = wait_for(f"~/{RD}/work_g2b/progress.txt", "batch done", max_h=5)
    r = sh(f"cd {ROOT}/src/robustness_B && {PY} bs1_orin.py g2_pull")
    log("g2_pull", r.returncode, r.stderr[-300:])
    return ok


def stage_g2s():
    if not precheck():
        log("NOT LOCKED: g2s not started"); return False
    r = sh(f"cd {ROOT}/src/t6 && CUDA_VISIBLE_DEVICES= {PY} device_fp16_seeds.py push")
    log("g2s push", r.returncode, r.stdout[-400:], r.stderr[-400:])
    if r.returncode != 0:
        return False
    ok = wait_for(f"~/{RD}/device/g2s/progress.txt", " done", max_h=5)
    r = sh(f"cd {ROOT}/src/t6 && CUDA_VISIBLE_DEVICES= {PY} device_fp16_seeds.py collect", t=1800)
    log("g2s collect", r.returncode, r.stdout[-1500:], r.stderr[-400:])
    return ok


def stage_energy():
    if not precheck():
        log("NOT LOCKED: energy not started"); return False
    r = sh(f"cd {ROOT}/src/robustness_B && {PY} bs1_orin.py e_start")
    log("e_start", r.returncode, r.stdout[-400:], r.stderr[-400:])
    if r.returncode != 0:
        return False
    ok = wait_for("~/dendritic-edge-bench/T6/energy_g1/out/progress.txt", "energy done", max_h=3)
    r = sh(f"cd {ROOT}/src/robustness_B && {PY} bs1_orin.py e_pull")
    log("e_pull", r.returncode, r.stderr[-300:])
    return ok


def stage_a2():
    if not precheck():
        log("NOT LOCKED: a2 not started"); return False
    if OT.ssh("test -e ~/dendritic-edge-bench/T4a/trt7 && echo EXISTS").stdout.strip() == "EXISTS":
        log("trt7 exists: a2 not restarted"); return False
    subprocess.run(["rsync", "-a", f"{ROOT}/src/t4a/trt_enh_regress_bench.py", f"{ROOT}/src/t4a/orin_run7.sh",
                    f"{OT.ORIN}:dendritic-edge-bench/T4a/bin/"], check=True)
    cmd = ("cd ~/dendritic-edge-bench/T4a && REQUIRE_LOCK_HZ=624750000 setsid nohup bash bin/orin_run7.sh "
           "> run7.log 2>&1 < /dev/null &")
    subprocess.run(["ssh", "-f", OT.ORIN, cmd], check=True, timeout=60, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    log("a2 started")
    ok = wait_for("~/dendritic-edge-bench/T4a/trt7/progress.txt", "DONE7", max_h=4)
    os.makedirs(f"{ROOT}/results/T4a/enhance/orin/raw_a2", exist_ok=True)
    r = sh(f"rsync -a --exclude '*.engine' {OT.ORIN}:dendritic-edge-bench/T4a/trt7/ "
           f"{ROOT}/results/T4a/enhance/orin/raw_a2/ && rsync -a {OT.ORIN}:dendritic-edge-bench/T4a/run7.log "
           f"{ROOT}/results/T4a/enhance/orin/raw_a2/")
    log("a2 pull", r.returncode, r.stderr[-300:])
    return ok


STAGES = [("e13", stage_e13), ("g2b", stage_g2b), ("g2s", stage_g2s), ("energy", stage_energy), ("a2", stage_a2)]

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "e13"
    names = [n for n, _ in STAGES]
    for n, f in STAGES[names.index(start):]:
        log("=== stage", n)
        try:
            log("stage", n, "result", f())
        except Exception as e:  # noqa: BLE001
            log("stage", n, "EXCEPTION", repr(e))
    log("SESSION DRIVER DONE")

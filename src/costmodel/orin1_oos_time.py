"""E-13 timing driver (after H2 = 7e8f26d2018d81319864fb845b52d2bd91dca8b1). Runs on the workstation; Orin #1 is
only reached by ssh / rsync. PROTOCOL.md A3: src/t6/orin/t6_orin.sh batch, REQUIRE_LOCK_HZ = 624750000, every build and
run under flock /tmp/dn_orin_bench.lock (inside t6_orin.sh). jetson_clocks / nvpmodel are never run; no sudo.

  python orin1_oos_time.py check    read-only lock / disk / process check (refuses to continue if not locked)
  python orin1_oos_time.py push     sha check of the local ONNX vs graphs_pre.csv, rsync to ~/RD/onnx/e13/, remote sha
                                    check, start t6_orin.sh batch (setsid) in ~/RD/work_e13 with REQUIRE_LOCK_HZ
  python orin1_oos_time.py status   progress tail + lock state
  python orin1_oos_time.py pull     rsync ~/RD/work_e13 (engines excluded) -> results/costmodel/orin1_oos/raw/

Release notes. Device-side driver (needs the Orin #1 and the E-13 ONNX graphs, which are not in the release). Paths are
results/<exp>/ (the research repository used experiments/<exp>/); ROOT = this checkout.
"""
import csv
import hashlib
import os
import subprocess
import sys

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTD = f"{ROOT}/results/costmodel/orin1_oos"
ORIN = "user@orin.local"
RD = "dendritic-edge-bench/T6"
WD = "work_e13"
LOCK_HZ = 624750000
G = "/sys/devices/gpu.0/devfreq/17000000.ga10b"


def ssh(cmd, t=60):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", ORIN, cmd], capture_output=True,
                          text=True, timeout=t)


def lock_state():
    v = ssh(f"cat {G}/min_freq {G}/cur_freq {G}/max_freq").stdout.split()
    return tuple(int(x) for x in v[:3]) if len(v) >= 3 else None


def check():
    st = lock_state()
    print("devfreq min/cur/max", st)
    print(ssh("df -h / | tail -1; grep MemAvailable /proc/meminfo; "
              "ps -eo pid,args | grep -E 'trtexec|t6_sync|t6_orin' | grep -v grep || true").stdout)
    if st != (LOCK_HZ,) * 3:
        raise SystemExit("GPU not locked at 624.75 MHz: stop (PLAN BS-1 / BS-2: do not start, report)")
    return st


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def push():
    check()
    if ssh(f"test -e ~/{RD}/{WD} && echo EXISTS").stdout.strip() == "EXISTS":
        raise SystemExit(f"~/{RD}/{WD} exists on the Orin: refusing to start a second batch")
    rows = list(csv.DictReader(open(f"{OUTD}/graphs_pre.csv")))
    files = []
    for r in rows:
        p = f"{ROOT}/{r['file'].replace('experiments/', 'results/', 1)}"  # graphs_pre.csv cites research paths
        assert sha(p) == r["onnx_sha256"], (r["graph"], "local sha256 != graphs_pre.csv")
        files.append((p, r["onnx_sha256"]))
    stage = f"{OUTD}/onnx"   # the 24 new graphs are here; the 3 existing ones are rsynced from their own paths
    subprocess.run(["ssh", ORIN, f"mkdir -p ~/{RD}/onnx/e13 ~/{RD}/bin"], check=True, timeout=60)
    subprocess.run(["rsync", "-a", "--include=*.onnx", "--exclude=*", f"{stage}/", f"{ORIN}:{RD}/onnx/e13/"], check=True)
    for p, _ in files:
        if not p.startswith(stage):
            subprocess.run(["rsync", "-a", p, f"{ORIN}:{RD}/onnx/e13/"], check=True)
    got = dict(l.split()[::-1] for l in ssh(f"cd ~/{RD}/onnx/e13 && sha256sum *.onnx", t=300).stdout.splitlines())
    for p, s in files:
        assert got.get(os.path.basename(p)) == s, (p, "remote sha256 differs")
    print(f"remote sha256 ok for {len(files)} ONNX")
    here = f"{ROOT}/src/t6/orin"
    subprocess.run(["rsync", "-a", f"{here}/t6_orin.sh", f"{here}/t6_sync_bench.py", f"{ORIN}:{RD}/bin/"], check=True)
    subprocess.run(["rsync", "-a", f"{OUTD}/onnx/orin_manifest.txt", f"{ORIN}:{RD}/onnx/e13/"], check=True)
    cmd = (f"cd ~/{RD} && mkdir -p {WD} && sed \"s#^~#$HOME#\" onnx/e13/orin_manifest.txt > {WD}/e13_manifest.txt && "
           f"REQUIRE_LOCK_HZ={LOCK_HZ} setsid nohup bash bin/t6_orin.sh batch {WD}/e13_manifest.txt $HOME/{RD}/{WD} 3 "
           f"> {WD}/e13_manifest.out 2>&1 < /dev/null & echo started")
    # 2026-10-30 17:28: the first push timed out here (capture_output keeps ssh -f's pipe open) after the batch had
    # already started; stdout/stderr now go to DEVNULL.
    subprocess.run(["ssh", "-f", ORIN, cmd], check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("started")


def status():
    print("lock", lock_state())
    print(ssh(f"tail -3 ~/{RD}/{WD}/progress.txt; ls ~/{RD}/{WD}/engines | wc -l; "
              f"wc -l < ~/{RD}/{WD}/results.jsonl 2>/dev/null; cat ~/{RD}/{WD}/failures.txt ~/{RD}/{WD}/skipped.txt "
              f"~/{RD}/{WD}/unlocked.txt 2>/dev/null | tail -5; df -h / | tail -1").stdout)


def pull():
    os.makedirs(f"{OUTD}/raw", exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "*.engine", f"{ORIN}:{RD}/{WD}/", f"{OUTD}/raw/"],
                   check=True)
    print("pulled ->", f"{OUTD}/raw/")


if __name__ == "__main__":
    {"check": check, "push": push, "status": status, "pull": pull}[sys.argv[1]]()

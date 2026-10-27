"""E-12 timing driver (PLAN.md section 3g-10 B5 step 3 + PROTOCOL_A1.md A1-1 / A1-2). Runs on the workstation;
every Pi command is one foreground ssh call (nothing is left running on the Pi).

Order (written in PROGRESS before timing): phase 1 = rounds k = 1..3 of [main 27 in P2 order + 3 anchors];
phase 2 = rounds k = 1..3 of [secondary 5]. Before every round: the A1-2 round gate (>= 90 s after the previous
round; vmstat 1 60, mean of the 'id' column over its 59 one-second samples, i.e. the first since-boot line is
dropped; 1-min load <= 2.0; mean idle >= 40 %; no process > 20 % CPU in the ps list other than garbage.service
and this measurement's own ssh session (sshd / shell / ps, whose ps %CPU is a lifetime figure of a process
a few ms old)). Gate failure: retry every 10 min, give up after 60 min in total for that gate.
Per-graph gate: the 'before' env block; if a non-exempt process > 20 % CPU, the round is aborted, its rows go to
raw/results_aborted.jsonl and the round is re-run from its first graph after the round gate passes again.

Usage: DEB_PI_SSH=<user>@<pi host> python src/costmodel/pi_oos_time.py   (writes results/costmodel/pi_oos/raw/)
The released raw/ already exists, so the driver refuses to run on the release tree; point it at a fresh copy.
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pi_oos  # noqa: E402

R = os.environ.get("DEB_PI_SSH", "pi@raspberrypi.local")  # ssh target of the Raspberry Pi 4B
RD = "dendritic-edge-bench/pi_oos"
RAW = pi_oos.RAW
PY = "~/venvs/dendritic/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", R]

SELF = ('S=$$; P1=$PPID; P2=$(ps -o ppid= -p $PPID | tr -d " "); echo "exempt_pids=$S $P1 $P2"; ')
PS = "ps -eo pcpu,psr,pid,user,comm --sort=-pcpu | head -8"
GARB = ('P=$(systemctl show -p MainPID --value garbage.service); '
        'echo garbage.service pid=$P active=$(systemctl is-active garbage.service) '
        'psr=$( [ "$P" != 0 ] && ps -o psr= -p $P) pcpu=$( [ "$P" != 0 ] && ps -o pcpu= -p $P); ')


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ssh(cmd, timeout=1800, check=True):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {cmd}\n{r.stderr}")
    return r


def append(fn, text):
    with open(f"{RAW}/{fn}", "a") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def busy(text):
    """Processes > 20 % CPU in a ps block that are neither garbage.service nor this session's own processes."""
    ex = set()
    m = re.search(r"exempt_pids=([\d ]+)", text)
    if m:
        ex = set(m.group(1).split())
    g = re.search(r"garbage.service pid=(\d+)", text)
    if g and g.group(1) != "0":
        ex.add(g.group(1))
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 5 and re.fullmatch(r"[\d.]+", p[0]) and p[2].isdigit():
            if float(p[0]) > 20.0 and p[2] not in ex and p[4] != "ps":
                out.append(line.strip())
    return out


def env_block(tag):
    r = ssh(f"echo '## {tag}' $(date -Iseconds); {GARB} vcgencmd measure_temp; vcgencmd measure_clock arm; "
            f"vcgencmd get_throttled; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; "
            f"echo loadavg $(cat /proc/loadavg); {SELF} {PS}", timeout=60)
    append("env_log.txt", r.stdout)
    return r.stdout


def round_gate(label):
    """A1-2 round gate; returns True when passed, False after 60 min of failures."""
    t0 = time.time()
    while True:
        r = ssh(f"echo '#### round gate {label}' $(date -Iseconds); vmstat 1 60; echo loadavg $(cat /proc/loadavg); "
                f"{GARB} {SELF} {PS}", timeout=300)
        txt = r.stdout
        rows = [l.split() for l in txt.splitlines() if re.fullmatch(r"(\s*\d+){16,18}\s*", l)]  # vmstat data lines
        idle = [int(x[14]) for x in rows[1:]]  # drop the since-boot first line -> 59 one-second samples
        mean_idle = sum(idle) / len(idle) if idle else 0.0
        load1 = float(re.search(r"loadavg (\S+)", txt).group(1))
        b = busy(txt)
        ok = load1 <= 2.0 and mean_idle >= 40.0 and not b and len(idle) >= 59
        append("gate_log.txt", txt + f"GATE {label}: load1={load1} mean_idle={mean_idle:.1f} n={len(idle)} "
                                     f"busy={b} -> {'PASS' if ok else 'FAIL'}\n")
        log(f"gate {label}: load1={load1} idle={mean_idle:.1f} busy={b} -> {'PASS' if ok else 'FAIL'}")
        if ok:
            return True
        if time.time() - t0 + 600 > 3600:
            return False
        time.sleep(600)


def run_round(graphs, k):
    """One round; returns (rows, aborted_flag)."""
    rows = []
    for g, sub in graphs:
        before = env_block(f"before {g} run {k}")
        b = busy(before)
        if b:
            append("env_log.txt", f"#### ABORT round {k} before {g}: busy={b}")
            log("ABORT", k, g, b)
            return rows, True
        f = f"onnx/{g}.onnx"
        r = ssh(f"cd ~/{RD} && taskset -c 0,2,3 {PY} bench_ort.py {f} --provider CPUExecutionProvider --threads 3 "
                f"--warmup 50 --warmup_s 2 --iters 500 --precision fp32 --device rpi4b-8gb --label {g} --run {k} "
                f"--out results_r{k}.jsonl", timeout=3600, check=False)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if r.returncode != 0 or not line:
            append("failures.txt", f"PI FAILED: {g} run {k} rc={r.returncode} stderr={r.stderr.strip()[-2000:]!r}")
            log("FAILED", g, k, r.stderr.strip()[-300:])
        else:
            append("stdout.jsonl", line[-1])
            rows.append(json.loads(line[-1]))
            log(f"r{k} {g} p50={rows[-1]['p50_ms']}")
        env_block(f"after {g} run {k}")
    return rows, False


def phase(graphs, name, last_end):
    for k in (1, 2, 3):
        while True:
            wait = 90 - (time.time() - last_end)
            if wait > 0:
                time.sleep(wait)
            if not round_gate(f"{name} round {k}"):
                return False, last_end
            rows, aborted = run_round(graphs, k)
            last_end = time.time()
            if aborted:
                for x in rows:
                    append("results_aborted.jsonl", json.dumps(x))
                ssh(f"rm -f ~/{RD}/results_r{k}.jsonl", check=False)
                continue
            for x in rows:
                append("results.jsonl", json.dumps(x))
            break
    return True, last_end


def main():
    if os.path.exists(f"{RAW}/results.jsonl"):
        raise SystemExit("raw/results.jsonl exists")
    os.makedirs(RAW, exist_ok=True)
    G = pi_oos.all_graphs()
    main_set = [(g, s) for g, s, role, grp in G if role.startswith("main") or role == "anchor"]
    sec = [(g, s) for g, s, role, grp in G if role == "secondary"]
    r = ssh("echo baseline_throttled=$(vcgencmd get_throttled | cut -d= -f2) boot=$(uptime -s)")
    m = re.search(r"baseline_throttled=(\S+) boot=(.+)", r.stdout)
    append("env_log.txt", f"## baseline throttled={m.group(1)} boot={m.group(2).strip()}")
    log("baseline", m.group(1), m.group(2))
    ssh(f"mkdir -p ~/{RD}/onnx")  # new directory created by this driver (a restart after a crash re-uses it)
    files = [pi_oos.onnx_path(g, s) for g, s, _, _ in G]
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{pi_oos.ROOT}/src/bench/bench_ort.py", f"{R}:{RD}/"], check=True)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes"] + files + [f"{R}:{RD}/onnx/"], check=True)
    loc = {os.path.basename(p): pi_oos.sha_file(p) for p in files}
    rs = ssh(f"cd ~/{RD}/onnx && sha256sum *.onnx").stdout
    rem = {l.split()[1]: l.split()[0] for l in rs.splitlines()}
    assert rem == loc, "copied ONNX sha256 mismatch"
    log("copied", len(files), "files, sha256 ok")
    t0 = time.time()
    ok, last = phase(main_set, "main", 0.0)
    if ok:
        ok2, last = phase(sec, "secondary", last)
        if not ok2:
            append("env_log.txt", "#### secondary set: round gate not passed within 60 min; not timed")
    else:
        append("env_log.txt", "#### main set: round gate not passed within 60 min; session ended")
    for k in (1, 2, 3):
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{R}:{RD}/results_r{k}.jsonl", f"{RAW}/remote_results_r{k}.jsonl"])
    log("done main_ok", ok, "elapsed_min", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()

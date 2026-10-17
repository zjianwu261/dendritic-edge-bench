"""Sequential runner for the task file $DEB_QUEUE/tasks.txt (default: $DEB_ROOT/work/queue).

Task file format (append-only; written by src/t6/make_tasks.py and the src/t4a/*select*.py scripts):
    <id> | <mem_MB> | <shell command>
    <id> | <mem_MB> | prio=<int> | <shell command>
Lines starting with '#' are comments. Without prio=, ids starting with "t4a_" get priority 100, others 0.
Pending tasks run one at a time in (priority desc, file order); the file is re-read after every task, so tasks
appended while the runner is going are picked up. A task whose id has a state file ($DEB_QUEUE/state/<id>.done,
.failed or .running) is skipped; delete that file to re-run it. Commands run through bash with cwd src/t6
(the T6 task commands use paths relative to it) and env DN_MEM_MB=<mem_MB>, which the training scripts turn
into a per-process CUDA memory fraction. Output goes to $DEB_QUEUE/logs/<id>.log.

Usage: python src/tools/run_tasks.py [--wait] [--dry] [--only PREFIX]
       python src/tools/run_tasks.py --status
  --wait   keep polling for new tasks after the file is exhausted (the select scripts append in stages)
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
Q = os.environ.get("DEB_QUEUE", f"{ROOT}/work/queue")
TASKS, STATE, LOGS = f"{Q}/tasks.txt", f"{Q}/state", f"{Q}/logs"
CWD = f"{ROOT}/src/t6"
PRIORITY_PREFIX = [("t4a_", 100)]


def default_priority(tid):
    for prefix, pr in PRIORITY_PREFIX:
        if tid.startswith(prefix):
            return pr
    return 0


def parse_tasks():
    out = []
    if not os.path.exists(TASKS):
        return out
    for ln in open(TASKS):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = [p.strip() for p in ln.split("|", 2)]
        if len(parts) != 3:
            continue
        tid, mem, cmd = parts[0], int(parts[1]), parts[2]
        prio = default_priority(tid)
        if cmd.startswith("prio=") and "|" in cmd:
            ptxt, cmd = [x.strip() for x in cmd.split("|", 1)]
            prio = int(ptxt[5:])
        out.append((tid, mem, cmd, prio))
    return sorted(out, key=lambda t: -t[3])  # stable: file order within a priority


def state_of(tid):
    for s in ("done", "failed", "running"):
        if os.path.exists(f"{STATE}/{tid}.{s}"):
            return s
    return None


def write_state(tid, s, info):
    for old in ("running", "done", "failed"):
        p = f"{STATE}/{tid}.{old}"
        if old != s and os.path.exists(p):
            os.remove(p)
    with open(f"{STATE}/{tid}.{s}", "w") as f:
        json.dump(info, f)


def run_one(tid, mem, cmd):
    info = {"id": tid, "mem": mem, "cmd": cmd, "start": time.strftime("%Y-%m-%d %H:%M:%S"), "t_start": time.time()}
    write_state(tid, "running", info)
    env = dict(os.environ, DN_MEM_MB=str(mem), PYTHONUNBUFFERED="1")
    print(f"{info['start']} START {tid}", flush=True)
    with open(f"{LOGS}/{tid}.log", "a") as lf:
        rc = subprocess.call(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, cwd=CWD, env=env,
                             executable="/bin/bash")
    info.update(rc=rc, t_end=time.time(), minutes=round((time.time() - info["t_start"]) / 60, 1))
    write_state(tid, "done" if rc == 0 else "failed", info)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {'DONE' if rc == 0 else 'FAILED'} {tid} rc={rc} "
          f"({info['minutes']} min)", flush=True)
    return rc


def status():
    counts = {}
    for tid, mem, cmd, prio in parse_tasks():
        s = state_of(tid) or "queued"
        counts[s] = counts.get(s, 0) + 1
        if s in ("running", "failed"):
            print(f"  {s:8s} {tid}")
    print(f"{TASKS}: {counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the pending tasks in run order and exit")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--only", default="", help="run only tasks whose id starts with this prefix")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    if a.status:
        return status()
    pending = lambda: [t for t in parse_tasks() if t[0].startswith(a.only) and state_of(t[0]) is None]
    if a.dry:
        for tid, mem, cmd, prio in pending():
            print(f"{tid} | {mem} | prio={prio} | {cmd}")
        return
    os.makedirs(STATE, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    n_fail = 0
    while True:
        todo = pending()
        if not todo:
            if not a.wait:
                break
            time.sleep(a.poll)
            continue
        tid, mem, cmd, _ = todo[0]
        n_fail += run_one(tid, mem, cmd) != 0
    print(f"no pending tasks; {n_fail} failed in this session")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

"""Download a subset of the SACSoN/HuRoN rosbags (MIT license; https://rail.eecs.berkeley.edu/datasets/huron/).

All sessions of cory1, bww1, bww2, soda3; for bww8 (23.7 GB) only a few sessions (~6 GB).
Resumable: an existing file with the server's Content-Length is skipped.
Usage: python download_huron.py [--dest $DEB_DATA/raw/huron] [--workers 6]
"""
import argparse
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

DEB_DATA = os.environ.get("DEB_DATA", os.path.join(os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data"))

BASE = "https://rail.eecs.berkeley.edu/datasets/huron/"
BWW8_KEEP = {"Nov-17-2022-bww8", "Jan-12-2023-bww8", "Feb-03-2023-bww8-intloss", "Feb-09-2023-bww8-intloss"}


def listing(url):
    html = subprocess.run(["curl", "-sL", "--noproxy", "*", "-m", "60", url], capture_output=True, text=True).stdout
    return re.findall(r'href="([^"?/][^"]*)"', html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(DEB_DATA, "raw/huron"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = []
    for d in listing(BASE):
        if not d.endswith("/"):
            continue
        sess = d.rstrip("/")
        if "bww8" in sess and sess not in BWW8_KEEP:
            continue
        for f in listing(BASE + d):
            if f.endswith(".bag"):
                jobs.append((sess, f))
    print(f"{len(jobs)} bags", flush=True)

    def get(job):
        sess, f = job
        os.makedirs(os.path.join(a.dest, sess), exist_ok=True)
        out = os.path.join(a.dest, sess, f)
        url = f"{BASE}{sess}/{f}"
        for _ in range(5):
            r = subprocess.run(["curl", "-sfL", "--noproxy", "*", "-C", "-", "-o", out, url])
            if r.returncode in (0, 33):  # 33: range not satisfiable = already complete
                return
        print(f"FAILED {url}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(get, jobs))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()

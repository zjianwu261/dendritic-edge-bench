"""T1m board measurement, host side (after step4_flash_and_measure.sh).

Sends N Fashion-MNIST test images to the STC8A8K64D4 running uart_main.c, reads back
class / int32 logits / Timer0 cycles, and checks them bit-for-bit against the gcc build of the
same infer.c (results/T1m/build/<tag>/pc_infer, itself bit-exact with quant.ref_forward).

Usage (needs numpy and pyserial):
  python host_uart.py --tag dann_d16_s64_t1 --n 100 \
      --f_mhz <actual trimmed MHz reported by stcgal> [--port /dev/ttyUSB0]
"""
import argparse
import csv
import os
import struct
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP, load_fmnist_u8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--f_mhz", type=float, required=True, help="actual SYSclk after stcgal trim")
    a = ap.parse_args()
    import serial  # pyserial

    bdir = f"{EXP}/build/{a.tag}"
    xte, yte = load_fmnist_u8()["test"]
    x = xte[:a.n]
    # expected logits from the PC build of the same C code
    x.tofile(f"{bdir}/host_x.u8")
    subprocess.run([f"{bdir}/pc_infer", f"{bdir}/host_x.u8", f"{bdir}/host_ref.bin"], check=True)
    ref = np.fromfile(f"{bdir}/host_ref.bin", np.uint8).reshape(-1, 41)
    ref_logits = ref[:, :40].copy().view("<i4")
    ref_cls = ref[:, 40]

    s = serial.Serial(a.port, a.baud, timeout=5)
    time.sleep(0.2)
    s.reset_input_buffer()
    s.write(b"I")
    ident = s.readline().decode(errors="replace").strip()
    print("board:", ident)
    rows, n_log, n_cls = [], 0, 0
    for i in range(a.n):
        s.write(bytes([0xA5]) + x[i].tobytes())
        r = s.read(50)
        assert len(r) == 50 and r[0] == 0x5A, (i, r)
        cls = r[1]
        logits = np.frombuffer(r[2:42], "<i4")
        cyc, cyc0 = struct.unpack("<II", r[42:50])
        ok = bool((logits == ref_logits[i]).all())
        n_log += ok
        n_cls += int(cls == ref_cls[i])
        rows.append(dict(i=i, label=int(yte[i]), cls=cls, logits_match=ok, cycles=cyc - cyc0,
                         cycles_raw=cyc, overhead=cyc0, ms=(cyc - cyc0) / (a.f_mhz * 1e3)))
    out = f"{EXP}/board_{a.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c = np.array([r["cycles"] for r in rows])
    print(f"{a.tag}: logits bit-exact {n_log}/{a.n}, class {n_cls}/{a.n}, "
          f"cycles median {np.median(c):.0f} (min {c.min()}, max {c.max()}), "
          f"{np.median(c) / (a.f_mhz * 1e3):.2f} ms @ {a.f_mhz} MHz -> {out}")
    os.remove(f"{bdir}/host_x.u8")
    os.remove(f"{bdir}/host_ref.bin")


if __name__ == "__main__":
    main()

"""Board power logger for Orin #1 (T1 energy protocol).

Why jtop and not tegrastats: on Orin #1 (JetPack 5.1.5 / L4T R35.6) the INA3221 hwmon nodes
(/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/{in,curr}*_input) are root:root 0400, so a
non-root `tegrastats` prints NO VDD_* fields at all. The jtop service (root, /run/jtop.sock,
group jtop) reads the same INA3221 channels and serves them to group members. VDD_IN is jtop's
power['tot'] (name 'VDD_IN'; 'power' = instantaneous volt*curr in mW, 'avg' = running mean since
the client connected -- we only use 'power').

Each sample is stamped with time.time() on arrival, so the analysis can cut the exact inference
window that trt_energy_bench.py reports (wall-clock). Also logged per sample: the two sub-rails,
GPU devfreq cur_freq (sysfs, readable without root), tj / GPU temperature, loadavg, RAM used, and
(every ~1 s) the number of running trtexec processes (a heavy build in the window = flagged run).

Usage: python3 power_logger.py out.csv [--interval 0.1] [--duration S]   (stops on SIGTERM/SIGINT)
"""
import argparse
import os
import signal
import time

GPU_FREQ = "/sys/devices/gpu.0/devfreq/17000000.ga10b/cur_freq"


def rd(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except Exception:  # noqa: BLE001
        return ""


def n_trtexec():
    n = 0
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            if rd(f"/proc/{pid}/comm") == "trtexec":
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until SIGTERM")
    a = ap.parse_args()
    stop = [False]

    def h(*_):
        stop[0] = True
    signal.signal(signal.SIGTERM, h)
    signal.signal(signal.SIGINT, h)
    from jtop import jtop
    t_end = time.time() + a.duration if a.duration > 0 else None
    ntrt, t_trt = 0, 0.0
    with open(a.out, "w", buffering=1) as f, jtop(interval=a.interval) as j:
        f.write(f"# jtop {j.interval}s interval; columns: t_unix,vdd_in_mW,vdd_cpu_gpu_cv_mW,vdd_soc_mW,"
                "gpu_freq_hz,temp_tj_C,temp_gpu_C,load1,ram_used_MB,n_trtexec\n")
        f.write("t_unix,vdd_in_mW,vdd_cpu_gpu_cv_mW,vdd_soc_mW,gpu_freq_hz,temp_tj_C,temp_gpu_C,load1,ram_used_MB,n_trtexec\n")
        while j.ok() and not stop[0]:
            t = time.time()
            p = j.power
            rail = p.get("rail", {})
            tot = p.get("tot", {})
            tm = j.temperature
            now = time.time()
            if now - t_trt > 1.0:
                ntrt, t_trt = n_trtexec(), now
            ram = j.memory.get("RAM", {}) if hasattr(j, "memory") else {}
            f.write(",".join(str(x) for x in [
                f"{t:.3f}", tot.get("power", ""),
                rail.get("VDD_CPU_GPU_CV", {}).get("power", ""), rail.get("VDD_SOC", {}).get("power", ""),
                rd(GPU_FREQ), tm.get("tj", {}).get("temp", ""), tm.get("GPU", {}).get("temp", ""),
                rd("/proc/loadavg").split(" ")[0], round(ram.get("used", 0) / 1024.0) if ram else "",
                ntrt]) + "\n")
            if t_end and t >= t_end:
                break


if __name__ == "__main__":
    main()

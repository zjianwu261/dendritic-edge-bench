"""R-158: minimal tests for audit.g1_fair_energy_rows (G1 fair-protocol energy read from T6/g1_energy.csv).
  python -m pytest src/robustness_B/test_audit_coverage.py   (or: python src/robustness_B/test_audit_coverage.py)
"""
import csv
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit  # noqa: E402

FAIR_HDR = ["group", "kind", "family", "cfg", "orin_n_builds"]
FAIR = [["G1", "per_size", "dannG", "dannG_p4_m8_s64_sg1_lr0.001_wd0_fair", "5"],
        ["G1", "per_size", "mlp", "mlp_h128_lr0.001_wd0_fair", "5"],
        ["G1", "per_size", "cnn", "cnn_c2_4_lr0.001_wd0.01_fair", ""],          # no 5-build latency -> ignored
        ["G1", "per_size", "cnn", "cnn_c8_16_lr0.001_wd0_fair", "5"],
        ["G1", "per_size", "cnn", "cnn_c32_64_lr0.001_wd0.01_fair", "5"]]
EN = [["g1f_dannG_p4_m8_s64_sg1_lr0.001_wd0_fair_t1__bmm", "0.11447"],
      ["g1f_mlp_h128_lr0.001_wd0_fair_t1__gemm", "0.10775"],
      ["g1f_cnn_c32_64_lr0.001_wd0.01_fair_t1__conv", "0.12163"]]


def _w(path, hdr, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)


def test_rows_from_g1_energy():
    with tempfile.TemporaryDirectory() as d:
        f, e = f"{d}/fair.csv", f"{d}/en.csv"
        _w(f, FAIR_HDR, FAIR); _w(e, ["graph", "E_net_mJ"], EN)
        r = {x[0]: x for x in audit.g1_fair_energy_rows(e, f)}
        dn, ml, cn = (r[lab] for _, lab in audit.G1_FAIR_FAMILIES)
        assert dn[2:4] == ("yes", "m8 0.1145 mJ")
        assert ml[2:4] == ("yes", "h128 0.1078 mJ")                         # half-up, not float 0.1077
        assert cn[2:4] == ("partial", "c32_64 0.1216 mJ") and cn[5].endswith("not_measured: c8_16")
        assert "c2_4" not in cn[5]


def test_missing_file_reproduces_old_rows():
    with tempfile.TemporaryDirectory() as d:
        f = f"{d}/fair.csv"; _w(f, FAIR_HDR, FAIR)
        assert audit.g1_fair_energy_rows(f"{d}/absent.csv", f) == [
            ("dANN-G grouped GEMM (T6 G1 fair)", "Orin", "no", "-", "yes (5 builds)", "fair_summary orin_energy = not_measured"),
            ("MLP G1 fair (T6)", "Orin", "no", "-", "yes (5 builds)", ""),
            ("CNN G1 fair (T6)", "Orin", "no", "-", "yes (5 builds)", "")]


def test_real_data_matches_committed_csv():
    rows = audit.g1_fair_energy_rows()
    got = [l for l in open(f"{audit.OUT}/energy_coverage.csv", newline="") if "G1 fair" in l]
    assert len(rows) == len(got) == 3 and all(r[0] in g for r, g in zip(rows, got))


if __name__ == "__main__":
    for t in (test_rows_from_g1_energy, test_missing_file_reproduces_old_rows, test_real_data_matches_committed_csv):
        t(); print("ok", t.__name__)

"""E-12 (PLAN.md section 3g-10 B5) unit tests for src/costmodel/pi_oos.py.

Run: python -m unittest src/costmodel/tests/test_pi_oos.py   (or pytest src/costmodel/tests)
The two feature-recomputation tests need the exported ONNX graphs, which are not part of the release; they are
skipped without them. The list and amendment tests (synthetic logs on temporary copies) always run.
"""
import csv
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pi_oos  # noqa: E402

CM = pi_oos.CM


def _need(g, sub):
    p = pi_oos.onnx_path(g, sub)
    if not os.path.exists(p):
        raise unittest.SkipTest(f"input not in the release: {os.path.relpath(p, pi_oos.ROOT)}")
    return p


class PiOOS(unittest.TestCase):
    def test_t1_pi_predictions_reproduced(self):
        """Same prediction function on the three T1 Pi anchor graphs == predictions.csv pi pred_ms (1e-9)."""
        P = {r["graph"]: float(r["pred_ms"]) for r in csv.DictReader(open(f"{CM}/predictions.csv"))
             if r["tier"] == "pi" and r["pred_ms"]}
        FR = pi_oos.feature_rows()
        for g in ("mlp_h85_match_d32_s128", "dann_lrf_d32_s128", "active_dend_k4_cached"):
            _need(g, "T1")
            f = pi_oos.recompute(g, "T1", FR)
            self.assertAlmostEqual(pi_oos.t_hat(f["K"], f["F"], f["B"]), P[g], delta=1e-9, msg=g)

    def test_g1f_h128_features_equal_orin1_row(self):
        g = "g1f_mlp_h128_lr0.001_wd0_fair_t1__gemm"
        _need(g, "g1f")
        FR = pi_oos.feature_rows()
        f = pi_oos.recompute(g, "g1f", FR)  # asserts internally as well
        ref = FR[("orin1", g)]
        for k in pi_oos.FEAT:
            self.assertEqual(f[k], int(float(ref[k])), k)
        self.assertEqual(f["params_eff"], int(float(ref["params_eff"])))

    def test_lists_match_protocol(self):
        G = pi_oos.all_graphs()
        self.assertEqual(sum(r[2].startswith("main") for r in G), 27)
        self.assertEqual(sum(r[2] == "secondary" for r in G), 5)
        self.assertEqual(sum(r[2] == "anchor" for r in G), 3)
        self.assertEqual(len(pi_oos.PAIRS), 19)
        names = {r[0] for r in G}
        for p in pi_oos.PAIRS:
            self.assertIn(p[1], names)
            self.assertIn(p[2], names)
        pi_graphs = {r["graph"] for r in csv.DictReader(open(f"{CM}/features.csv")) if r["tier"] == "pi"}
        # PROTOCOL P2 says "36"; features.csv has 32 pi rows (T1 graphs). Only disjointness matters.
        self.assertGreater(len(pi_graphs), 0)
        self.assertFalse((names - set(pi_oos.ANCHORS)) & pi_graphs)
        self.assertTrue(set(pi_oos.ANCHORS) <= pi_graphs)


class AmendmentA1(unittest.TestCase):
    """PROTOCOL_A1.md A1-1 / A1-2 / A1-4 on synthetic logs (temporary copies; the real pi_oos/ is untouched)."""

    def test_throttled_rule(self):
        ok = pi_oos.throttled_ok
        self.assertTrue(ok("0x50000", "0x50000", False))
        self.assertFalse(ok("0x50004", "0x50000", False))   # current bit set
        self.assertFalse(ok("0x70000", "0x50000", False))   # new sticky bit during the session
        self.assertFalse(ok("0x50000", "0x50000", True))    # rebooted: strict P6 rule
        self.assertTrue(ok("0x0", "0x0", True))
        self.assertTrue(ok("0x0", "0x0", False))

    def test_verdict_void_categories(self):
        import json
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp(prefix="pi_oos_test_", dir=os.environ.get("COSTMODEL_TMP"))
        try:
            out, raw = f"{tmp}/out", f"{tmp}/out/raw"
            os.makedirs(raw)
            for fn in ("PROTOCOL.md", "graphs_pre.csv", "pairs_pre.csv", "pre.sha256", "PROTOCOL_A1.md"):
                shutil.copy(f"{pi_oos.OUTD}/{fn}", out)
            pre = {r["graph"]: r for r in csv.DictReader(open(f"{out}/graphs_pre.csv"))}
            M8, H128, M32 = pi_oos.M8, pi_oos.H128, pi_oos.M32
            THR, TWO, FAIL = "adc_h256_ffrand__gemm", "mlp_h512__gemm", "gru_h64__seq"
            env = ["## baseline throttled=0x50000 boot=2026-09-30 10:12:28\n"]
            rows, aborted = [], []
            for k in (1, 2, 3):
                for g, sub, role, grp in pi_oos.all_graphs():
                    if role == "secondary" or g == FAIL or (g == TWO and k == 3):
                        continue
                    if k == 2 and g == M8:  # an aborted attempt of round 2 (A1-2), then the re-run
                        env += [f"## before {g} run 2\n", "throttled=0x50000\n", f"## after {g} run 2\n",
                                "throttled=0x50004\n"]
                        aborted.append(dict(label=g, run=2, p50_ms=999.0))
                    env += [f"## before {g} run {k}\n", "throttled=0x50000\n", f"## after {g} run {k}\n",
                            f"throttled={'0x50001' if (g == THR and k == 2) else '0x50000'}\n"]
                    th = float(pre[g]["T_hat_ms"] or 10.0) * 1.1
                    rows.append(dict(label=g, run=k, p50_ms=th, p95_ms=th, vm_hwm_MB=50.0, iters=500, threads=3,
                                     affinity=[0, 2, 3], warmup=60, runtime="onnxruntime 1.30.0"))
            open(f"{raw}/env_log.txt", "w").writelines(env)
            open(f"{raw}/results.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows)
            open(f"{raw}/results_aborted.jsonl", "w").writelines(json.dumps(r) + "\n" for r in aborted)
            open(f"{raw}/failures.txt", "w").write(f"PI FAILED: {FAIL} run 1 (synthetic)\n")
            old = pi_oos.OUTD, pi_oos.RAW
            pi_oos.OUTD, pi_oos.RAW = out, raw
            try:
                pi_oos.verdict()
            finally:
                pi_oos.OUTD, pi_oos.RAW = old
            V = json.load(open(f"{out}/verdict.json"))
            cat = {v["pair"]: v["categories"] for v in V["void"]}
            self.assertEqual(cat["PG15"], ["throttled"])                     # adc_h256 run 2 current bit
            self.assertEqual(cat["PG16"], ["fewer than 3 valid rounds"])     # mlp_h512 only 2 rounds
            self.assertEqual(cat["PG13"], ["load or run failure"])           # gru_h64 failed
            self.assertNotIn("PG1", cat)                                     # aborted attempt not counted
            self.assertEqual(V["aborted_rows_skipped"], 1)
            self.assertEqual(V["n_void"], 3)
            self.assertEqual(V["protocol_amendment"]["sha256"], pi_oos.sha_file(f"{out}/PROTOCOL_A1.md"))
            T = {r["graph"]: r for r in csv.DictReader(open(f"{out}/timing.csv"))}
            self.assertEqual(int(T[M8]["n_runs"]), 3)
            self.assertNotIn(999.0, [float(T[M8][f"p50_run{k}"]) for k in (1, 2, 3)])
            self.assertTrue(H128 in T and M32 in T)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()

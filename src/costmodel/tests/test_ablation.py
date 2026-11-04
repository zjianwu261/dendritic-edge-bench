"""E-14 (PLAN.md section 3g-16 item 1) unit tests for src/costmodel/ablation.py.

Run: python -m unittest src/costmodel/tests/test_ablation.py   (or pytest src/costmodel/tests)
Needs only the released results/costmodel/ files (features.csv, fit_<tier>.json, pairs_pred.csv, verdict.json).
"""
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ablation as A  # noqa: E402
import fit  # noqa: E402

RUN = A.run(n_boot=0, do_loo=False)


class Ablation(unittest.TestCase):
    def test_m6_reproduces_verdict(self):
        rep = A.verify_m6(RUN)
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(rep["direction"], "39/42")
        self.assertAlmostEqual(math.exp(rep["pooled_median"]), 1.09, places=2)
        f = {t: math.exp(v) for t, v in rep["per_device_median"].items()}
        self.assertAlmostEqual(f["orin1"], 1.29, places=2)
        self.assertAlmostEqual(f["pi"], 1.25, places=2)
        self.assertAlmostEqual(f["mcu"], 1.01, places=2)
        self.assertEqual(rep["max_abs_ln_diff_rhat"], 0.0)
        self.assertEqual(rep["max_abs_ln_diff_rhat_fold_vs_blind_mcu"], 0.0)

    def test_subset_columns(self):
        self.assertEqual(A.gpu_cols("M1"), [0, 2])
        self.assertEqual(A.gpu_cols("M2"), [0, 3])
        self.assertEqual(A.gpu_cols("M5"), [0, 1, 3])
        self.assertEqual(A.gpu_cols("M6"), [0, 1, 2, 3])
        self.assertEqual(A.mcu_cols("M1"), A.mcu_cols("M3"))
        self.assertEqual(A.mcu_cols("M4"), A.mcu_cols("M6"))
        self.assertEqual(A.mcu_cols("M2"), [0])
        self.assertEqual(A.mcu_cols("M5"), [0, 3])

    def test_excluded_terms_are_zero(self):
        for s in A.SUBSETS:
            for t in A.DEV:
                for v in (A.mcu_variants(s) if t == "mcu" else (None,)):
                    th = RUN["res"][(s, t, v)]["model"]["theta"]
                    cols = A.mcu_cols(s) if t == "mcu" else A.gpu_cols(s)
                    for i in range(4):
                        if i not in cols:
                            self.assertEqual(th[i], 0.0, (s, t, v, i))
                    self.assertTrue(np.all(th >= 0.0), (s, t, v))

    def test_mcu_equivalences(self):
        R = RUN["res"]
        for a, b in (("M1", "M3"), ("M4", "M6")):
            for v in A.MCU_VARIANTS:
                ra = [p["rhat"] for p in R[(a, "mcu", v)]["pairs"]]
                rb = [p["rhat"] for p in R[(b, "mcu", v)]["pairs"]]
                self.assertEqual(ra, rb)

    def test_drop_has_no_sparse_cost(self):
        """drop: dendritic prediction equals the dense formula without gather / PRNG terms."""
        T = RUN["tiers"]["mcu"]
        m = RUN["res"][("M6", "mcu", "drop")]["model"]
        for r in T.den:
            c = float(np.dot(m["theta"], fit.mx(r)))
            self.assertAlmostEqual(A.predict_ms(T, m, r, "drop"), c / r["mhz"] / 1000.0, delta=1e-12)

    def test_fit_cols_full_matches_fit_log(self):
        T = RUN["tiers"]["orin1"]
        th, _ = A.fit_cols(T.X, T.y, [0, 1, 2, 3])
        th2, _ = fit.fit_log(T.X, T.y)
        self.assertTrue(np.array_equal(th, th2))

    def test_bootstrap_indices_deterministic_and_match_fit(self):
        """Index sequence = fit.py's: M6 pi bootstrap CI equals fit_pi.json exactly."""
        i1 = A.boot_indices(RUN["tiers"], n_boot=5)
        i2 = A.boot_indices(RUN["tiers"], n_boot=5)
        for t in A.DEV:
            for a, b in zip(i1[t], i2[t]):
                self.assertTrue(np.array_equal(a, b))
        full = A.boot_indices(RUN["tiers"])
        T = RUN["tiers"]["pi"]
        B = A.boot_constants(T, "M6", None, full["pi"])
        ci = json.load(open(f"{fit.OUT}/fit_pi.json"))["bootstrap"]["ci95"]
        for i, n in enumerate(A.GPU_NAMES):
            self.assertEqual(float(np.percentile(B[:, i], 2.5)), ci[n][0], n)
            self.assertEqual(float(np.percentile(B[:, i], 97.5)), ci[n][1], n)


if __name__ == "__main__":
    unittest.main()

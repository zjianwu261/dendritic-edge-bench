"""E-13 (PLAN.md section 3g-12 BS-2) unit tests for src/costmodel/orin1_oos.py.

Run: python -m unittest src/costmodel/tests/test_orin1_oos.py   (or pytest src/costmodel/tests)
Release notes: tests that need ONNX graphs not in the release (orin1_oos/onnx/, T6/lut/onnx/g2|g3) are skipped
without them; features.csv cites files in the research layout (experiments/<exp>/), mapped here to results/<exp>/.
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import data  # noqa: E402
import orin1_oos as E  # noqa: E402

CM = E.CM


def _release_path(cited):
    """features.csv / manifest.csv path (research layout experiments/<exp>/) -> file in this checkout."""
    rel = "results/" + cited[len("experiments/"):] if cited.startswith("experiments/") else cited
    return f"{data.ROOT}/{rel}"


def _need(p):
    if not os.path.exists(p):
        raise unittest.SkipTest(f"input not in the release: {os.path.relpath(p, data.ROOT)}")
    return p


def orin1_features():
    return {r["graph"]: r for r in csv.DictReader(open(f"{CM}/features.csv")) if r["tier"] == "orin1"}


class Prediction(unittest.TestCase):
    def test_t_hat_reproduces_e11_orin1_predictions(self):
        """The prediction function on every orin1 row of features.csv == predictions.csv pred_ms (1e-9)."""
        FR = orin1_features()
        n = 0
        for r in csv.DictReader(open(f"{CM}/predictions.csv")):
            if r["tier"] != "orin1" or not r["pred_ms"]:
                continue
            f = FR[r["graph"]]
            self.assertAlmostEqual(E.t_hat(int(f["K"]), int(f["F"]), int(f["B"])), float(r["pred_ms"]), delta=1e-9,
                                   msg=r["graph"])
            n += 1
        self.assertGreater(n, 150)

    def test_features_recomputed_on_reference_onnx(self):
        """features.graph_features on E-11 reference ONNX files == their features.csv rows."""
        FR = orin1_features()
        paths = {g: _need(_release_path(FR[g]["file"])) for g in (E.R_M8, E.R_H128G, E.R_DH25, E.R_LRU, E.R_ADC, E.C3264)}
        import features
        for g, p in paths.items():
            f = features.graph_features(p)
            for k in E.FEAT:
                self.assertEqual(int(f[k]), int(float(FR[g][k])), (g, k))

    def test_e11_orin1_pairs_reproduced(self):
        """rhat, rhat_blind, P-param, P-MAC of the E-11 pairs O15-O33 recomputed with this module's functions."""
        FR = orin1_features()
        n = 0
        for r in csv.DictReader(open(f"{CM}/pairs_pred.csv")):
            if r["tier"] != "orin1" or int(r["pair"][1:]) < 15:
                continue
            dn = {k: int(float(FR[r["dendritic"]][k])) for k in ("K", "F", "B", "F_eff", "params_eff")}
            ds = {k: int(float(FR[r["dense"]][k])) for k in ("K", "F", "B", "F_eff", "params_eff")}
            Td = float(r["T_dense_ms"])
            self.assertAlmostEqual(E.t_hat(dn["K"], dn["F"], dn["B"]) / Td, float(r["rhat"]), delta=1e-9)
            self.assertAlmostEqual(E.t_blind(dn, ds["K"]) / Td, float(r["rhat_blind"]), delta=1e-9)
            self.assertAlmostEqual(dn["params_eff"] / ds["params_eff"], float(r["rhat_P_param"]), delta=1e-9)
            self.assertAlmostEqual(dn["F_eff"] / ds["F_eff"], float(r["rhat_P_MAC"]), delta=1e-9)
            n += 1
        self.assertEqual(n, 19)


class GraphList(unittest.TestCase):
    def test_lists_match_protocol(self):
        self.assertEqual(len(E.PAIRS), 14)
        self.assertEqual([p[0] for p in E.PAIRS], [f"PG{k}" for k in range(1, 15)])
        names = {g[0] for g in E.GRAPHS} | set(E.MLP1.values())
        for p in E.PAIRS:
            self.assertIn(p[1], names)
            self.assertIn(p[2], names)
        self.assertEqual(sum(g[2] == "main_dendritic" for g in E.GRAPHS), 3 + 3 + 4)
        self.assertEqual(sum(g[2] == "anchor" for g in E.GRAPHS), 2)

    def test_new_shapes_not_in_features(self):
        """P2: no new graph's shape key, (K, F, B) triple or sha256 equals a tier = orin1 row of features.csv."""
        keys, triples, shas = E.orin1_shapes()
        self.assertEqual(len(orin1_features()), 202)
        man = {r["graph"]: r for r in csv.DictReader(open(_need(f"{E.ONNXD}/manifest.csv")))}
        import features
        new = [g for g in E.GRAPHS if g[3] != "existing"]
        self.assertEqual(len(new), 24)
        for lab, *_ in new:
            self.assertNotIn(E.shape_key(lab), keys, lab)
            f = features.graph_features(E.onnx_file(lab))
            self.assertNotIn((int(f["K"]), int(f["F"]), int(f["B"])), triples, lab)
            self.assertEqual(E.sha_file(E.onnx_file(lab)), man[lab]["onnx_sha256"])
            self.assertNotIn(man[lab]["onnx_sha256"], shas, lab)
            self.assertEqual(E.ops_string(f), orin1_features()[man[lab]["reference_graph"]]["ops"], lab)

    def test_shape_key_matches_e11_names(self):
        """The shape key identifies E-11 graphs regardless of training tags (sanity check of the P2 assertion)."""
        self.assertEqual(E.shape_key("g1f_mlp_h128_lr0.001_wd0_fair_t1__gemm"), "mlp_h128__gemm")
        self.assertEqual(E.shape_key("e13_g1_mlp_h128__gemm"), "mlp_h128__gemm")
        self.assertEqual(E.shape_key("e13_g2_dhssm_L2_n128_b4_cont__K25"), "dhssm_L2_n128_b4_cont__K25")
        self.assertEqual(E.shape_key("lru_L2_H64_N128_lr0.001_vs_t1__K50"), "lru_L2_H64_N128__K50")
        keys, _, _ = E.orin1_shapes()
        self.assertIn(E.shape_key("e13_g3_mlp_h1024__gemm"), keys)
        self.assertIn(E.shape_key("e13_g1_dannG_p4_m32_s64_sg1__bmm"), keys)


class Rules(unittest.TestCase):
    def test_params_eff_rules_reproduce_reference_shapes(self):
        """The params_eff functions used for the new graphs give features.csv params_eff on the E-11 shapes."""
        FR = orin1_features()
        cases = [("dannG", dict(m=8), E.R_M8), ("dannG", dict(m=32), "g1f_dannG_p4_m32_s64_sg1_lr0.001_wd0.01_fair_t1__bmm"),
                 ("mlp1", dict(h=128, form="gemm"), E.R_H128G), ("dhssm", dict(n=128, K=50), E.R_DH50),
                 ("dhssm", dict(n=256, K=25), "dhssm_L2_n256_b4_cont__K25"), ("dssmU", dict(n=128), E.R_DSSMU),
                 ("lru", dict(N=128), E.R_LRU), ("adc", dict(h=256), E.R_ADC), ("adc", dict(h=1024), "adc_h1024_ffrand__gemm"),
                 ("mlp3", dict(h=256), E.R_MLP3)]
        for fam, spec, ref in cases:
            _, _, pe, _ = E.build_net(fam, spec)
            self.assertEqual(pe, int(float(FR[ref]["params_eff"])), (fam, spec, ref))

    def test_dense_form_rule_reproduces_cited_cells(self):
        """A2: fastest of {Gemm, Conv} x {default, nocublas} by 5-build median (tie -> build minimum) == the E-11
        O20-O27 MLP cells (data.H128 / H256 / H512 / H1024)."""
        cells = {}
        for r in csv.DictReader(open(f"{data.LUT}/buildvar_summary.csv")):
            g = r["graph"]
            if g.startswith("g1f_mlp_"):
                h = g.split("_")[2]
                cells.setdefault(h, []).append((float(r["median_p50"]), float(r["min_p50"]), g))
        want = {"h128": data.H128, "h256": data.H256, "h512": data.H512, "h1024": data.H1024}
        for h, g in want.items():
            self.assertEqual(E.select_target(cells[h])[2], g, h)


class Verdict(unittest.TestCase):
    def test_graph_targets_and_clock_void(self):
        """Synthetic raw results: median of 3 runs per engine, median of 3 builds, faster flag; one run with a
        non-locked timed sample voids its graph ('clock'); a missing run voids its graph."""
        tmp = tempfile.mkdtemp(prefix="orin1_oos_test_", dir=os.environ.get("COSTMODEL_TMP"))
        try:
            env = f"{tmp}/env"
            os.makedirs(env)
            res = {}
            labs = [g[0] for g in E.GRAPHS]
            ok_env = "## gpu devfreq governor/cur/min/max\nnvhost_podgov\n{0}\n{0}\n{0}\n".format(E.LOCK_HZ)
            for i, lab in enumerate(labs):
                for b in E.FLAGS:
                    for k in (1, 2, 3):
                        bl = f"b{k}_{lab}"
                        for run in (1, 2, 3):
                            if lab == labs[1] and b == "nocublas" and k == 2 and run == 3:
                                continue  # missing run
                            eng = f"{bl}_{b}_abcdef012345.engine"
                            p50 = 1.0 + 0.1 * k + 0.01 * run + (0.5 if b == "default" else 0.0)
                            frac = 0.9 if (lab == labs[2] and run == 1 and k == 1 and b == "default") else 1.0
                            j = dict(label=bl, build=b, run=run, engine=eng, iters=500, precision="fp16", p50_ms=p50,
                                     gpu_freq_timed_frac_at_max=frac,
                                     gpu_freq_hz_timed_min_med_max=[E.LOCK_HZ] * 3)
                            res.setdefault((bl, b), []).append(j)
                            open(f"{env}/{eng[:-7]}_r{run}.txt", "w").write(ok_env + ok_env)
            T, bad, timing = E.graph_targets(res, [], [], env)
            self.assertAlmostEqual(T[labs[0]][0], 1.22)       # nocublas, build 2, run 2
            self.assertEqual(T[labs[0]][2], "nocublas")
            self.assertIn(labs[1], bad)
            self.assertEqual(bad[labs[1]][0][0], "fewer than 3 valid runs")
            self.assertIn(labs[2], bad)
            self.assertEqual(bad[labs[2]][0][0], "clock")
            self.assertNotIn(labs[3], bad)
            # reading on synthetic pair rows
            rows = [dict(status="ok", decisive=1, dir_ok_model=1, abs_lnerr_model=0.1, dir_ok_P_param=0,
                         abs_lnerr_P_param=1.0, dir_ok_P_MAC=0, abs_lnerr_P_MAC=1.0, dir_ok_blind=1, abs_lnerr_blind=0.3)
                    for _ in range(5)]
            self.assertEqual(E.reading(rows)["reading"], "PASS′")
            self.assertTrue(E.reading(rows[:3])["reading"].startswith("信息不足"))
            for r in rows:
                r["abs_lnerr_model"] = 0.5
                r["abs_lnerr_blind"] = 1.2
            self.assertEqual(E.reading(rows)["reading"], "DIRECTION′")
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()

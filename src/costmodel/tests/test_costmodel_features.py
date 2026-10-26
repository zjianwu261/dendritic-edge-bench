"""E-11 feature-counting unit tests (PROTOCOL.md P9: hand counts of MLP h128, DH-SRNN Loop, dANN-G m32;
plus synthetic graphs and the 8051 term counts).

Run: COSTMODEL_TMP=<scratch> python -m unittest src/costmodel/tests/test_costmodel_features.py
The hand-counted graphs need the exported ONNX graphs and the 8051 tests the SDCC builds (model.h) of the E-10 /
cnnvar selected cells; neither is part of the release (regenerate them with the export scripts and
`src/mcu/e10.py verify` / `src/mcu/cnnvar.py verify`). Tests whose input is missing are skipped; the synthetic-graph
tests always run.
"""
import glob
import os
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import features  # noqa: E402
import mcu_features  # noqa: E402

ROOT = os.environ.get("DEB_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
EXP = f"{ROOT}/results"
E10 = f"{EXP}/T1m/e10"


def _need(path):
    if not glob.glob(path):
        raise unittest.SkipTest(f"input not in the release: {os.path.relpath(path, ROOT)}")
    return path


class HandCountedGraphs(unittest.TestCase):
    """The three graphs named in P9, counted by hand from the architectures."""

    def test_mlp_h128(self):
        # 784-128-128-10, Gemm + LeakyReLU, Gemm + LeakyReLU, Gemm
        f = features.graph_features(_need(f"{EXP}/T6/lut/onnx/g1v2/g1v2_mlp_h128__gemm.onnx"))
        self.assertEqual(f["K"], 3 + 2)
        self.assertEqual(f["F"], 784 * 128 + 128 * 128 + 128 * 10)                    # 118016
        self.assertEqual(f["B"], 784 * 128 + 128 + 128 * 128 + 128 + 128 * 10 + 10)    # 118282
        self.assertEqual(f["F_eff"], f["F"])

    def test_dhsrnn_loop(self):
        # outer: Reshape, Gemm (input projection of all 1000 steps, 700 -> 64x8 branches), Reshape, Transpose,
        # Loop; body per step (22 nodes): Mul, Gather, Gemm(spk 1x64 . w_rec 64x512), Reshape, Mul, Mul, Add,
        # ReduceSum, Mul, Mul, Add, Sub, Greater, Cast, Gemm(1x64 . 64x20), Mul, Mul, Add, Softmax, Gather,
        # Mul, Add.  T = 1000.
        f = features.graph_features(_need(f"{EXP}/T1/models/dhsrnn_n64_b8_val_seqloop.onnx"))
        self.assertEqual(f["loop_trips"], [1000])
        self.assertEqual(f["K"], 5 + 1000 * 22)
        self.assertEqual(f["F"], 1000 * 700 * 512 + 1000 * (64 * 512 + 64 * 20))     # 392448000
        # w_in, b_in, w_rec, beta, omb, alpha, oma, a_ro, oma_ro, w_ro, b_ro, win(1000), the three zero initial
        # states (20, 64, 64x8) and two scalar constants of the body
        B = (700 * 512 + 512 + 64 * 512 + 512 + 512 + 64 + 64 + 20 + 20 + 64 * 20 + 20 + 1000
             + 20 + 64 + 512 + 2)
        self.assertEqual(f["B"], B)                                                    # 395770
        # masked input weights: the dense count includes the zeros; F_eff only the nonzeros
        self.assertLess(f["F_eff"], f["F"])

    def test_danng_m32(self):
        # x 784 -> 49 groups of a 4x4 patch (16 px); l1 = batched MatMul (49, 1, 16) x (49, 16, 32);
        # l2 = MatMul (1, 1, 1568) x (1, 1568, 64); out Gemm 64 -> 10.
        f = features.graph_features(
            _need(f"{EXP}/T6/lut/onnx/g1f/g1f_dannG_p4_m32_s64_sg1_lr0.001_wd0.01_fair_t1__bmm.onnx"))
        self.assertEqual(f["K"], 5 + 2 + 2 + 2 + 2 + 1)    # Reshape x5, Transpose x2, MatMul x2, Add x2, LReLU x2, Gemm
        self.assertEqual(f["F"], 49 * 16 * 32 + 1568 * 64 + 64 * 10)                  # 126080
        self.assertEqual(f["B"], 49 * 16 * 32 + 49 * 32 + 1568 * 64 + 64 + 64 * 10 + 10)  # 127722


def _save(model):
    fd, p = tempfile.mkstemp(suffix=".onnx", dir=features.SCRATCH)
    os.close(fd)
    onnx.save(model, p)
    return p


class SyntheticGraphs(unittest.TestCase):
    def test_grouped_conv_and_masked_gemm(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((8, 16)).astype(np.float32)
        W[:, :12] = 0.0                                     # 25 % nonzero
        Wc = rng.standard_normal((6, 2, 3, 3)).astype(np.float32)   # 6 out, groups=2 -> 2 in per group
        nodes = [helper.make_node("Gemm", ["x", "W"], ["h"]),
                 helper.make_node("Reshape", ["h", "shp"], ["h4"]),
                 helper.make_node("Conv", ["h4", "Wc"], ["y"], group=2, pads=[1, 1, 1, 1]),
                 helper.make_node("Identity", ["y"], ["z"])]
        g = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
                              [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 6, 4, 4])],
                              [numpy_helper.from_array(W, "W"), numpy_helper.from_array(Wc, "Wc"),
                               numpy_helper.from_array(np.array([1, 4, 2, 2], np.int64), "shp")])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        # Reshape (1,16)->(1,4,2,2) gives 4 channels, conv groups=2 -> 2 in/group; spatial 2x2 -> out 2x2
        nodes[1] = helper.make_node("Reshape", ["h", "shp"], ["h4"])
        f = features.features_from_model(m)
        self.assertEqual(f["K"], 3)                         # Identity not counted
        self.assertEqual(f["F"], 8 * 16 + 6 * 2 * 2 * (2 * 3 * 3))
        self.assertEqual(f["B"], 8 * 16 + 6 * 2 * 3 * 3)    # int64 shape constant not counted
        self.assertEqual(f["F_eff"], 8 * 4 + 6 * 2 * 2 * (2 * 3 * 3))

    def test_loop_multiplies_body(self):
        body = helper.make_graph(
            [helper.make_node("Identity", ["cond_in"], ["cond_out"]),
             helper.make_node("MatMul", ["s_in", "Wl"], ["s_mm"]),
             helper.make_node("Tanh", ["s_mm"], ["s_out"])],
            "body",
            [helper.make_tensor_value_info("it", TensorProto.INT64, []),
             helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
             helper.make_tensor_value_info("s_in", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
             helper.make_tensor_value_info("s_out", TensorProto.FLOAT, [1, 4])])
        Wl = numpy_helper.from_array(np.ones((4, 4), np.float32), "Wl")
        g = helper.make_graph(
            [helper.make_node("Loop", ["M", "c", "s0"], ["s"], body=body)], "t",
            [helper.make_tensor_value_info("s0", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 4])],
            [numpy_helper.from_array(np.array(7, np.int64), "M"), numpy_helper.from_array(np.array(True), "c"),
             Wl])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        f = features.features_from_model(m)
        self.assertEqual(f["loop_trips"], [7])
        self.assertEqual(f["K"], 1 + 7 * 2)                # Loop + 7 x (MatMul, Tanh); Identity not counted
        self.assertEqual(f["F"], 7 * 16)
        self.assertEqual(f["B"], 16)


class MCUCounts(unittest.TestCase):
    """8051 terms, hand counts from the kernels in src/mcu/c/infer.c (padding taps skipped)."""

    def _get(self, d):
        p = glob.glob(_need(f"{E10}/{d}/model.h"))
        self.assertEqual(len(p), 1, d)
        return mcu_features.mcu_features(p[0])

    def test_mlp_h48(self):
        f = self._get("build/mlp_h48_lr0.001_wd0_t1")      # 784-48-48-10
        self.assertEqual(f["F_fc"], 784 * 48 + 48 * 48 + 48 * 10)
        self.assertEqual(f["N_out"], 96)
        self.assertEqual((f["F_conv"], f["F_gather"], f["F_prng"]), (0, 0, 0))

    def test_dann_d16_s64(self):
        f = self._get("build/dann_d16_s64_lr0.003_wd0_t1")  # 64 somas x 16 dendrites x 16 synapses
        self.assertEqual(f["F_gather"], 64 * 16 * 16)
        self.assertEqual(f["F_fc"], 64 * 16 + 64 * 10)
        self.assertEqual(f["N_out"], 64 * 16 + 64)

    def test_prng_d16_s128(self):
        f = self._get("build/dann_prng_d16_s128_lr0.003_wd0.01_t1")
        self.assertEqual(f["F_prng"], 128 * 16 * 16)
        self.assertEqual(f["F_gather"], 0)
        self.assertEqual(f["F_fc"], 128 * 16 + 128 * 10)

    def test_cnn_c2_4(self):
        # conv1 3x3 s2 p1 28->14: 13x13 interior x 9 + 13 top x 6 + 13 left x 6 + corner 4 = 1681 taps/channel
        # conv2 3x3 s2 p1 14->7 : 6x6 x 9 + 6 x 6 + 6 x 6 + 4 = 400 taps per (out, in) channel pair
        f = self._get("build/cnn_c2_4_lr0.003_wd0.01_t1")
        self.assertEqual(f["F_conv"], 2 * 1681 + 4 * 2 * 400)
        self.assertEqual(f["F_fc"], 10 * 4 * 49)
        self.assertEqual(f["N_out"], 2 * 196 + 4 * 49)

    def test_cnns4_c2_4(self):
        # conv1 4x4 s4 p0: 49 outputs x 16 taps; conv2 3x3 s1 p1 on 7x7: 5x5x9 + 4 edges x 5 x 6 + 4 corners x 4 = 361
        f = self._get("cnnvar/build/cnns4_c2_4_lr0.01_wd0_t1")
        self.assertEqual(f["F_conv"], 2 * 49 * 16 + 4 * 2 * 361)
        self.assertEqual(f["F_fc"], 10 * 4 * 49)
        self.assertEqual(f["N_out"], 2 * 49 + 4 * 49)


if __name__ == "__main__":
    unittest.main()

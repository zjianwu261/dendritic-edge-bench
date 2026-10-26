# E-10 CNN-variant extension ("cnnvar"): protocol

Written 2026-10-26 08:20 +0800 by dn-exec. No cnnvar training, quantisation or evaluation had been run when this was written.
This file is not edited after the first run. Any deviation is recorded in PROGRESS.md, not here.

**Status: exploratory.** The extension was decided by the planner on 2026-10-26 (Paper B risk item), *after* the E-10 and
E-10b results were known. Its purpose is to check that the 8051 conclusion "a small CNN matches every dANN/MLP" does not
rest on one CNN architecture (the plain two-conv family c2_4 … c32_64). None of its results changes a pre-set E-10 reading.
In particular, PROTOCOL.md §7's verdict for dann_d16_s64 is computed over the plain family only, as pre-set. Any new
match is reported as exploratory.

## 1. Models (3 families, 15 architectures; no BN unless stated; ReLU after every conv)
- **(a) Pooling: `cnnpool_c{C1}_{C2}`.**
  conv3x3 s2 p1 (1->C1, 28->14) - ReLU - maxpool 2x2 (14->7) - conv3x3 s1 p1 (C1->C2, 7->7) - ReLU - maxpool 2x2 floor
  (7->3) - fc (C2*9 -> 10).
  - Widths (C1, C2): (2,4), (4,8), (8,16), (16,32), (32,64).
  - Pooling after conv2 too, because it cuts fc flash/cycles by 49/9.
- **(b) Depthwise-separable, MLPerf-Tiny DS-CNN style: `cnnds_c{C}`.**
  conv3x3 s2 p1 (1->C, 28->14) + BN - ReLU
  - [dw3x3 s2 p1 (C, 14->7) + BN - ReLU - pw1x1 (C->C) + BN - ReLU]
  - [dw3x3 s1 p1 (C, 7->7) + BN - ReLU - pw1x1 (C->C) + BN - ReLU]
  - global average pool (7x7 -> 1) - fc (C -> 10).
  - Constant width C (as in MLPerf-Tiny DS-CNN). Widths C: 4, 8, 16, 32, 64.
  - BN as in the MLPerf-Tiny reference, with conv bias=False before BN. BN is folded into the conv weights and bias
    before quantisation (eval-mode running statistics).
- **(c) Larger stride: `cnns4_c{C1}_{C2}`.**
  conv4x4 s4 p0 (1->C1, 28->7, non-overlapping) - ReLU - conv3x3 s1 p1 (C1->C2, 7->7) - ReLU - fc (C2*49 -> 10).
  - Widths (C1, C2): (2,4), (4,8), (8,16), (16,32), (32,64).
  - The "stride-2 + pool" first layer is already covered by family (a).
- **Planned flash range.** About 8 KiB (smallest of each family) to about 57 KiB (cnns4_c32_64). The models must fit in
  65,024 B. A width that does not fit is reported and dropped; no other width is substituted.
- **Init.** PyTorch default init (as SmallCNN). `torch.manual_seed(seed)` before construction.

## 2. Training: identical to PROTOCOL.md §2
- The code path is `src/mcu/e10.py train_eval`, the same function that trained every E-10/E-10b model. Only the model
  constructor and the quantiser branch are added for the new kinds.
- Fashion-MNIST /255. Validation = the last 6,000 indices of `default_rng(seed).permutation(60000)`. Batch 128, Generator(seed).
- AdamW (default betas/eps), 25 epochs, no schedule, no early stopping; final-epoch model. Seeds 1, 2, 3. CPU only.
- Grid: lr {3e-4, 1e-3, 3e-3} x wd {0, 1e-2}.
- Conditional extension: if the int8-val pick is lr 3e-3, add lr 1e-2 (both wd, 3 seeds) and re-select. Stop there.
  If 1e-2 is picked, report that the optimum is still at the edge.
- Runs are written to `experiments/T1m/e10/cnnvar/runs/` (not mixed with e10/runs).

## 3. Quantiser (deployed integer model; the bit-exact Python reference is the selection metric)
- Same conventions as `cnn_t1m.quantize_cnn`:
  - per-layer symmetric int8 weights (|w| <= 127), biases round(b / (s_in*s_w)) as int16/int32;
  - uint8 x int8 products, int32 accumulators;
  - requant t = (acc + 2^(sh-1)) >> sh, ReLU clip to 0..127;
  - shift sh = max(ceil(log2(p99.99(pre>0)/127)) - o, 0) with offset o in {0,1,2,3};
  - offsets chosen on the first 2,000 validation images, then shifts rebuilt on the full 6,000-image validation split;
  - int32 logits, first argmax.
- Max-pool acts on the requantised uint8 activations. This is exact, because ReLU-clip and max commute.
- (a) and (c) have 2 requantised layers, so their offset search is the full 4x4 grid, as for the plain family.
- (b) DS-CNN has 5 requantised conv layers plus the GAP requant. The full 4^6 grid is too costly, so the offsets are chosen
  by coordinate ascent:
  - start from all 0; 2 sweeps in layer order; for each layer try o in {0..3} with the others fixed;
  - keep a change only if it is strictly better on the 2,000 images.
- GAP (integer) = min((sum_49 * GM + 2^(GS-1)) >> GS, 127), with GS = 12 and GM = round(2^GS * 127 * 2^o / p99.99(sum)),
  clamped to [1, floor((2^31 - 1)/6223)]. The GAP real scale is s_a * 2^GS / (49*GM). The fc bias is quantised with it.
  So GAP is an average pool with a calibrated integer multiplier.
- The integer forward pass of the Python reference is evaluated with float64 BLAS matmuls of integer arrays, which is exact
  (all partial sums < 2^53). Accumulator bounds are asserted < 2^31.

## 4. Selection and reporting: identical to PROTOCOL.md §3–§4
- Primary: maximise the 3-seed mean int8 validation accuracy. Ties (to 1e-4) go to the higher fp32 val, then the smaller lr.
- Sensitivity: the fp32 val_acc_last pick.
- The test set is never used for selection.
- Reported per selected cell: lr/wd; int8 val/test mean ± sample SD (ddof 1, plus ddof 0); fp32 test; flash
  (SDCC board build, seed 1); ucsim cycles (seed 1, first 5 test images); estimated board ms.

## 5. Kernel verification and timing
- Int8 kernels (max-pool, depthwise conv, pointwise conv, GAP, 4x4 s4 conv, 3x3 s1 conv) live in the existing 8051 C/SDCC
  flow: `src/mcu/c/infer.c`, new `#elif` branches. Existing model branches are not changed.
- PC (gcc) build vs Python reference: bit-exact logits on all 10,000 test images, seeds 1, 2, 3 of every selected cell.
- ucsim (s51, -t 52): bit-exact logits on the first 5 test images, seed-1 selected build; cycles recorded.
- Board ms = ucsim cycles x 0.854 / 35.004 MHz, the CNN factor of the plain kernel. Every such value is labelled
  **estimate**. No flashing. The factor was measured on the plain-CNN kernel; new kernels have a different instruction mix,
  so the estimate error may exceed the 0.75 % seen on the plain family.

## 6. Matching and recomputation: the E-10 rule (PROTOCOL.md §6), over all CNN families
- A CNN is accuracy-matched to X if mean_CNN >= mean_X - 2.48 x max(sd_CNN, sd_X): int8 test, 3 seeds, sample SD.
  ddof-0 and fp32-val-selection variants are reported as sensitivity.
- **Smallest matched CNN** = the matched CNN with the fewest ucsim cycles of its seed-1 selected build, over the 7 plain
  CNNs (ucsim of the E-10/E-10b builds) plus the 15 new ones.
  - Time ratio = board ms (CNN) / board ms (X). Measured board ms is used where it exists, else the estimate.
  - The ucsim-only ratio and the flash share (CNN flash / X flash) are also reported.
- Recompute for every dANN/MLP row of `e10/results.csv`:
  - the smallest matched CNN over all families, its ratio and flash share;
  - the fronts F / C / 3-objective with all CNNs included.
- Then state, as exploratory, whether each existing 8051 conclusion holds:
  - the flash advantage of dANNs over matched CNNs;
  - the time-ratio ranges 0.30–0.74 (MLPs) and 0.57–1.66 (dANNs);
  - the small-dANN wins d8_s32 1.66x and d8_s64 1.46x;
  - the d16_s64 match. Its pre-set verdict is not reinterpreted; the new match is reported beside it.
- Also listed: the smallest matched new CNN per small dANN, as candidates for later board timing (needs user approval).

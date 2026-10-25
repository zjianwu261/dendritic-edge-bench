# E-10 protocol (REVIEW R-089): one training budget and one selection rule for every 8051 model in the matched-accuracy comparison

Written 2026-10-24 20:42 +0800 by dn-exec, before any E-10 training or evaluation was run.
This file is not edited after the first run. Any deviation is recorded in PROGRESS.md, not here.

## 1. Models (10 architectures)
- Non-CNN, all 5 board-measured rows of `experiments/T1m/board_summary.csv`:
  dann_d16_s64 (official masks), dann_prng_d16_s128, dann_prng_d20_s128 (PRNG masks), mlp_h48, mlp_h64.
- CNNs (plain two-conv family, `SmallCNN` in `src/t6/dann_g.py`): c2_4, c3_6, c4_8, c6_12, c8_16.
- Architecture, masks and init are unchanged:
  - dANN / MLP: `src/mcu/train.py` `Net`. Glorot init, zero bias, masks from `dann_masks` / `prng_masks` with the run's seed, masked gradients.
  - CNN: `SmallCNN`, PyTorch default init.

## 2. Training (identical for all 10)
- Fashion-MNIST, input pixel/255.
- Validation = the last 6,000 indices of `np.random.default_rng(seed).permutation(60000)`. This is the split already used by both train.py and dann_g.py. Training uses the other 54,000.
- `torch.manual_seed(seed)` before model construction. Batch order comes from a torch Generator seeded with `seed`. Batch size 128.
- Optimizer `torch.optim.AdamW(lr, weight_decay=wd)` (default betas / eps), 25 epochs, no schedule, no early stopping. The final-epoch model is used.
- Seeds 1, 2, 3. Device: CPU (the GPU is busy). One code path for all models: `src/mcu/e10.py`.
- Grid:
  - lr in {3e-4, 1e-3, 3e-3}: the E-9 grid {3e-4, 1e-3} plus one step above its top (3x).
  - wd in {0, 1e-2}, the E-9 values.
  - That is 6 configs x 3 seeds per model.
- Conditional extension, per model: if the selected lr (rule 3a) is 3e-3, add lr = 1e-2 (both wd, 3 seeds) and re-select. 1e-2 is the next half-decade step (about 3.3x), as in REVIEW R-089 item 4. Stop there. If 1e-2 is selected, report that the optimum is still at the edge.
- Note: AdamW with wd = 0 equals Adam. So the (1e-3, 0) cell is the old 8051 dANN/MLP recipe, and it is also E-9's selected CNN cell for c3_6..c8_16, retrained here on the same code path.

## 3. Selection (per model, over the grid cells)
- (a) **Primary.** Maximise the 3-seed mean **int8 validation accuracy**: the accuracy, on the 6,000 validation images, of the bit-exact integer model produced by the deployed quantiser.
  - dANN / MLP: `quant.quantize`, with shifts grid-searched on the same validation split.
  - CNN: `cnn_t1m.quantize_cnn`.
  - This is the same export as the deployed path.
  - Ties (equal to 1e-4) are broken by higher fp32 val, then by the smaller lr.
- (b) **Sensitivity (reported, not used for the headline).** 3-seed mean fp32 `val_acc_last`. This is the rule E-9 / dann_g.py `--fair` actually used: E-9 selected on fp32 val, not int8 val.
- The test set is never used for selection.

## 4. Reported per model (selected cell)
- Selected lr / wd.
- int8 val and int8 test accuracy, mean +- sd over 3 seeds. sd is the sample sd (ddof = 1); the ddof = 0 value is also given.
- fp32 test accuracy.
- Flash bytes, ucsim cycles and board ms.
- PC (gcc) bit-exactness of the seed-1 selected model on 10,000 test images.

## 5. Timing (not retrained, not re-measured)
- Board ms and ucsim cycles depend only on the architecture and kernel, so they are reused.
- Board ms are the measured values (seed 1, ~35.0 MHz) from `board_summary.csv`.
- c2_4 and c3_6 were never measured on the board. They use the estimate ucsim x 0.853 / 35.004 MHz (mean CNN board/ucsim from E-9), labelled "estimate".
- Flash is taken from a fresh SDCC build of the seed-1 selected model, to confirm that it is unchanged or nearly so (bias int16/int32 types may differ). No flashing.

## 6. Matched-accuracy rule (unchanged)
- dn-exec may not read paper/. The rule is therefore quoted from REVIEW R-084 / R-073 ("2.48 SD rule, §3.8") and the E-4 / E-9 PROGRESS entries: "95 % t interval = +-2.48·std".
- Two 3-seed means are treated as different only if |delta| > 2.48 x the larger of the two SDs. (2.48 = t_{0.975,2} / sqrt(3).)
- A CNN is **accuracy-matched** to model X if mean_CNN >= mean_X - 2.48 x max(sd_CNN, sd_X), using int8 test accuracy.
- The **smallest accuracy-matched CNN** is the matched CNN with the fewest cycles. In this family the cycle order is c2_4 < c3_6 < c4_8 < c6_12 < c8_16.
- Time ratio = board ms (CNN) / board ms (X).
- The primary computation uses sample sd (ddof = 1). The ddof = 0 variant is reported as a sensitivity check. Existing 8051 dANN/MLP tables used np.std (ddof = 0); the CNN tables used statistics.stdev (ddof = 1).
- dn-plan should check that this matches main.tex §3.8 verbatim.

## 7. Questions answered and pre-set reading
- Q1: the smallest accuracy-matched CNN to dann_d16_s64, and its time ratio.
- Q2: the same for the "best dANN" under the new protocol. Best dANN = the highest 3-seed mean int8 test accuracy among the three dANN models (official d16_s64, PRNG d16_s128, PRNG d20_s128). Q2 is also reported for every non-CNN model.
- The headline "about half the time of the smallest accuracy-matched CNN" (for d16_s64) is read as:
  - **survives** if the Q1 ratio is >= 1.8;
  - **weakened** if 1.2 <= ratio < 1.8;
  - **fails** if ratio < 1.2;
  - if no CNN up to c8_16 is matched, the result is reported as "no matched CNN in the tested family". The ratio is then > c8_16 / X; this is not "fails".
- The old-recipe numbers (current paper) are reported alongside.

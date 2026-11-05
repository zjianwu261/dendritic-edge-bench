# dendritic-edge-bench

Code and every per-run result behind the paper
**When Do Dendritic Networks Pay Off on Robot Edge Hardware? A Mapping-Aware Cost Model Tested on GPU, CPU and 8-bit MCU**
(Jianwu Zhang, Jinxu Wang, Yaxin Lu, Xiang Luo; School of Artificial Intelligence and Big Data, Hefei University;
submitted to Neurocomputing, 2026). The question: is the cost of a dendritic network on robot edge hardware set by its
parameter count, or by how its structure maps onto the device's operators?
Three dendritic families (dANN with local receptive fields, DH-SNN, active dendrites) are compared with matched dense
baselines on two Jetson Orin Nano GPUs (TensorRT FP16, synchronous end-to-end latency and measured energy; Orin #2 as a
cross-device check), a Raspberry Pi 4B (onnxruntime FP32) and an 8-bit STC8A8K64D4 8051 microcontroller (int8, SDCC,
ucsim cycles and on-board timing); hardware-native rewrites (grouped dANN, DH-SSM parallel scan, cached active-dendrite
gates) are compared under a fair, validation-selected protocol that also includes non-dendritic scan models, sparse MLPs
and small CNNs; a mapping-aware cost model, fitted on non-dendritic graphs under a protocol written before the fit,
predicts 72 held-out dendritic/dense latency ratios on the four devices; and a pre-specified functional test of a
dendritic context-gated adapter on the frozen NoMaD navigation foundation model is reported together with its corrected
re-run.

**Paths.** The paper cites result files as `experiments/<exp>/...`; in this release they are `results/<exp>/...`
(e.g. `experiments/T1/results.csv` = `results/T1/results.csv`, `experiments/T4a/v3_verdict.json` =
`results/T4a/v3_verdict.json`). `src/...` paths are unchanged. Absolute paths and host names in the raw logs were
replaced by `~/` and `orin`. The dated protocol and plan records (`results/T1m/e10/PROTOCOL.md`,
`results/T1m/e10/PROTOCOL_cnnvar.md`, `results/costmodel/PROTOCOL.md`, `docs/prespecified_rules.md`) are released
unchanged, so they still mention internal plan versions, review items and project-log entries that are not part of
the release (the role names `dn-plan` and `dn-exec` in those records are the planning and the implementation roles of
the project's internal workflow); `results/costmodel/PROTOCOL.md` is byte-identical to the file whose SHA-256 the paper cites and `fit.py`
checks.

## Contents

| Path | What |
|---|---|
| `docs/prespecified_rules.md` | Verbatim excerpts of the dated internal plan that record the pre-specified stop-loss rule (v0.10, 2026-10-03) and the rerun under the same rule (v0.12.2, 2026-10-10); the rest of the plan is not released |
| `src/t1/` | T1, deployment audit (RQ1): dANN-LRF vs MLP on Fashion-MNIST (G1), DH-SRNN / LIF-SRNN / GRU on SHD (G2), active dendrites vs MLP on permuted MNIST (G3); ONNX export, streaming-step and `Loop` exports, device accuracy, device benchmark drivers, table builder `make_results.py` |
| `src/t6/` | T6, hardware-native rewrites (RQ2): dANN-G / dANN-2:4 (`dann_g.py`), DH-SSM, LRU, diagonal SSM, GRU, LSTM on SHD (`dh_ssm.py`), cached-gate active dendrites (`ad_cache.py`); Orin latency LUT, energy join, Pareto / latency-matched tables, fair-protocol summaries; locked-clock Orin sessions of the G1 fair-protocol models (`e7_g1_orin.py`), TensorRT build-to-build variance (`buildvar_orin.py`), re-export of the G1 MLP timing graphs (`g1v2_orin.py`, `g1v2_compare.py`); on-device FP16 accuracy of the fair-protocol G2 models, seed 1 (`device_fp16_g2.py`) and seeds 2–3 (`device_fp16_seeds.py`); task lists (`make_tasks.py`); Orin-side runner (`orin/`) |
| `src/costmodel/` | Mapping-aware cost model (RQ3): graph features of the deployed ONNX graphs (`features.py`), 8051 term counts from the int8 references and an instrumented copy of `c/infer.c` (`mcu_features.py`), data assembly and the 72 test pairs (`data.py`), feature table (`build_features.py`), fit with bootstrap intervals (`fit.py`; checks the SHA-256 of `results/costmodel/PROTOCOL.md` before fitting), verdict (`verdict.py`), sensitivity analysis (`sensitivity.py`), figure (`fig_f6.py`); post-hoc out-of-sample checks E-12 on the Raspberry Pi (`pi_oos.py`, timing driver `pi_oos_time.py`) and E-13 on Orin #1 with new shapes (`orin1_oos.py`: `export`, `predict`, `manifest`, `verdict`; timing driver `orin1_oos_time.py`); post-hoc feature ablation E-14 (`ablation.py`); unit tests (`tests/`, incl. `test_pi_oos.py`, `test_orin1_oos.py`, `test_ablation.py`) |
| `src/mcu/` | T1m, 8051 kernel (RQ1): training sweep, int8 quantisation with a bit-exact fixed-point reference, C table generator, SDCC builds, ucsim cycle counts, on-chip pseudo-random receptive fields (`prng_rf.py` / `c/prng_rf.c` + unit test), small CNNs (`cnn_t1m.py`), board flashing, UART measurement and board summary (`make_board_summary.py`); equal-budget protocol E-10 / E-10b (training grid, selection, verification, ucsim re-timing and matched-accuracy report `e10.py`, Pareto fronts `fronts_e10.py`, board runs of its builds `flash_e10_board.sh`); three further CNN families under `PROTOCOL_cnnvar.md` (`cnnvar.py`, report `cnnvar_report.py`, C kernels `c/infer_cnnvar*.inc`); C sources in `c/` |
| `src/figs_journal/` | Figures of the paper (Fig. 1 `f2_decomp.py`, Fig. 2 `f3_tiers.py`, Fig. 3 `f4_g2fair.py`, Supplementary Fig. S1 `f5_v3.py`; shared `style.py`), written to `results/T6/figs_journal/`; Fig. 4 is `src/costmodel/fig_f6.py` |
| `src/t4a/` | T4a, functional benchmark (RQ3): HuRoN / RECON processing, benchmark builder, synthetic low light (`lowlight.py`) and occlusion (`bench_data.py`), zero-shot NoMaD, continual-learning methods (`cl_methods.py`, `cl_run.py`, dendritic adapter `cl_dend2.py`), selection scripts, verdict (`v3_verdict.py`), Orin deployment latency and FP16 check |
| `src/bench/` | Shared device benchmarks: TensorRT synchronous timing (`trt_sync_bench.py`), on-device evaluation, energy loop / power logger / parser, onnxruntime timing for the RPi |
| `src/robustness_B/` | Robustness audit of the headline claims (Supplementary Section S6): per-claim audit over the stored result files (`audit.py`; `--only-coverage` rebuilds only `energy_coverage.csv`, unit test `test_audit_coverage.py`), G3 AD-cache / MLP TensorRT rebuild session on the Orin (`bv_g3.py`, protocol of `src/t6/buildvar_orin.py`); multi-seed check of the naive pairs: retraining of seeds 2–3 (`ms_train.py`, CPU schedulers `ms_sched.py`, `ms_sched2.py`, first driver `ms_run_all.sh`), Orin #1 timing and energy (`ms_orin.py`, Orin side `ms_orin/ms_orin.sh`), tables (`ms_table.py`); cross-device check on Orin #2 (`orin2/`: `orin2.py`, Orin side `orin2_run.sh` and the TensorRT 10 port of the timing script `trt10_sync_bench.py`); locked-clock Orin #1 session of 2026-10-30: three TensorRT builds of the fair-protocol G2 graphs and G1 fair-protocol energy (`bs1_orin.py`), session driver (`orin1_session.py`, which also ran E-13 and the G2 FP16 seeds) |
| `src/tools/run_tasks.py` | Sequential runner for the training task file written by `src/t6/make_tasks.py` and the `src/t4a/*select*.py` scripts |
| `src/third_party/fetch.sh` | Clones the third-party code at the pinned commits of `src/third_party_VERSIONS.txt` |
| `results/T1/` | T1 tables (`results.csv`, `results_orin_builds.csv`, `energy_orin1*.csv`), model metadata (`models/*.json`), raw device JSON lines and environment logs (`raw/`), SHD validation split |
| `results/costmodel/` | Cost-model protocol `PROTOCOL.md` (in Chinese; SHA-256 `bb3f0e1d…694c148`, byte-identical to the file the fit checked), features of every timed graph (`features.csv`), fitted constants with bootstrap intervals (`fit_orin1.json`, `fit_pi.json`, `fit_orin2.json`, `fit_mcu.json`), predictions (`predictions.csv`, `pairs_pred.csv` = the 72 test ratios, `sweep_pred.csv`), verdict (`verdict.json`), figure (`f6-costmodel.pdf`), logs (`logs/`), sensitivity analysis (`e11b/`); post-hoc checks `pi_oos/` (E-12) and `orin1_oos/` (E-13: `PROTOCOL.md`, `graphs_pre.csv`, `pairs_pre.csv`, `pre.sha256`, `timing.csv`, `pairs_pred.csv`, `verdict.json`; raw Orin #1 data in `raw/`: `results.jsonl`, manifests, `progress.txt`, build logs and per-layer dumps `logs/`, per-run environment logs `env/`, tegrastats `tegra/`); post-hoc feature ablation `e14_ablation/` (E-14: `ablation.csv`, `constants.csv`, `pairs_ablation.csv`, `reproduction.json`, `summary.md` in Chinese) |
| `results/T1m/` | 8051 tables (`results.csv`, `results_all.csv`, `board_summary.csv`, per-image board logs `board_<model>_t1.csv` of the 26 board-timed builds (24 equal-budget configurations and the two stride-4 CNN variants), `board_per_image.csv`, flashing logs `logs/stcgal_*.log`), equal-budget protocol (`e10/`: `PROTOCOL.md`, per-cell results `cells.csv`, selected models `results.csv` (all 24 board-measured) and its version before the first four board runs `results_preboard.csv`, `matched.csv`, `fronts_e10.csv`, `verify.json`, `verify_seeds.json`, `ucsim.json`, grid / selection / report logs; CNN variants `PROTOCOL_cnnvar.md` and `cnnvar/`: per-run records `runs/*.json`, cells, selection, verification, ucsim, results, matched-accuracy table and fronts), small-CNN records and table (`cnn/`), per-model quantisation / build / ucsim records (`quant/`), training records (`models/`), PRNG unit-test record |
| `results/T6/` | Fair-protocol summaries, latency LUTs and the locked-clock G1 tables (`lut/`, incl. `buildvar_*.csv`, `lut_g1f.csv`, `lut_g1v2*.csv`), energy (G1 fair-protocol energy `g1_energy.csv`, raw runs `raw/energy_g1/`), device FP16 accuracy (seed 1 `device_fp16_acc.csv`; three seeds of the G2 models `fp16_acc_seeds.csv`, with the workstation FP32 references `device/ref_g2s/` and the Orin outputs and logs `device/raw_g2s/`), Pareto and latency-matched tables, per-run training records (`models/g1..g3/`), raw Orin data (`raw/`: `orin1/` LUT sessions; `orin1_e7locked/`, `orin1_g1v2/`, `orin1_bv1/`, `orin1_bv2/` locked-clock sessions with per-run environment logs in `env/`), paper figures (`figs_journal/`) |
| `results/robustness_B/` | Audit summary (`summary.md`), per-claim tables `Ca_g1_dannG_vs_mlp.csv`, `Cb_g2_scan_vs_gru.csv`, `Cc_naive_pairs.csv`, `Cd_g3_adcache.csv`, `Ce_build_spread.csv`, `Cf_8051_summary.csv`, energy coverage matrix `energy_coverage.csv`, G3 rebuild table `g3_buildvar_builds.csv` (one row per engine) and session log `bv_g3_run.log`; raw Orin data of the G3 rebuild session (`raw_orin1_bvg3/`: `results.jsonl`, manifests, `progress.txt`, per-run environment logs in `env/`); multi-seed check (`Cc_multiseed/`: tables `pairs_seed.csv`, `pairs_summary.csv`, `Cc_multiseed_summary.csv`, `accuracy_seeds.csv`, `accuracy_summary.csv`, `latency_engines.csv`, `energy_engines.csv`, `energy_runs.csv`; training records `models/s2/`, `models/s3/` (JSON); raw Orin #1 timing sessions `raw_orin1_A/`, `raw_orin1_B/`, `raw_orin1_C/` (`results.jsonl`, manifests, `onnx_info.json`, per-run environment logs) and energy sessions `raw_orin1_EB/`, `raw_orin1_EC/` (`out/`: `runs.jsonl` and per-run `_bench.json`, `_env.txt`, `_power.csv`)); Orin #2 check (`orin2/`: `env_orin2.txt`, `orin2_builds.csv`, `orin2_vs_orin1.csv`, raw `results.jsonl`, manifests and per-run environment logs in `raw/`); three TensorRT builds of the fair-protocol G2 graphs on Orin #1 (`g2_builds/`: `summary.csv`, `builds.csv`, `pairs.csv`, `spec.json`, raw `results.jsonl`, manifest, `progress.txt`, build logs `logs/`, `env/`, `tegra/`) |
| `results/T4a/` | Zero-shot, continual-learning per-run records (`cl/`), summaries, verdicts, Orin latency (`orin_latency.csv`, `dend2_orin/`), FP16 check, single-task gain check |

Not included: datasets, trained weights and ONNX/engine files (incl. the E-13 ONNX graphs `costmodel/orin1_oos/onnx/`
and the engines of the G2 rebuild session `robustness_B/g2_builds/engines/`), firmware images, TensorRT build logs and
per-layer engine dumps and tegrastats traces (except those of the 2026-10-30 Orin #1 session in
`costmodel/orin1_oos/raw/`, `robustness_B/g2_builds/` and `T6/device/raw_g2s/`), the training logs of the multi-seed
retraining, third-party code (see below).

## Paper → data → code

Numbers follow the submitted manuscript: Tables 1–7, Figs. 1–4 and Secs. 1–10 are in the main paper; extended methods
and secondary results are in the supplementary file (Supplementary Tables S1–S22 with S2a and S7a, Fig. S1, Sections
S1–S8). Result files are
relative to `results/`, scripts to `src/`.

| Paper item | Result file(s) (`results/...`) | Produced by (`src/...`) |
|---|---|---|
| Table 1 (summary matrix), Sec. 7 and Table 7 (decision rules) | the files below | — |
| Sec. 3.2 and Supplementary Section S1.1, latency protocol and build-to-build variance (bimodal builds, max/min up to 1.22, FP16 choice of MLP h1024 and CNN c32_64) | `T6/lut/buildvar_summary.csv`, `T6/lut/buildvar_builds_p1.csv`, `T6/lut/buildvar_builds_p2.csv` (one row per engine build, per-layer tactics and precision), `T6/raw/orin1_bv1/`, `T6/raw/orin1_bv2/` | `t6/buildvar_orin.py` (`run 1`, `run 2`, then `summary`, `ratios`, `cited_csv`; Orin side `t6/orin/t6_orin.sh` with `REQUIRE_LOCK_HZ`) |
| Supplementary Section S1.1, corrected development table (Identity export artifact of the random-weight G1 MLP graphs; runs keyed by ONNX hash) | `T6/lut/lut_g1v2.csv`, `T6/lut/lut_g1v2_session.csv`, `T6/lut/g1v2_anchors.csv`, `T6/raw/orin1_g1v2/`, `T6/lut/lut_g1.csv`, `T6/lut/lut_g3.csv`, `T6/pareto_g1.csv`, `T6/latmatch_g1.csv`, `T6/g1v2_diff_pareto.csv`, `T6/g1v2_diff_latmatch.csv` | `t6/g1v2_orin.py`, `t6/lut.py table g1 / g3`, `t6/pareto.py`, `t6/g1v2_compare.py` |
| Sec. 3.3 and Supplementary Section S1.4, energy protocol (idle power, run-to-run ranges) | `T1/energy_orin1.csv`, `T1/energy_orin1_runs.csv`, `T1/raw/orin1_energy/` | `bench/energy_bench.sh` → `bench/energy_chain.sh` / `energy_one.sh` / `trt_energy_bench.py` / `power_logger.py` → `bench/energy_parse.py` |
| Sec. 4, Fig. 1 (`f2-decomp.pdf`), Table 2, gate caching and Raspberry Pi paragraphs | `T1/results.csv`, `T1/energy_orin1.csv`, `T1/energy_orin1_runs.csv`; Fig. 1 also `T6/lut/buildvar_cited.csv`, `T6/energy_compare.csv`, `T6/g2_fair_summary.csv`; figure `T6/figs_journal/f2-decomp.pdf` | training `t1/dann_fmnist.py`, `t1/shd.py`, `t1/shd_val.py` + `t1/export_val_models.py`, `t1/shd_gru_grid.py`, `t1/active_dendrites.py`, `t1/ad_cached_export.py`; devices `t1/run_bench.sh` (→ `bench/run_trt_sync_jetson.sh`, `bench/trt_sync_bench.py`, `bench/bench_ort.py`), `t1/run_t1r_locked.sh`, `t1/run_t1r_rest.sh`, `t1/device_acc.py`; table `t1/make_results.py`; figure `figs_journal/f2_decomp.py` |
| Sec. 4.1, Supplementary Table S1 (Section S2.1), streaming single step | `T1/results.csv` (`*_step` rows) | `t1/shd.py` (step export), `t1/step_chain_check.py`, `t1/gru_loop_export.py`, `t1/shd_loop_export.py`; `t1/make_results.py` |
| Sec. 3.2, Supplementary Table S2 (Section S2.2), default vs nocublas builds | `T1/results_orin_builds.csv` | `t1/run_bench.sh` (BUILD = default / nocublas); `t1/make_results.py` |
| Table 4 (note), Supplementary Section S2.2 and Table S2a: three TensorRT builds of the fair-protocol G2 graphs on Orin #1 (build 1 = the engine behind Table 4, re-timed; builds 2–3 fresh; locked clock, 3 runs per engine) | `robustness_B/g2_builds/` (`summary.csv`, `builds.csv`, `pairs.csv`, `spec.json`, raw `results.jsonl`, `env/`, `tegra/`, `logs/`) | `robustness_B/bs1_orin.py` (`g2_start`, `g2_pull`, `g2_table`; Orin side `t6/orin/t6_orin.sh` with `REQUIRE_LOCK_HZ`), driven by `robustness_B/orin1_session.py` |
| Sec. 4.2, Fig. 2 (`f3-tiers.pdf`; panel (a) Orin vs RPi, see also Sec. 4), Supplementary Sections S1.3, S2.3 and S2.5: 8051 dANN / MLP / CNN first-run models, board timing, board-time estimates, board/simulator ratios, cycles per MAC | `T1m/results.csv`, `T1m/results_all.csv`, `T1m/cnn/cnn_results.csv`, `T1m/cnn/*.json`, `T1m/board_summary.csv` (24 timed models; column `build` = `first-run` or the `e10b:<cell>` build that was flashed), `T1m/board_<model>_t1.csv`, `T1m/board_per_image.csv`, `T1m/logs/stcgal_*.log`, `T1m/prng_rf_unittest.json`, `T1m/quant/`; CNN training records `T6/models/g1/cnn_c*_fair_t*.json`; Fig. 2(a) also `T1/results.csv`; figure `T6/figs_journal/f3-tiers.pdf` | `mcu/train.py` → `mcu/run_t1m.py` (`quant.py`, `gen_c.py`, `c/`) → `mcu/make_results.py`; small CNNs `t6/dann_g.py --arch cnn --fair` → `mcu/cnn_t1m.py` (`--summary` writes `cnn_results.csv`); PRNG fields `mcu/prng_rf.py`, `mcu/test_prng_rf.py`; board `mcu/step4_flash_and_measure.sh` → `mcu/host_uart.py` → `mcu/make_board_summary.py`; figure `figs_journal/f3_tiers.py` |
| Sec. 3.6 and 4.2, Table 3, Fig. 2(b) (`f3-tiers.pdf`), Supplementary Sections S1.5, S2.4 and Tables S3–S5: equal-budget 8051 protocol (E-10 for the ten models of the first comparison, E-10b extension to all 24 models that fit in flash, ucsim re-timing of the selected builds, board runs of the sixteen equal-budget seed-1 builds that differ from the first-run builds, so that all 24 rows are board-measured), accuracy-matched CNN comparison and its sensitivity variants, Pareto fronts | protocol `T1m/e10/PROTOCOL.md` (written before any run under it); Table 3 and Table S3: `T1m/e10/results.csv` (selected cell, accuracy, flash, cycles, board time, all board-measured; `results_preboard.csv` = before the first four board runs), `T1m/e10/cells.csv`; Tables S4–S5: `T1m/e10/matched.csv`; fronts (Table 3, Fig. 2(b)): `T1m/e10/fronts_e10.csv`; checks `T1m/e10/verify.json`, `T1m/e10/verify_seeds.json`, `T1m/e10/ucsim.json`; logs `T1m/e10/grid_*.log`, `select_*.log`, `report*.log`; board `T1m/board_<model>_t1.csv`, `T1m/logs/stcgal_*.log`, `T1m/board_summary.csv`, report log `T1m/e10/report_e10b12.log`; figure `T6/figs_journal/f3-tiers.pdf` | `mcu/e10.py grid` (`--lrs`, `--models`) → `e10.py select` → `e10.py verify` (PC bit-exactness, SDCC flash) → `e10.py ucsim` → `e10.py report`; fronts `mcu/fronts_e10.py`; board `mcu/flash_e10_board.sh` → `mcu/host_uart.py --bdir` → `mcu/make_board_summary.py`; figure `figs_journal/f3_tiers.py` |
| Supplementary Section S2.6 and Tables S6–S7: three further CNN families (pooled, stride-4, depthwise-separable) under the equal-budget protocol (exploratory; protocol written before their runs), board runs of `cnns4_c2_4` and `cnns4_c4_8` | protocol `T1m/e10/PROTOCOL_cnnvar.md`; `T1m/e10/cnnvar/` (`results.csv`, `matched.csv`, `fronts.csv`, `cells.csv`, per-run records `runs/*.json`, `verify*.json`, `ucsim*.json`, logs); board `T1m/board_cnns4_c2_4_t1.csv`, `T1m/board_cnns4_c4_8_t1.csv` | `mcu/cnnvar.py grid` → `select` → `verify` → `ucsim` → `report` (→ `mcu/cnnvar_report.py`; training through `mcu/e10.py`); board `mcu/flash_e10_board.sh cnns4_c2_4` / `cnns4_c4_8` |
| Sec. 5.1, Table 4, Fig. 3 (`f4-g2fair.pdf`), Supplementary Sections S3.5 and S3.6: G2 fair protocol, energy, FP16 check (seed 1), memory | `T6/g2_fair_summary.csv`, `T6/lut/lut_g2x.csv`, `T6/lut/lut_g2.csv`, `T6/energy.csv`, `T6/energy_compare.csv`, `T6/device_fp16_acc.csv`, `T6/models/g2/`, `T6/data/shd_val_speaker_split.json`; figure `T6/figs_journal/f4-g2fair.pdf` | `t6/make_tasks.py g2v` + `tools/run_tasks.py` (→ `t6/dh_ssm.py`); `t6/lut.py`, `t6/lut_extra.py`; `t6/energy_table.py`; `t6/device_fp16_g2.py`; `t6/g2_fair_summary.py`; figure `figs_journal/f4_g2fair.py` |
| Supplementary Section S3.6 and Table S7a: on-device FP16 accuracy of the fair-protocol G2 models over three seeds (Orin #1, full SHD test set) | `T6/fp16_acc_seeds.csv` (seed-1 rows copied from `T6/device_fp16_acc.csv`), `T6/device/ref_g2s/`, `T6/device/raw_g2s/` | `t6/device_fp16_seeds.py` (`ref`, `push`, `status`, `collect`; Orin side `t6/orin/t6_orin.sh`, `t6_trt_eval.py`) |
| Supplementary Section S3.6 and Table S7b, Sec. 5.2 (one sentence): seed check on the earlier fixed-settings G1 sweep (42 configurations, seeds 4–5 added; accuracy only, outside the fair protocol) | `T6/g1_seeds45.csv`, `T6/g1_seeds45_summary.md`; script `src/t6/g1_seeds45.py` (per-run JSON files of seeds 4–5 are not released, same as seeds 1–3) |
| Sec. 5.2, Supplementary Sections S3.3 and S3.6: G1 fair protocol: dANN-G vs MLP vs CNN accuracy, sparse-MLP control | `T6/fair_summary.csv`, `T6/models/g1/` (`dannG_*`, `mlp_*`, `cnn_*`, `mlpsp_*` `_fair_t{1,2,3}.json`) | `t6/dann_g.py` (task lists `t6/make_tasks.py g1 / g1r`; CNNs and sparse MLPs: see "Re-running experiments"); `t6/fair_summary.py` |
| Sec. 5.2, Table 5, Supplementary Sections S3.2 and S3.4: G1 fair-protocol Orin latencies and ratios (five builds per graph, locked clock) | `T6/lut/buildvar_cited.csv` (config, ratio and fastest-model rows), `T6/lut/buildvar_ratios.csv`, `T6/lut/buildvar_summary.csv`, `T6/fair_summary.csv` (`orin_*` columns), `T6/raw/orin1_bv1/`, `T6/raw/orin1_bv2/` | `t6/e7_g1_orin.py make` (export of the seed-1 weights), `t6/buildvar_orin.py`; `t6/fair_summary.py` |
| Sec. 3.3, Table 5 (column E) and Supplementary Table S11 (G1 rows): G1 fair-protocol energy (median-latency build of the five, 3 runs of ≥ 30 s, locked clock) | `T6/g1_energy.csv`, raw `T6/raw/energy_g1/` (`g1_energy_runs.csv`, `out/`, `spec.json`) | `robustness_B/bs1_orin.py` (`e_start`, `e_pull`, `e_table`; Orin side `ms_orin.sh energy` = the `bench/energy_one.sh` protocol); coverage rows `robustness_B/audit.py --only-coverage` |
| Supplementary Section S3.1, earlier single-build locked session (within 3 % of the build medians) and re-timed reference engines (0.998× / 1.003×) | `T6/lut/lut_g1f.csv`, `T6/lut/e7_locked_anchors.csv`, `T6/lut/e7_locked_compare.csv`, `T6/raw/orin1_e7locked/`; reference runs in `T6/raw/orin1/results.jsonl` | `t6/e7_g1_orin.py` (`auto_locked`, `table`, `anchors`, `compare`) |
| Sec. 5.2, G3 fair protocol (AD-cache vs MLPctx) and the cached-gate latency / energy pairs | `T6/fair_summary.csv`, `T6/models/g3/`, `T6/lut/lut_g3.csv`, `T6/energy_compare.csv`, `T6/energy.csv` | `t6/ad_cache.py` (task lists `t6/make_tasks.py g3 / g3ctx`); `t6/lut.py`; `t6/fair_summary.py`; `t6/energy_table.py` |
| Sec. 3.5, structured 2:4 sparsity | `T6/lut/lut_g1.csv` (`*_sp24` rows), `T6/raw/orin1/results.jsonl` | `t6/dann_g.py`, `t6/lut.py` |
| Sec. 8, Supplementary Sections S1.6, S4.1 and S4.4: v3 outcome, stop-loss rule, earlier dend2 configuration | `T4a/v3_verdict.json`, `T4a/v3_summary.csv`, `T4a/v3_mean_traj_ref.json`, `T4a/cl/*.json`, `T4a/cl/v3_dend2_selection.json` | data `t4a/run_data_pipeline.sh` (`process_huron.py`, `process_recon.py`, `build_bench.py`, `make_val_split.py`, `zero_shot.py`); runs `t4a/cl_run.py`, `t4a/cl_dend2.py`; selection `t4a/v3_select_dend2.py`; `t4a/v3_ref.py`, `t4a/v3_verdict.py` |
| Sec. 8, Supplementary Table S8 (Section S4.1), Table S9 (Section S4.2), Fig. S1 (`f5-v3.pdf`) and Section S4.6: v3fix outcome, paired seed differences, segment gains | `T4a/v3fix_verdict.json`, `T4a/v3fix_summary.csv`, `T4a/cl_summary.csv`, `T4a/cl/dend2_id_k05_mf10_v3fix_lr0.001_s*.json`; figure `T6/figs_journal/f5-v3.pdf` | `t4a/v3fix_select.py` + `tools/run_tasks.py`; `t4a/v3_verdict.py --fix`; `t4a/make_tables.py`; figure `figs_journal/f5_v3.py` |
| Sec. 8, Supplementary Section S4.3: learning-rate grid extended to 3×10⁻³ (selection file, assertion failure at 3×10⁻³, diagnostic run without the assertion) | `T4a/cl/e1_selection.json`, `T4a/cl/lora_bank_id_v3fix_r10_lr0.003_s0.json`, `T4a/cl/FAILED_dend2_id_v3fix_lr0.003_s0.json`, `T4a/cl/dend2_id_k05_mf10_v3fix_noassert_lr0.003_s0.json` | the seed-0 selection runs of `t4a/v3fix_select.py` at lr 3e-3 (`t4a/cl_dend2.py ... --assert_own 0.9`, and `--assert_own 0` for the diagnostic run; `t4a/cl_run.py --method lora_bank`) |
| Sec. 8 and Supplementary Section S4.6: other methods, mean-trajectory reference, single-task gain | `T4a/zero_shot.csv`, `T4a/gain_check*.csv`, `T4a/gain_check/`, `T4a/v3_mean_traj_ref.json` | `t4a/zero_shot.py`, `t4a/gain_check.py`, `t4a/v3_ref.py`, `t4a/make_tables.py`; context features `t4a/ctx_feat_diag.py`, `t4a/ctx_feat_eval.py` |
| Sec. 3.7, Supplementary Section S6.1 and Tables S10–S11: robustness audit of the headline claims C-a to C-f (seeds, rebuilds, sensitivity, energy coverage) and the G3 rebuild session (Orin #1, 2026-10-25, 96 engines × 5 runs, locked clock) | Table S10: `robustness_B/Ca_g1_dannG_vs_mlp.csv`, `Cb_g2_scan_vs_gru.csv`, `Cc_naive_pairs.csv`, `Cd_g3_adcache.csv`, `Ce_build_spread.csv`, `Cf_8051_summary.csv`, summary `robustness_B/summary.md`; Table S11: `robustness_B/energy_coverage.csv` (G1 rows from `T6/g1_energy.csv`); G3 rebuilds: `robustness_B/g3_buildvar_builds.csv`, `robustness_B/bv_g3_run.log`, `robustness_B/raw_orin1_bvg3/`; inputs are the released T1, T1m and T6 files cited in the rows above | `robustness_B/audit.py` (reads only stored result files); `robustness_B/bv_g3.py` (`run`, then `table`; Orin side `t6/orin/t6_orin.sh` with `REQUIRE_LOCK_HZ`) |
| Sec. 3.2 and Supplementary Section S6.2, Table S12: cross-device check on Orin #2 (JetPack 6.2, TensorRT 10.7, DVFS; the same ONNX graphs, 3 builds × 2 flags × 3 runs) | `robustness_B/orin2/orin2_builds.csv`, `orin2_vs_orin1.csv`, `env_orin2.txt`, `raw/` | `robustness_B/orin2/orin2.py` (`manifest`, `push`, `start`, `pull`, `table`; Orin side `orin2_run.sh`, `trt10_sync_bench.py`) |
| Sec. 3.4 and 4, Supplementary Section S6.3 and Tables S13–S14: multi-seed check of the naive pairs (seeds 2–3 of the 17 T1 models retrained on a CPU; locked-clock Orin #1 timing of every seed on fresh builds, energy per seed) | `robustness_B/Cc_multiseed/` (`pairs_seed.csv`, `pairs_summary.csv`, `Cc_multiseed_summary.csv`, `accuracy_seeds.csv`, `accuracy_summary.csv`, `latency_engines.csv`, `energy_engines.csv`, `energy_runs.csv`, `models/`, `raw_orin1_*/`) | `robustness_B/ms_train.py` (scheduler `ms_sched2.py`), `robustness_B/ms_orin.py` (`start A/B/C`, `start_energy B/C`, `pull`), `robustness_B/ms_table.py` |
| Sec. 6, Table 6, Fig. 4 (`f6-costmodel.pdf`), Supplementary Sections S7.1–S7.6 and Tables S15–S19: mapping-aware cost model (protocol, features, constants, the 72 predictions, per-device statistics, sensitivity) | `costmodel/PROTOCOL.md`, `costmodel/features.csv`, `costmodel/fit_*.json`, `costmodel/predictions.csv`, `costmodel/pairs_pred.csv`, `costmodel/sweep_pred.csv`, `costmodel/verdict.json`, `costmodel/logs/`, `costmodel/e11b/` (`sensitivity.json`, `per_device.csv`, `variants.csv`, `pairs_pred_alt.csv`, `tab_model.tex`); inputs are the released T1, T6, T1m and `robustness_B/orin2/` tables | `costmodel/build_features.py` (needs the ONNX graphs and 8051 builds, not included) → `costmodel/fit.py` → `costmodel/verdict.py`; `costmodel/sensitivity.py`; figure `costmodel/fig_f6.py`; tests `costmodel/tests/` |
| Sec. 6, Supplementary Section S7.7 and Table S20: post-hoc out-of-sample check on the Raspberry Pi (E-12, outside the pre-specified protocol) | `costmodel/pi_oos/` (`PROTOCOL.md`, `PROTOCOL_A1.md`, `graphs_pre.csv`, `pairs_pre.csv`, `pre.sha256`, `raw/`, `timing.csv`, `pairs_pred.csv`, `verdict.json`) | `costmodel/pi_oos.py` (`predict`, `verdict`), timing `costmodel/pi_oos_time.py`; tests `costmodel/tests/test_pi_oos.py` |
| Sec. 6, Supplementary Section S7.8 and Table S21: post-hoc out-of-sample check on Orin #1 with new shapes (E-13, outside the pre-specified protocol) | `costmodel/orin1_oos/` (`PROTOCOL.md`, `graphs_pre.csv`, `pairs_pre.csv`, `pre.sha256`, `raw/`, `timing.csv`, `pairs_pred.csv`, `verdict.json`) | `costmodel/orin1_oos.py` (`export`, `predict`, `manifest`, `verdict`), timing `costmodel/orin1_oos_time.py`; tests `costmodel/tests/test_orin1_oos.py` |
| Sec. 6, Supplementary Section S7.9 and Table S21a: post-hoc feature ablation (E-14, outside the pre-specified protocol) | `costmodel/e14_ablation/` (`ablation.csv`, `constants.csv`, `pairs_ablation.csv`, `reproduction.json`, `summary.md`) | `costmodel/ablation.py`; tests `costmodel/tests/test_ablation.py` |
| Sec. 3.2 and Supplementary Section S8, Table S22: sources of variation and what each result covers | the files above | — |
| Supplementary Section S4.5, NoMaD / adapter latency on the Orin, FP16 check | `T4a/orin_latency.csv`, `T4a/dend2_orin/dend2_latency.csv`, `T4a/fp16_check.json` | `t4a/export_onnx.py`, `t4a/export_enc_v2.py`, `t4a/export_dend2_onnx.py`; `t4a/orin_run*.sh` (`trt_pipeline_bench.py`, `trt_sync_bench.py`); `t4a/dend2_orin_table.py`, `t4a/make_tables.py`; `t4a/fp16_check.py`, `t4a/fp16_check_orin.py` |

The numbers of the paragraph "Degraded inputs" (Supplementary Section S4.6: SCI enhancer in front of NoMaD, and the single-task LoRA
reference on the three low-light tasks) come from a companion study by the same authors that is not yet published;
its code and results are not part of this release.

All summary tables in `results/` are built by the release's scripts from the released per-run files. `T6/pareto_g2.csv`
and `T6/latmatch_g2.csv` use the random-split G2 runs; the speaker-split (fair-protocol) G2 runs are summarised in
`T6/g2_fair_summary.csv`. `T6/pareto_g1.csv`, `T6/latmatch_g1.csv` and the G1 LUT (`T6/lut/lut_g1.csv`) belong to the
random-weight development table, which the paper does not use for its G1 conclusions (those rest on Table 5).
A first session of the G1 fair-protocol graphs ran with the GPU clock not locked; it is void and its data are not
included. Rebuilding `lut_<task>.csv` with `t6/lut.py table` needs the exported graphs in `results/T6/lut/onnx/<task>/`
(their hashes select the runs; every LUT row carries `onnx_sha256`). `t6/buildvar_orin.py table` needs the TensorRT
per-layer dumps, which are not included; its result is `buildvar_builds_p*.csv`, from which `summary`, `ratios` and
`cited_csv` rebuild the other build-variance tables. Likewise `robustness_B/bv_g3.py table` fills the per-layer tactic
columns of `g3_buildvar_builds.csv` only when the per-layer dumps are present; its latency columns are rebuilt from
`raw_orin1_bvg3/`. `robustness_B/audit.py` also reports two descriptive ratios from the void unlocked G1 session
(rows "clock: unlocked DVFS session" of `Ca_g1_dannG_vs_mlp.csv`); since that session is not included, a re-run
without it omits these two rows and leaves every other row unchanged.

## Setup

- Workstation: Python 3.11, PyTorch 2.11 (CUDA 12.8): `pip install -r requirements.txt` (versions as used for the paper).
  The Fashion-MNIST / permuted-MNIST training scripts also run on the CPU (`T1_DEV=cpu` for `src/t1/dann_fmnist.py`,
  `active_dendrites.py`; `T6_DEV=cpu` for `src/t6/dann_g.py`, `ad_cache.py`; `--device cpu` for `src/mcu/train.py`);
  the SHD scripts (`shd.py`, `dh_ssm.py`) need a CUDA GPU; the T4a scripts take `--device` (GPU recommended for training).
- Third-party code (not redistributed): `bash src/third_party/fetch.sh` clones it into `src/third_party/`:

| Repository | Commit | Licence | Needed for |
|---|---|---|---|
| Poirazi-Lab/dendritic_anns | fadc384 | GPL-3.0 | dANN-LRF masks (`src/t1/dann_fmnist.py`, `src/mcu/`) |
| visualnav-transformer | dca7981 | MIT | NoMaD (`src/t4a/`) |
| diffusion_policy | 5ba07ac | MIT | NoMaD UNet (`src/t4a/`) |
| DH-SNN | 79ab1fb | MIT | reference only (port in `src/t1/shd.py`) |
| nupic.research | 7a287be | AGPL-3.0 | reference only (re-implementation in `src/t1/active_dendrites.py`) |

- Environment variables (all optional): `DEB_ROOT` (repository root; default: derived from the script location),
  `DEB_DATA` (dataset folder, default `$DEB_ROOT/data`; see "Data"),
  `DEB_QUEUE` (task-runner directory, default `$DEB_ROOT/work/queue`), `PYTHON` (interpreter used in generated task
  lines and shell drivers), `DENDRITIC_ANNS` (path to `dendritic_anns/codes_`), `ORIN_HOST` / `ORIN2_HOST` / `RPI_HOST` (ssh hosts of
  the boards, default `orin` / `orin2` / `rpi`), `STCGAL` and `PORT` (8051 flashing), `S51` (ucsim `s51` binary).

## Data (download separately)

All datasets live under `$DEB_DATA` (default `data/` in the repository root, git-ignored; a symlink works):

| Data | Source (licence) | Location under `$DEB_DATA` |
|---|---|---|
| Fashion-MNIST, MNIST | torchvision, e.g. `torchvision.datasets.FashionMNIST("$DEB_DATA", download=True)` (Fashion-MNIST: MIT) | `FashionMNIST/raw/`, `MNIST/raw/` |
| Spiking Heidelberg Digits | https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/ (files https://zenkelab.org/datasets/shd_train.h5.gz, `shd_test.h5.gz`, gunzip them; CC BY 4.0) | `SHD/shd_train.h5`, `SHD/shd_test.h5` |
| RECON | https://sites.google.com/view/recon-robot/dataset (`recon_dataset.tar.gz`, 50 GB; MIT) | `raw/recon/` → `src/t4a/process_recon.py` → `gnm/recon/` |
| SACSoN / HuRoN | https://sites.google.com/view/sacson-review/huron-dataset (rosbags at https://rail.eecs.berkeley.edu/datasets/huron/, fetched by `src/t4a/download_huron.py`) | `raw/huron/` → `src/t4a/process_huron.py` → `gnm/huron/` |
| NoMaD weights | "Pre-Trained Models" link in the visualnav-transformer README (`nomad.pth`) | `gnm_weights/checkpoints/` |

`src/t4a/build_bench.py` writes the benchmark tasks to `$DEB_DATA/t4a_bench/`.

The T1 validation split of SHD (`results/T1/data/shd_val_split.json`) and the speaker-disjoint split
(`results/T6/data/shd_val_speaker_split.json`) are included. The on-device evaluation sets `eval_*.npz` are rebuilt from
the test sets by `src/t1/device_acc.py`.

## Reproducing the paper

Rebuild the tables from the released per-run files (CPU, minutes; T4a needs the third-party code for its imports):

```bash
(cd src/t1  && python make_results.py)        # results/T1/results.csv, results_orin_builds.csv
(cd src/mcu && python make_results.py && python make_board_summary.py)  
(cd src/mcu && python cnn_t1m.py --summary)    # results/T1m/cnn/cnn_results.csv
(cd src/mcu && python fronts_e10.py)           # results/T1m/e10/fronts_e10.csv
(cd src/t6  && python buildvar_orin.py summary && python buildvar_orin.py ratios && python buildvar_orin.py cited_csv)
(cd src/t6  && python e7_g1_orin.py table && python e7_g1_orin.py anchors && python e7_g1_orin.py compare)
(cd src/t6  && python g1v2_orin.py table && python g1v2_orin.py anchors)
(cd src/t6  && python fair_summary.py && python g2_fair_summary.py && python energy_table.py)
(cd src/figs_journal && for f in f2_decomp f3_tiers f4_g2fair f5_v3; do python $f.py; done)
(cd src/t4a && python make_tables.py && python v3_verdict.py && python v3_verdict.py --fix)
python src/robustness_B/audit.py              # results/robustness_B/C*.csv, energy_coverage.csv
python src/robustness_B/audit.py --only-coverage   # energy_coverage.csv only (byte-identical to the released file)
python src/robustness_B/ms_table.py           # results/robustness_B/Cc_multiseed/*.csv
python src/mcu/cnnvar_report.py               # results/T1m/e10/cnnvar/{results,matched,fronts}.csv
(cd src/costmodel && python fit.py && python verdict.py && python sensitivity.py && python fig_f6.py)
COSTMODEL_TMP=/tmp python -m unittest src/costmodel/tests/test_costmodel_features.py
python -m unittest src/costmodel/tests/test_pi_oos.py src/costmodel/tests/test_orin1_oos.py src/costmodel/tests/test_ablation.py
python src/costmodel/ablation.py              # results/costmodel/e14_ablation/ (E-14; about 1.5 min on one CPU core)
python src/robustness_B/test_audit_coverage.py
```

`fit.py` refuses to run unless `results/costmodel/PROTOCOL.md` has the recorded SHA-256; from the released
`features.csv` it reproduces `fit_*.json`, `predictions.csv`, `pairs_pred.csv`, `sweep_pred.csv` and `verdict.json`
byte for byte. `features.csv` itself is computed by `src/costmodel/build_features.py` from the exported ONNX graphs and
the SDCC builds of the 8051 models, which are not included; the unit tests that need them are skipped without them
(the synthetic-graph tests always run).

**Post-hoc Raspberry Pi check (E-12, outside the pre-specified E-11 protocol):** frozen `fit_pi.json` constants predict
19 pairs before timing on the Pi 4B, `results/costmodel/pi_oos/` (pre-specified `PROTOCOL.md`, `PROTOCOL_A1.md`,
`graphs_pre.csv`, `pairs_pre.csv`, `pre.sha256`; timing `raw/`, `timing.csv`, `pairs_pred.csv`, `verdict.json`).
`python src/costmodel/pi_oos.py verdict --out <dir>` checks `pre.sha256` and reproduces `timing.csv`, `pairs_pred.csv`
and all verdict numbers; without the research git history the commit checks are skipped and noted in `verdict.json`.

**Post-hoc Orin #1 check with new shapes (E-13, outside the pre-specified E-11 protocol):** frozen `fit_orin1.json`
constants predict 14 ratios of graph shapes never timed on Orin #1, `results/costmodel/orin1_oos/` (pre-specified
`PROTOCOL.md`, `graphs_pre.csv`, `pairs_pre.csv`, `pre.sha256`; timing `raw/`, `timing.csv`, `pairs_pred.csv`,
`verdict.json`). The reading is FAIL′: S1′ (9/9 decisive directions) and S2′ (1.33×) hold, S3′ fails only against the
mapping-blind proxy; the E-11 v1 verdict, 72 ratios and constants are unchanged. `python src/costmodel/orin1_oos.py
verdict --out <dir>` checks `pre.sha256` and reproduces all three files (commit check skipped as for E-12).

**Post-hoc feature ablation (E-14, outside the pre-specified E-11 protocol):** the cost model refitted with subsets of
its terms (M1–M6; M6 = the full model) on the same calibration graphs, the same 72 ratios and the same statistics,
with bootstrap and leave-one-calibration-graph-out ranges, and the 8051 gather / PRNG terms fitted, dropped or folded
into the dense MAC term; no new measurement. `python src/costmodel/ablation.py` checks the SHA-256 of
`results/costmodel/PROTOCOL.md`, verifies that M6 reproduces `fit_*.json`, `pairs_pred.csv` and `verdict.json`
(`reproduction.json`), and rewrites `results/costmodel/e14_ablation/`, reproducing the four CSV/JSON files byte for
byte; `summary.md` is the Chinese summary written by the same script. The E-11 v1 verdict is unchanged.

Re-running experiments:

- **T1** (one model per row): e.g. `python src/t1/dann_fmnist.py --model dann`, `--model mlp`;
  `python src/t1/shd.py --model dhsrnn`; `python src/t1/active_dendrites.py --model ad --segments 4`
  (usage line in every docstring). Device runs: `src/t1/run_bench.sh {orin|rpi|both} <model.onnx> ...`.
- **T6** (3 seeds per configuration): `python src/t6/make_tasks.py g2v` (likewise `g1`, `g1r`, `g2`, `g2base`, `g3`,
  `g3ctx`) appends task lines to `$DEB_QUEUE/tasks.txt`; `python src/tools/run_tasks.py` runs them one after another
  (`--dry` lists them, `--status` shows done / failed). Orin latency: `python src/t6/lut.py`.
  Small CNNs and sparse MLPs of the G1 fair protocol (run in `src/t6/`; one call per configuration, over
  LR in {1e-3, 3e-4}, WD in {0, 1e-2} and S in {1, 2, 3}):
  `python dann_g.py --arch cnn --c1 C1 --c2 C2 --no_export --fair --lr LR --wd WD --seed S` for
  (C1, C2) = (2, 4), (3, 6), (4, 8), (6, 12), (8, 16), (16, 32), (32, 64), and
  `python dann_g.py --arch mlpsp --h 1024 --prune {random,magnitude} --sparsity {0.8,0.9,0.95} --no_export --fair --lr LR --wd WD --seed S`.
  G1 locked-clock Orin timing: `python src/t6/e7_g1_orin.py make` exports the seed-1 weights of the selected
  configurations; `python src/t6/buildvar_orin.py run 1` and `run 2` build every graph five times per build flag and
  time each engine in five fresh processes (the GPU clock must already be locked, e.g. with `sudo jetson_clocks`; the
  scripts check the lock in every run and never change it).
- **T1m** (8051): `python src/mcu/train.py` → `S51=<s51> python src/mcu/run_t1m.py` (PC and ucsim bit-exact checks,
  SDCC builds, cycle counts) → `python src/mcu/make_results.py`; unit test `python src/mcu/test_prng_rf.py`.
  `bash src/mcu/get_ucsim.sh` unpacks ucsim 4.2.0 locally. Small CNNs on the 8051:
  `python src/mcu/cnn_t1m.py --cfgs <cfg> ...` (cfg = the selected `cnn_c*_lr*_wd*_fair` name, e.g.
  `cnn_c8_16_lr0.001_wd0_fair`), then `python src/mcu/cnn_t1m.py --summary`.
  Equal-budget protocol (`results/T1m/e10/PROTOCOL.md`): `python src/mcu/e10.py grid --jobs 8` (AdamW, learning rate
  {3e-4, 1e-3, 3e-3} × weight decay {0, 1e-2} × seeds 1–3; `--lrs 0.01 --models ...` for the models listed in
  `extension_models.json`), then `e10.py select`, `e10.py verify` (needs SDCC), `e10.py ucsim` (needs `S51`) and `e10.py report`;
  `python src/mcu/fronts_e10.py` computes the fronts. Board runs of the selected builds:
  `bash src/mcu/flash_e10_board.sh <dann_d8_s32|dann_d8_s64|cnn_c2_4|cnn_c3_6>` (erases the board).
  `make_board_summary.py`, `e10.py select` and `e10.py report` read the per-run records `results/T1m/e10/runs/*.json`
  (included, 552 runs); the per-run weights (`.npz`, 175 MB) and SDCC builds are not included and are regenerated by
  `e10.py grid` and `e10.py verify`.
  CNN variants (`results/T1m/e10/PROTOCOL_cnnvar.md`): `python src/mcu/cnnvar.py grid`, `select`, `verify`, `ucsim`,
  `report`; per-run weights are not included.
- **Robustness checks**: multi-seed retraining `python src/robustness_B/ms_train.py --cpu <job> <2|3>` for each job of
  its docstring (or `python src/robustness_B/ms_sched2.py`), then `ms_train.py export 2` / `export 3`; Orin #1 timing
  `python src/robustness_B/ms_orin.py start A` (likewise `B`, `C`, `start_energy B`, `start_energy C`; GPU clock locked
  beforehand, as for the G1 sessions), `pull`, then `ms_table.py`. Orin #2: `python src/robustness_B/orin2/orin2.py
  manifest`, `push`, `start`, `pull`, `table` (no clock lock; DVFS as found).
- **T4a**: `src/t4a/run_data_pipeline.sh` (CPU) builds the five benchmark tasks and the zero-shot table; the dend2
  configuration is chosen by `src/t4a/v3_select_dend2.py`, the v3fix learning rates by `src/t4a/v3fix_select.py
  --enqueue_select` followed by `src/t4a/v3fix_select.py`; both append their runs to the task file. One five-task run
  takes about 1–1.5 GPU hours.

Wherever a model, configuration or learning rate is selected, the selection uses validation data only; test numbers are
reported and never used to choose.

## Hardware notes

- **Jetson Orin Nano 8 GB** (JetPack 5.1.5, TensorRT 8.5.2.2, 15 W mode, FP16, batch 1). The workstation drives it
  over ssh (`ORIN_HOST`). Engines are built with `trtexec` twice (default tactics and `--tacticSources=-CUBLAS,-CUBLAS_LT`),
  for T6 / T4a with a 1 GB builder workspace, skipped below 2 GB free memory; engine names carry the first 12 hex digits of the ONNX
  SHA-256. Latency = pinned H2D copy + execute + D2H copy + synchronise, 3 fresh processes per configuration, warm-up
  ≥ 2 s and ≥ 50 iterations; the scripts never change the clocks (DVFS as deployed; supplementary runs with externally locked clocks: `src/t1/run_t1r_locked.sh`). For the G1 fair-protocol sessions the GPU was locked at 624.75 MHz with `jetson_clocks`, and `REQUIRE_LOCK_HZ=624750000` makes `src/t6/orin/t6_orin.sh` skip any timed run whose GPU devfreq min / cur / max differ from that value (`LOCK_CHECK` lines in `raw/*/env/`). Energy: `jtop` board power, net of
  idle (`src/bench/power_logger.py`). All builds and timed runs hold `flock /tmp/dn_orin_bench.lock`.
- **Second Jetson Orin Nano** (Orin #2, cross-device check only; JetPack 6.2, TensorRT 10.7.0.23, MAXN SUPER, GPU
  clock not locked (DVFS); `ORIN2_HOST`). Same ONNX graphs (SHA-256 checked against the Orin #1 records), the same
  build flags with a 1 GB workspace, three builds per flag, three fresh-process runs per engine; the scripts only read
  the clock state.
- **Raspberry Pi 4B 8 GB** (onnxruntime 1.30.0, FP32, 3 threads pinned to cores 0, 2, 3; `RPI_HOST`).
- **STC8A8K64D4** 8051 board (65,024 B flash, 8 KiB XRAM, CH340 USB-serial). Firmware: SDCC 4.2.0,
  `-mmcs51 --model-large --opt-code-speed`, F_OSC 35 MHz, UART 115200 8N1. Flash with stcgal 1.10
  (`bash src/mcu/step4_flash_and_measure.sh <tag> --yes-overwrite-board`, which erases the board; power-cycle the board
  when stcgal waits; `-t 35000` trims the internal oscillator) and pass the frequency stcgal reports to
  `src/mcu/host_uart.py --f_mhz` (needs pyserial). Cycle counts in the paper are from ucsim (`s51`); on-board timing
  uses Timer0.

## Licence

- Code (`src/`): MIT, see `LICENSE`.
- Result files (`results/`): CC BY 4.0, see `results/LICENSE`.
- Third party: no third-party code is included; `src/third_party/fetch.sh` fetches it and it keeps its own licence
  (dendritic_anns GPL-3.0, nupic.research AGPL-3.0, DH-SNN / visualnav-transformer / diffusion_policy MIT). The DH-SNN
  port and the visualnav-transformer data-processing logic are credited in `THIRD_PARTY_NOTICES.md`.
- Trained weights are not released (available from the corresponding author on request).
- The figures in `results/T6/figs_journal/` were rendered with matplotlib 3.6.3; the scripts also run with the version in
  `requirements.txt`.

## Citation

See also `CITATION.cff`.

```bibtex
@misc{zhang2026dendriticedge,
  author = {Jianwu Zhang and Jinxu Wang and Yaxin Lu and Xiang Luo},
  title  = {When Do Dendritic Networks Pay Off on Robot Edge Hardware? A Mapping-Aware Cost Model Tested on GPU, CPU and 8-bit MCU},
  year   = {2026},
  note   = {Submitted to Neurocomputing},
  url    = {https://github.com/zjianwu261/dendritic-edge-bench}
}
```

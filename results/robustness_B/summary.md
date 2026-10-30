# Paper B robustness audit (2026-10-25)

Claims C-a..C-f; per-claim CSVs in this folder, produced by src/robustness_B/audit.py; new Orin G3 rebuild session by src/robustness_B/bv_g3.py (Orin #1, 2026-10-25 22:07-23:33, 96 engines x 5 runs, 624.75 MHz locked, no energy).

| claim | seeds | builds (worst/best) | sensitivity | energy | wording |
|---|---|---|---|---|---|
| C-a G1 dANN-G m32 vs MLP (Orin) | 3 seeds acc, gaps +0.80..+1.38 pp, worst seed pairing +0.53; latency seed-1 weights | 5 builds/model; m32/h256 median 1.125 (1.12-1.14), extreme pairing 0.93-1.15; m32/h128 1.20 (1.19-1.22) | direction holds over impl/flag, p95, session, random weights | not measured | keep; add "median of 5 builds, 1.12-1.14x across builds" |
| C-b G2 scan vs GRU | 3 seeds; LRU val-best on mean only (seeds 1-2), paired t=1.4 | BS-1 (10-30): 3 builds/graph at the E-11 flag (original engine + 2 fresh), 3 runs each, 624.75 MHz; build spread 1.000-1.030 (GRU h256 1.11); GRU h768 / scan 21.0-34.1x over all 9 build pairs | K: 2.6-4.4 ms; LRU vs DH-SSM n128 1.07x lat (1.005x same K), 0.96x energy | 1 engine x 3 runs; DH-SSM K25 vs LRU K50 | "highest mean val acc, not separated over 3 seeds"; "2.6-4.4 ms depending on K" |
| C-c naive pairs | one trained seed per model | 1 engine/flag x 3 runs; 1.08-5.85 | clock DVFS vs locked changes 2.09->2.92 | Orin all T1 pairs; F2 15/18 bars; Pi none | "1.1-5.9x slower than param-matched dense (one seed each), 0.9-10.6x energy"; cite T1 table, not F2 |
| C-d G3 AD-cache vs MLP | 3 seeds acc; latency random weights | new 5 builds: h1024 1.56 (1.53-1.62), h2048 1.42 (1.38-1.47) | falls with width 2.80 (h256) .. 1.25 (h4096) | h1024/h2048 gemm only: 1.12/1.02 | "1.4-1.6x at h1024-2048 (1.38-1.62 over 5 rebuilds), falling with width"; energy "1.02-1.12x (h1024/h2048 only)" |
| C-e rebuild spread | n/a | 62 cells x 5 builds; max 1.22, median 1.04 (G1) / 1.02 (G3) | FP16 picked only by MLP h1024, CNN c32_64 | n/a | "up to 1.22x (median 1.04x)" |
| C-f 8051 | 3 seeds int8 | ucsim re-timing -0.46..+3.18%, fronts unchanged | variants 1.11/0.75/2.04 | none | use cycles/latency, never energy |

Energy measured on the Orin only. Not measured: G1 fair-protocol models (dANN-G, fair MLPs, CNNs), other G2 configs, LSTM, other AD-cache widths/forms, context-input MLP, all Pi and all 8051 models.
Open: G2 has no rebuild data; the LRU vs DH-SSM n128 tie (1.005-1.07x) is inside the G1 rebuild spread.

**C-b update (BS-1, Orin #1, 2026-10-30 22:15-23:31; g2_builds/{builds.csv, summary.csv, pairs.csv}).** Each of the 9 table-4 graphs at its E-11 target flag: build 1 = original engine (re-timed this session), builds 2-3 fresh; 3 processes x 500 iters; all 81 runs at 624.75 MHz.
Build medians (ms): DH-SSM n128 cont K50 2.761, K25 2.590, spk K25 2.645, n256 cont K25 4.180; LRU K50 2.781; dssmU K50 3.529; GRU h768 88.17, h64 17.08, h256 24.09. Build 1 / E-11 = 0.994-1.002.
LRU / DH-SSM n128 over all 3x3 build pairs: cont K50 (same K) 1.006-1.011 (median-of-builds 1.007); cont K25 1.067-1.075; spk K25 1.020-1.054. **Every pair > 1.0: LRU is slightly slower than DH-SSM n128 at every K, never faster.** Per the BS-3 rule ("LRU is not slower than DH-SSM" only if all pairs <= 1.0) that sentence must change to an interval, e.g. "LRU is 0.6-7.5% slower than DH-SSM n128 (1.006-1.075 over 3 builds each)". LRU / DH-SSM n256 0.664-0.666 (LRU faster). GRU h768 / scan models 21.0-34.1x; the C-b ordering otherwise unchanged.

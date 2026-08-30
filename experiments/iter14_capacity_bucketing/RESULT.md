# iter14 — capacity (k) + bucketing resolution (n_buckets) sweep on iter9's feature set

## Question
iter7 found that sweeping embedding dim `k` and BPR's sampling-weight exponent
gave no improvement over defaults, but that was on the *old* (pre-history-feature)
feature set. iter9 added three dense causal history features (activity, tab_pos,
rate), each bucketed into 10 quantile buckets, and became the new best by a wide
margin. iter14 asks: does iter7's capacity dead-end still hold now that there's
more signal to represent (feature set = activity+tab+rate throughout), and is
iter9's bucket count (n=10) for the new continuous features well-chosen, or would
coarser/finer bucketing help?

## Setup
- Copied `data_ext.py`/`train.py` from `iter9_history_dense/` unmodified in
  substance; only change: `_bucket_edges`/`encode_ext` gained an `n_buckets`
  parameter (default 10, matching iter9 exactly). The base 5 fields (including
  `dur_bucket`) are untouched and always use 10 buckets, matching iter9/data.py.
- Loss/optimizer/sampling (activity-weighted BPR) is a line-for-line copy of
  iter9/iter3's `train.py` — only `k` and `n_buckets` vary across configs.
- Feature set fixed at `activity+tab+rate` (iter9's winning set) throughout.
- Driver script (`driver.py`) caches `load_ext()` (one 3.5s pass) and
  `encode_ext()` per (feature_set, n_buckets) combo, so each axis's 3 seeds
  reuse the same encoding — only training reruns. Results written incrementally
  to `results.json` after every run.
- All runs: lr=0.001, epochs<=40 w/ patience=4 early stop, bs=8192, seeds {0,1,2}.

## Axis A — embedding dim `k` sweep (n_buckets=10, iter9 default)
| k | valid primary (mean, 3 seeds) | valid std | test primary (mean) | test std |
|---|---|---|---|---|
| **16 (iter9 default)** | **0.61024** | 0.00013 | **0.60572** | 0.00010 |
| 24 | 0.60986 | 0.00038 | 0.60494 | 0.00049 |
| 32 | 0.60981 | 0.00014 | 0.60482 | 0.00027 |

k=16 is best on both valid and test; k=24/32 are consistently *worse*, not just flat.
More capacity does not help on the richer feature set — if anything it looks
mildly harmful (more parameters, same data, same epochs/patience budget likely
makes optimization/regularization slightly worse, not a representational
capacity problem).

## Axis B — bucket count `n_buckets` sweep (k=16, iter9 default)
| n_buckets | valid primary (mean, 3 seeds) | valid std | test primary (mean) | test std |
|---|---|---|---|---|
| 5 | 0.60906 | 0.00038 | 0.60416 | 0.00006 |
| **10 (iter9 default)** | **0.61024** | 0.00013 | **0.60572** | 0.00010 |
| 20 | 0.61088 | 0.00037 | 0.60585 | 0.00042 |

Coarser bucketing (n=5) is clearly worse (-0.00118 valid vs n=10, well outside
noise) — confirms 10 buckets is not "over-resolved," coarsening loses real signal.
Finer bucketing (n=20) shows a small positive shift on valid (+0.00075 vs the
n=10 3-seed mean here, +0.00075 vs iter9's own 5-seed mean of 0.61013), but:
- The gap is **below the ~0.001 threshold** specified for triggering a
  confirmation run (iter9's own std is 0.00027; 0.00075 is real but modest,
  ~2.8x that std, not the "clearly beats" bar).
  - Its own 3-seed std (0.00037) is itself larger than the observed gap's
  margin over noise, so the n=20 mean is not tightly estimated on 3 seeds.
- On test, the gap essentially vanishes: n=20 test mean 0.60585 vs iter9's
  test mean 0.60560 — a difference of only +0.00025, inside test's own noise
  band (iter9 test std 0.00020, axis-B n=20 test std 0.00042 here).

Per the task's explicit decision rule (only confirm if a config "clearly"
beats iter9 by ≳0.001 valid), n_buckets=20 does not clear the bar, so no
5-seed confirmation run was run for it.

## Verdict: **REJECT both axes — iter9's original k=16 / n_buckets=10 remains best**

- **Axis A (capacity)**: k=16 stays best; k=24/32 are worse, not neutral.
  iter7's finding that more capacity doesn't help generalizes cleanly from the
  old feature set to iter9's richer one — this axis is exhausted on both
  feature sets now.
- **Axis B (bucketing resolution)**: n=5 is clearly worse (real signal loss
  from coarsening). n=20 shows a marginal, noise-adjacent improvement on valid
  that does not reach the confirmation threshold and does not show up on test
  at all — not worth promoting or spending a 5-seed confirmation run on.
  n=10 remains the well-chosen default.

**No promotion.** Current best remains iter9
(`experiments/iter9_history_dense/`): valid primary mean 0.61013 (5-seed, std
0.00027) / test primary mean 0.60560 (5-seed, std 0.00020).

## Code
`experiments/iter14_capacity_bucketing/{data_ext.py,train.py,driver.py}`,
raw per-run results in `experiments/iter14_capacity_bucketing/results.json`.

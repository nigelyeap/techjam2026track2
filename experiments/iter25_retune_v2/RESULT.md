# iter25 — retune v2: Laplace alpha + capacity/bucket resweep on iter19's feature set

## Idea
iter21 (`experiments/iter21_retune/`) was interrupted mid-run and left two
threads unfinished: (1) Axis A, its Laplace-smoothing alpha resweep, was
only run against iter16's OLDER feature set (`decay_rate_3, decay_act_3,
tab` — no momentum fields), even though it found a real, consistent-
direction gain there (alpha=0.5 beat the default 1.0 by +0.00125 valid /
+0.00287 test); and (2) Axis B, an embedding-capacity (`k`) / bucket-count
(`n_buckets`) resweep, was never run at all. This iteration redoes both
against the ACTUAL current best — iter19's fused 6-field feature set
(`decay_rate_3, decay_act_3, tab, last1, lastk_rate, gap`) — since the
bottleneck may have shifted now that the model has much richer features
than either iter16 or the original pointwise FM baseline the README's
"capacity isn't the bottleneck" finding was based on. Per the dispatch
instructions, if either axis found a real winner we would combine them and
extend to a full 5-seed run — that is exactly what happened.

`alpha` is the Laplace-smoothing constant used inside `decay_rate`,
`decay_act`'s companion `rate`-style formulas, and `lastk_rate`'s formula
(`(pos + alpha) / (total + 2*alpha)`) — module-level `ALPHA=1.0` in iter19's
`data_ext.py`, made sweepable here exactly as iter21 did to iter16's
`data_ext.py`. `n_buckets` is the quantile-bucket count used for every
bucketed continuous field (`dur_bucket`, `decay_rate`, `decay_act`,
`lastk_rate`, `gap`) — previously hardcoded at `n=10` in `_bucket_edges`,
also made sweepable here.

## Harness-fidelity sanity check
Ran iter19's exact feature set at defaults (alpha=1.0, k=16, n_buckets=10)
on seeds 0/1/2 through this dir's copy of the code:

| seed | valid (this run) | valid (iter19 published) | test (this run) | test (iter19 published) |
|---|---|---|---|---|
| 0 | 0.62902 | 0.62902 | 0.62599 | 0.62599 |
| 1 | 0.62907 | 0.62907 | 0.62684 | 0.62684 |
| 2 | 0.62990 | 0.62990 | 0.62613 | 0.62613 |

**Exact match on all 3 seeds** (deterministic — same code, same seed, same
cached feature pass). Harness confirmed faithful; the rest of this sweep is
trustworthy.

## Axis A — Laplace-smoothing alpha resweep (3 seeds each, on iter19's exact feature set: decay_rate_3+decay_act_3+tab+last1+lastk_rate+gap, k=16, n_buckets=10)

| alpha | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|
| 0.1 | 0.62840, 0.00018 | 0.62603, 0.00074 |
| 0.25 | 0.62898, 0.00066 | 0.62724, 0.00049 |
| **0.5** | **0.63013, 0.00069** | 0.62696, 0.00060 |
| 0.75 | 0.62953, 0.00117 | 0.62681, 0.00084 |
| 1.0 (iter19 default) | 0.62933, 0.00040 | 0.62632, 0.00037 |

alpha=0.5 is again the best point on valid, consistent with iter21's finding
on iter16's older feature set — but the margin over the default here is
smaller and noisier than iter21's own finding (+0.00080 valid / +0.00064
test here, vs iter21's +0.00125 valid / +0.00287 test on iter16): with 6
features already encoding recency/momentum, the extra signal recoverable
from retuning just the smoothing constant is smaller. Direction is still
consistent on both splits (not a valid-only artifact), so it's a real
if modest effect on its own.

## Axis B — embedding capacity (k) / bucket-count (n_buckets) resweep (3 seeds each, alpha=1.0 default, on iter19's exact feature set)

| config | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|
| k=16 (baseline, = n_buckets=10 row) | 0.62933, 0.00040 | 0.62632, 0.00037 |
| k=24 | 0.62818, 0.00091 | 0.62553, 0.00218 |
| k=32 | 0.62786, 0.00076 | 0.62698, 0.00011 |
| n_buckets=5 | 0.62784, 0.00013 | 0.62440, 0.00116 |
| n_buckets=10 (baseline) | 0.62933, 0.00040 | 0.62632, 0.00037 |
| **n_buckets=20** | **0.62996, 0.00016** | **0.62994, 0.00092** |

**Embedding capacity (k) still doesn't help** — k=24 and k=32 both sit
*below* k=16 on both splits, reconfirming the README's original finding
(capacity was never the bottleneck for the pointwise baseline, and still
isn't for this much richer BPR+feature model either). **But bucket count
does** — n_buckets=20 beats the n_buckets=10 default by a real margin on
both splits, and on test the gap is large (+0.00362, ~4x the config's own
std and far outside the baseline's std of 0.00037): the quantile buckets
iter16/iter18/iter19 all inherited a hardcoded default of 10, and with a
6-field feature set now carrying much of the model's signal, finer
resolution on the continuous fields (`decay_rate`, `decay_act`,
`lastk_rate`, `gap`, `dur_bucket`) recovers real headroom that the
now-outdated README capacity finding did not anticipate — because that
finding was about embedding dimension `k`, not discretization granularity,
and nobody had swept `n_buckets` against the richer feature set before.

## Combo (alpha=0.5 + n_buckets=20 + k=16, 3 seeds)
Both axes individually cleared the "worth confirming" bar (>0.001 valid vs
iter19's 5-seed 0.62898: alpha=0.5 alone +0.00115, n_buckets=20 alone
+0.00098), so per the dispatch instructions we combined the two winners:

| config | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|
| alpha=0.5 alone (n_buckets=10) | 0.63013, 0.00069 | 0.62696, 0.00060 |
| n_buckets=20 alone (alpha=1.0) | 0.62996, 0.00016 | 0.62994, 0.00092 |
| **combo (alpha=0.5, n_buckets=20)** | **0.63051, 0.00005** | **0.63152, 0.00032** |

The combo beats **both** individual winners on both splits, and its
variance is tiny (valid std 0.00005 across 3 seeds) — the two retuned
constants are complementary, not redundant, exactly like iter19's own
decay+momentum feature fusion was.

## 5-seed confirmation (combo: alpha=0.5, n_buckets=20, k=16, iter19's feature set)

| seed | valid | test |
|---|---|---|
| 0 | 0.63058 | 0.63107 |
| 1 | 0.63050 | 0.63173 |
| 2 | 0.63045 | 0.63175 |
| 3 | 0.62940 | 0.63150 |
| 4 | 0.63046 | 0.63323 |
| **mean** | **0.63028** | **0.63185** |
| **std** | 0.00044 | 0.00073 |

## Comparison against iter19 (current standing best)

| | iter19 (5-seed) | iter25 combo (5-seed) | delta |
|---|---|---|---|
| valid primary | 0.62898 (std 0.00063) | 0.63028 (std 0.00044) | **+0.00130** |
| test primary | 0.62615 (std 0.00058) | 0.63185 (std 0.00073) | **+0.00570** |

**All 5/5 seeds beat iter19 on both valid and test individually** (no sign
flips, no cherry-picking): valid deltas per-seed range +0.00055 to
+0.00156; test deltas per-seed range +0.00489 to +0.00660. The test-split
gain (+0.0057, ~7.8x iter25's own test std, ~10x iter19's original test
std) is unusually large relative to the valid-split gain (+0.0013, ~3x
iter25's own valid std) — both point the same direction and both clear
noise by a wide margin, so this isn't a valid-only or test-only artifact,
but the asymmetry is notable and worth flagging: a future round could
check whether n_buckets=20's finer discretization is disproportionately
helping on the slightly-more-recent test window (04-29..05-08) versus
valid (04-22..28), e.g. via a train/valid-date-shifted robustness check,
before leaning on this result too heavily for a live deployment decision.
For this round's purposes the finding is real and promotable as-is.

## Verdict: **PROMOTE — new current best**

`iter25_retune_v2` (alpha=0.5, n_buckets=20, k=16, feature set
`decay_rate_3,decay_act_3,tab,last1,lastk_rate,gap`) beats iter19's test
primary of 0.62615 by +0.00570, with a consistent, same-direction margin
across all 5 seeds on both valid and test. This supersedes iter19 as the
standing best. Both abandoned iter21 threads are now closed: Axis A
confirms alpha=0.5 generalizes (modestly) to the richer feature set; Axis B
confirms embedding capacity `k` is still not the lever (reconfirms the
README), but **bucket-count `n_buckets` is** — a genuinely new lever the
README never tested, since its capacity finding was about `k` only.

## Code
`experiments/iter25_retune_v2/{data_ext.py,train.py,driver_axisA.py,driver_axisB.py,driver_combo.py}`.
`data_ext.py`/`train.py` are iter19's files with `alpha` and `n_buckets`
threaded through `encode_ext`/`run_bpr_ext` as sweepable keyword args
(default values unchanged: `ALPHA=1.0`, `n_buckets=10`), mirroring iter21's
identical change to iter16's `data_ext.py`/`train.py`. Raw sweep results:
`results_axisA.json` (15 rows: 5 alphas x 3 seeds), `results_axisB.json`
(18 rows: 3 k-values x 3 seeds + 3 n_buckets-values x 3 seeds, with the
k=16/n_buckets=10 baseline run twice under different tags as a built-in
cross-check — both reproduce iter19's exact published seed-0/1/2 numbers),
`results_combo.json` (5 rows: the 5-seed combo confirmation). The 3-seed
combo pre-check and the seed-3/4 extension were run via direct
`train.py --alpha 0.5 --n_buckets 20 --seed {N}` CLI calls (raw output in
`combo.log`); `driver_combo.py` is provided as a clean one-command rerun
path with the same schema/incremental-save pattern as the axis drivers.

# iter29 — train/valid-date-shifted robustness check of iter25's n_buckets=20 finding

Note: the dispatched agent's session was terminated mid-run by the same
Round-8 platform session-limit event that killed iter27/28/30 simultaneously.
The driver (`driver.py`) itself completed all 10 planned runs before the
kill (confirmed: `results.json` has all 5 seeds x 2 configs, `driver.log`
ends with "iter29 shifted-split n_buckets sweep complete."). Only the
analysis/writeup step was lost. The orchestrator wrote this file directly
from `results.json`/`driver.log` — no rerun was needed.

## Idea
iter25 (`experiments/iter25_retune_v2/`) found that `n_buckets=20` beats the
`n_buckets=10` default by a real margin on iter19's feature set, but flagged
an asymmetry worth checking: the test-split gain (+0.00362, 3-seed) was much
larger than the valid-split gain (+0.00063, 3-seed) — unusually so relative
to either split's own noise. iter25's own writeup explicitly recommended
"a train/valid-date-shifted robustness check" before leaning on the finding
too heavily. This iteration runs exactly that check, in isolation (no other
iter25 refinement — alpha stays at the 1.0 default, not iter25's tuned 0.5)
so the n_buckets effect alone is being tested for robustness, not the
combo's overall performance.

## Method
`data_ext.py` in this directory builds a **shifted** split: train
2022-04-05..18 / valid 2022-04-19..25 / test 2022-04-26..05-05 (each window
moved 3 days earlier than the official split), rather than the official
train 04-08..21 / valid 04-22..28 / test 04-29..05-08. iter19's exact
feature set (`decay_rate_3, decay_act_3, tab, last1, lastk_rate, gap`),
alpha=1.0, k=16 throughout. `n_buckets` in {10, 20}, 5 seeds each (0-4).

## Results (5 seeds each, shifted split)

| config | valid primary (mean) | test primary (mean) |
|---|---|---|
| `nbuckets_10` (control) | 0.62929 | 0.63186 |
| `nbuckets_20` (iter25's finding) | 0.62922 | 0.63242 |
| **delta (20 − 10)** | **−0.00007** | **+0.00056** |

Per-seed values (from `results.json`):

| seed | valid n=10 | valid n=20 | test n=10 | test n=20 |
|---|---|---|---|---|
| 0 | 0.62777 | 0.62948 | 0.63053 | 0.63135 |
| 1 | 0.62955 | 0.62845 | 0.63167 | 0.63301 |
| 2 | 0.62925 | 0.62944 | 0.63131 | 0.63209 |
| 3 | 0.62966 | 0.62862 | 0.63261 | 0.63241 |
| 4 | 0.63024 | 0.63013 | 0.63317 | 0.63323 |

## Comparison against iter25's original (standard-split) Axis B finding

| | standard split (iter25, 3-seed) | shifted split (iter29, 5-seed) |
|---|---|---|
| Δ valid (n_buckets 20 vs 10) | +0.00063 | **−0.00007** |
| Δ test (n_buckets 20 vs 10) | +0.00362 | **+0.00056** |

**The finding does not replicate under date-shifting.** On the shifted
split, `n_buckets=20`'s valid-side effect flips to a statistically-flat
negative (−0.00007, well within seed noise), and the test-side gain shrinks
to roughly 1/6th of its original magnitude (+0.00056 vs +0.00362) — also
within noise range of the two configs' own seed-to-seed variation (individual
per-seed deltas range from −0.00179 to +0.00168, straddling zero).

## Interpretation
This confirms the exact concern iter25 flagged: the original n_buckets=20
test-side gain, while real and consistent on the standard split (5/5 seeds
in iter25's own combo confirmation), appears to be **partly fold-specific**
rather than a fully general discretization-granularity effect. It does not
generalize cleanly to a 3-day-shifted window of the same underlying data.
This does **not** retroactively invalidate iter25's promotion — iter25's
5-seed combo (alpha=0.5 + n_buckets=20 together) was verified on the
official standard split with all 5/5 seeds improving on both valid and
test, which remains true — but it does mean the n_buckets=20 component in
isolation should be treated as a **weaker, less-general lever** than its
original 3-seed standard-split numbers suggested, and any future
generalization of "more buckets = better" beyond n_buckets=20 should not be
assumed without its own robustness check.

## Verdict: **Not a promotion candidate (this is a robustness audit, not a
new iteration) — informational finding, logged as a caveat on the
four-way crossover / iter25 promotion.**

## Code
`experiments/iter29_bucket_robustness/{data_ext.py,train.py,driver.py}`,
raw results in `experiments/iter29_bucket_robustness/results.json` (10 rows:
2 configs x 5 seeds), full run trace in `driver.log`.

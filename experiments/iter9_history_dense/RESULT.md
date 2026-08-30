# iter9 — FM + activity-weighted BPR + coarse causal history features

## Idea
iter6 tried a per-author user-affinity causal feature and found it useless (0.70%
nonzero coverage — users almost never re-encounter an author they'd previously
long-viewed within the ~1-month log window). iter9 asks: does the same causal
history *idea* work if the granularity is coarsened enough to get real density?

Three candidate features, all computed via the same strict-causal (`<`, never
`<=`) two-phase date-grouped traversal iter6 validated — see
`data_ext.py::compute_causal_features`:
- **activity**: count of this user's prior rows (any label) seen so far.
- **tab_pos**: count of this user's prior positive rows within the same `tab`
  (only 15 distinct tab values, vs. tens of thousands of authors).
- **rate**: this user's Laplace-smoothed prior positive rate so far
  (`(prior_pos+1)/(prior_total+2)`).

Everything else (FM, BPR loss, activity-weighted user sampling) is a line-for-line
copy of iter3's `train.py`, so the feature set is the only variable changed.

## Coverage (vs iter6's 0.70%)
- `activity` / `rate`: **92.29%** of rows have nonzero prior history (1,325,836 / 1,436,609)
- `tab_pos`: **73.37%** of rows (1,054,046 / 1,436,609)

## Causality verification
`data_ext.py`'s `__main__` block brute-force spot-checks every feature against a
manual O(n) recount for sampled rows, including a same-date-pair edge case
(two same-user/same-tab positives on the same date correctly exclude each other,
both showing `tab_pos=1` not `2`). All spot-checks pass. No leakage detected.

## Sweep (3 seeds, 3 feature-set variants)
| feature set | valid primary (mean) | test primary (mean) |
|---|---|---|
| activity only | 0.6031 | 0.5967 |
| activity + tab | 0.6046 | 0.5986 |
| **activity + tab + rate** | **0.6103** | **0.6057** |

Adding `rate` (the smoothed prior-positive-rate) drives nearly all of the gain —
consistent with it being the most directly label-correlated of the three signals,
while `activity`/`tab_pos` mostly help by giving `rate` a non-degenerate
denominator/context.

## Final result — 5-seed confirmation, activity+tab+rate (seeds 0-4)
| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.61028 | 0.60562 |
| 1 | 0.61038 | 0.60567 |
| 2 | 0.61006 | 0.60585 |
| 3 | 0.60963 | 0.60525 |
| 4 | 0.61029 | 0.60559 |
| **mean** | **0.61013** | **0.60560** |
| **std** | 0.00027 | 0.00020 |

vs. current best (iter3, activity-weighted BPR, no history features):
valid 0.60258 / test 0.59658

**Δ vs iter3: +0.00755 valid, +0.00902 test** — roughly 25-45x iter3's own std
(0.00031 valid / 0.00035 test) and iter9's own std. This is by far the largest,
tightest, most reproducible gain seen in the entire AutoML run so far.

## Status: **PROMOTED — NEW CURRENT BEST**

Beats iter1 official baseline (test 0.5946) by +0.0110, and beats iter3
(prior best) by +0.0090 test. Conclusion: coarsening the causal-history idea
from per-author (iter6, 0.70% coverage, useless) to per-tab / per-user-rate
(92% coverage) turns a dead signal into the single best lever found so far —
density, not the underlying idea, was the bottleneck in iter6.

## Code
`experiments/iter9_history_dense/{data_ext.py,train.py}`,
sweep logs in `experiments/iter9_history_dense/logs/sweep/`.

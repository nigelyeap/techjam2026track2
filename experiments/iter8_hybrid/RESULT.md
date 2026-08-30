# iter8 — Hybrid pointwise + activity-weighted BPR (joint training)

## Idea
Train pointwise logloss (iter1-style) and activity-weighted BPR (iter3-style)
jointly on the SAME shared FM weights/optimizer state: each epoch does one
full pointwise pass, then `steps_per_epoch` BPR steps scaled by `--bpr_weight`.
Hypothesis: pointwise's implicit activity-weighted calibration and BPR's
explicit pairwise ranking signal are complementary enough that a joint
objective beats either alone (a cheaper alternative to iter10's two-model
ensemble — one model, one training run).

## Sweep — bpr_weight ∈ {0.5, 1.0, 2.0} (3 seeds each)
| bpr_weight | mean valid primary | mean test primary |
|---|---|---|
| **0.5** | **0.6016** | **0.5949** |
| 1.0 | 0.6014 | 0.5938 |
| 2.0 | 0.6014 | 0.5933 |

bw=0.5 (lightest BPR weighting) wins; increasing bpr_weight monotonically
hurts test primary. Reproducibility check: the 3-seed sweep and a separate
3-seed "final" run at bw=0.5 land on the same numbers (seed0: 0.6016/0.5952
in both runs) — reproducible, not a fluke.

## Final result — 5-seed confirmation, bpr_weight=0.5
| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.6016 | 0.5952 |
| 1 | 0.6020 | 0.5946 |
| 2 | 0.6012 | 0.5949 |
| 3 | 0.6009 | 0.5943 |
| 4 | 0.6018 | 0.5940 |
| **mean** | **0.6015** | **0.5946** |
| **std** | 0.0004 | 0.0004 |

vs. iter3 (activity-weighted BPR alone): valid 0.60258 / test 0.59658
→ **Δ -0.0011 valid, -0.0020 test** — worse, ~5x the size of iter8's own std,
a real regression not noise.

## Status: **REJECTED**
Joint pass-by-pass training hurts rather than helps: alternating a full
pointwise pass with a BPR pass on the same shared Adam moment accumulators
seems to make each pass partially undo the other's progress, rather than the
two losses reinforcing a shared representation. Contrast with iter10, which
trains the two losses on fully independent models and ensembles at the score
level — that approach got a small positive (not negative) blend effect,
suggesting decorrelated independent training beats sharing weights/optimizer
state for these two losses. Lighter BPR weighting (0.5) is consistently less
harmful than heavier (1.0, 2.0), but even the best setting doesn't recover
iter3's performance, let alone improve on it.

## Code
`experiments/iter8_hybrid/train.py`, sweep log in
`experiments/iter8_hybrid/sweep_log.txt` and
`experiments/iter8_hybrid/sweep_results.json`.

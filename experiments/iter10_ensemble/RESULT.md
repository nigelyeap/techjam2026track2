# iter10 — Ensemble of pointwise FM + activity-weighted BPR FM

## Idea
Train two independent FMs per seed (pointwise, iter1-style; activity-weighted
BPR, iter3-style), then blend their predicted scores at inference time:
`score = w * bpr_score + (1-w) * pointwise_score`. Hypothesis: pointwise's
implicit activity-weighted calibration and BPR's explicit pairwise ranking
signal are complementary enough that blending beats either alone.

## Standalone reproduction check (sanity)
Per-seed standalone numbers reproduce iter1 (pointwise) and iter3 (BPR)
reference numbers almost exactly, confirming the reimplementation is faithful:
- pointwise seed0: valid 0.6015 / test 0.5953 (iter1 ref: 0.6015/0.5953)
- bpr seed0: valid 0.6025 / test 0.5963 (iter3 ref: 0.6025/0.5963)

## Weight sweep (seeds 0-2, mean valid primary)
| w | mean valid primary |
|---|---|
| 0.25 | 0.60303 |
| **0.5** | **0.60309** |
| 0.75 | 0.60266 |

w=0.5 wins, but the sweep is essentially flat (spread of 0.0004 across all
three weights) — the ensemble is not sensitive to the blend ratio.

## Final result — 5-seed ensemble at w=0.5
| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.60356 | 0.59773 |
| 1 | 0.60324 | 0.59763 |
| 2 | 0.60246 | 0.59781 |
| 3 | 0.60296 | 0.59736 |
| 4 | 0.60363 | 0.59713 |
| **mean** | **0.60317** | **0.59753** |
| **std** | 0.00043 | 0.00025 |

vs. iter3 (activity-weighted BPR alone): valid 0.60258 / test 0.59658
→ Δ +0.00059 valid, +0.00095 test — a small, real-but-marginal lift (~3x
iter3's std on test), consistent with the two models' errors being only
partially correlated.

vs. **iter9 (current best, history features)**: valid 0.61013 / test 0.60560
→ iter10 is **0.007-0.008 below iter9 test** — the ensembling gain is
completely dwarfed by iter9's history-feature gain.

## Status: **REJECTED** (relative to current best)
Marginal, real improvement over iter3, but far below iter9. Ensembling two
FMs that share the same feature set caps out well below what a genuinely new
causal feature (iter9) delivers. Not promoted — no reason to pay 2x training/
inference cost for a smaller gain when iter9 alone is both simpler and better.
Worth revisiting only as a *combination* with iter9's features (does BPR+
pointwise ensembling on top of the history-feature encoding add anything?)
if a future round wants to explore stacking gains rather than a new feature.

## Code
`experiments/iter10_ensemble/train.py`, full results in
`experiments/iter10_ensemble/results.json`.

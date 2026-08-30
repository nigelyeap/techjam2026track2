# iter13 — Ensemble pointwise + BPR, both fed iter9's extended (history) features

## Idea
iter10 ensembled a pointwise FM and a BPR FM (both on the plain 5-field
encoding) and found a small real gain over BPR-alone (iter3): +0.00095 test.
iter13 asks whether that same ensembling trick adds further gain ON TOP of
iter9's much stronger extended feature set (activity+tab+rate) — i.e. does
pointwise+BPR complementarity still hold once the features are this much
better, or does the richer signal make the pointwise side redundant/harmful.

## Standalone comparison (3 seeds: 0,1,2), both fed activity+tab+rate features
| model | valid primary (mean) | test primary (mean) |
|---|---|---|
| BPR-on-extended (= iter9 unmodified) | **0.61024** | **0.60572** |
| pointwise-on-extended (new for iter13) | 0.59935 | 0.59252 |

Critical finding: pointwise trained on the SAME extended features performs
*worse* than plain pointwise (iter1, test 0.5953) — adding the causal history
fields actively hurts the pointwise loss, unlike BPR where they hugely help.
Likely explanation: pointwise's dense per-row logloss training already implicitly
weights by activity/frequency, so the new fields mostly add noisy dimensions to
optimize over rather than new resolving signal, whereas BPR's within-user
pairwise objective directly benefits from a feature that changes the relative
ranking of a user's own history.

## Ensemble weight sweep (3 seeds each, score = w*pointwise + (1-w)*bpr)
| w (pointwise weight) | mean valid primary |
|---|---|
| **0.25** | 0.60942 |
| 0.5 | 0.60769 |
| 0.75 | 0.60412 |

Monotonic: more pointwise weight = worse. Even the best ensemble weight found
(w=0.25, i.e. 75% BPR / 25% pointwise) still lands at valid 0.60942 — **below**
BPR-alone's 0.61024 for the same 3 seeds. There is no weight at which blending
in the (now much weaker) pointwise model helps; every blend point tested is at
or below pure BPR-on-extended-features.

## Status: **REJECTED** — no confirmation run needed
The 3-seed sweep already conclusively shows every ensemble weight underperforms
iter9 (BPR alone) on valid, so no 5-seed confirmation was run (would only
waste compute confirming a clear negative). Conclusion: iter10's
pointwise/BPR complementarity does NOT transfer once iter9's history features
are in play — those features are BPR-specific in how much they help, and
mixing in pointwise (whose performance the same features actively degrade)
only pulls the ensemble down. iter9 (BPR-on-extended alone) remains the best
model; no further gain from ensembling on top of it via this route.

## Code
`experiments/iter13_ensemble_on_features/train.py`, full sweep results in
`experiments/iter13_ensemble_on_features/results_sweep.json`.

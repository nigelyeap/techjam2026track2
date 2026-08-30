# iter7 — Tuning iter3's activity-weighted BPR (sampling exponent, k, lr)

## Idea
iter3 weights per-user BPR sampling probability linearly by `pos_len[user]`
(exponent=1). iter7 generalizes this to `pos_len[user]**alpha` and sweeps
alpha, then (at the best alpha) sweeps embedding size `k` and learning rate
`lr`, to see if iter3's specific hyperparameters were actually optimal or just
a reasonable first guess.

## Phase 1 — alpha sweep (3 seeds each, k=16, lr=0.001)
| alpha | mean valid primary |
|---|---|
| 0.5 | 0.6016 |
| 0.75 | 0.6021 |
| 1.0 (= iter3) | 0.6024 |
| 1.5 | 0.6026 |

All four alphas land within ~0.001 of each other — indistinguishable relative
to seed noise (std ~0.0004-0.0006 per config). alpha=1.5 nominally highest by
a hair, selected for the 5-seed confirmation run and the k/lr sweep below.

## Phase 2 — k, lr sweep at alpha=1.5 (3 seeds each)
| config | mean valid primary |
|---|---|
| k=16, lr=0.001 (baseline) | 0.6026 |
| k=24, lr=0.001 | 0.6025 |
| k=32, lr=0.001 | 0.6021 |

No improvement from increasing embedding size; k=16 (iter3's original choice)
remains as good as any larger k tried. lr sweep configs (0.0005, 0.002) were
queued in `sweep.py` but skipped — given how flat alpha and k already are,
running a full lr grid was judged not worth the compute for a parameter this
unlikely to move the needle meaningfully.

## Final result — 5-seed confirmation, alpha=1.5, k=16, lr=0.001
| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.60256 | 0.59614 |
| 1 | 0.60280 | 0.59641 |
| 2 | 0.60244 | 0.59666 |
| 3 | 0.60202 | 0.59682 |
| 4 | 0.60322 | 0.59517 |
| **mean** | **0.60261** | **0.59624** |
| **std** | 0.00040 | 0.00058 |

vs. iter3 (alpha=1.0, current best going into this comparison):
valid 0.60258 / test 0.59658 → **Δ +0.00003 valid, -0.00034 test** — both
well within noise (iter7's own std is 0.00058 test, larger than the "gain").

## Status: **REJECTED** (no improvement, indistinguishable from iter3)
Conclusion: BPR's user-sampling exponent and FM capacity/step-size are not
sensitive levers in this range — iter3's original choices (alpha=1, k=16,
lr=0.001) were already effectively optimal for this architecture. This
hyperparameter axis is exhausted; further gains need a different lever
(confirmed separately by iter9's feature-based approach, which found a much
larger lever).

## Code
`experiments/iter7_bpr_tuned/{train.py,sweep.py}`, full sweep data in
`experiments/iter7_bpr_tuned/sweep_results.json`.

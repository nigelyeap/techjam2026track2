# iter38 — score-level ensemble of iter27's 5 confirmed seeds

## Idea

Every iteration so far reports either a single seed or the arithmetic mean
of 5 seeds' *metrics* — never an actual ensemble of the 5 trained *models'
predictions*. iter27's own 5-seed std (~0.0007-0.0008 on both splits) is
pure random-init/minibatch-order variance, not signal, which is exactly the
setting where prediction averaging reduces variance and should raise the
expected score. This changes nothing about features, loss, or sampling —
purely a new axis (how final predictions are produced from already-trained
models), reopened after convergence per explicit instruction to keep
pushing on score.

## Method

Train iter27's exact winning config (`ITER24_FEATS`, `sampling_mode='decay'`,
`sampling_alpha=0.75`, `decay_halflife=3`, `alpha=0.5`, `n_buckets=20`) at
seeds 0-4, identical to the already-published run, but additionally keep
each model's raw per-row logits for valid/test (not just the aggregate
metric). Ensemble two ways: (a) mean of raw logits across the 5 models,
(b) mean of sigmoid-transformed scores across the 5 models. Evaluate each
ensembled score array with `evaluate.py`, compared against iter27's
published 5-seed mean-of-metrics.

## Harness-fidelity check

All 5 seeds' individual valid/test primaries matched iter27's published
numbers bit-exact (max abs Δ = 0.0 across all 5 seeds, both splits) before
the ensemble was trusted.

## Result

| | valid primary | test primary |
|---|---|---|
| iter27 5-seed mean-of-metrics (published, current best going in) | 0.63792 | 0.63889 |
| **iter38 ensemble, mean of raw logits** | 0.63986 (Δ+0.00193) | 0.64180 (Δ+0.00292) |
| **iter38 ensemble, mean of sigmoid scores** | **0.63988** (Δ+0.00195) | **0.64187** (Δ+0.00298) |

Both ensembling variants clear the 0.001 valid-primary promotion margin by
a comfortable margin (~2x), and the direction is consistent between valid
and test (no sign flip) and between the two combination methods (sigmoid
mean marginally ahead of raw-logit mean, consistent with sigmoid bounding
each model's contribution before averaging).

## Verdict: PROMOTE

**New current best.** Selected variant: **sigmoid-mean ensemble of the 5
seed-0..4 models**, valid primary 0.63988, test primary 0.64187. This is a
genuinely new selection decision (an ensemble is a different predictor from
any single seed), made on valid per protocol, with test checked only after
the valid margin cleared.

Cost: 5x the training/inference compute of a single seed (~5x24s ≈ 120s
total on this hardware) and 5x the final submission's per-row scoring cost
(five FM forward passes averaged instead of one) — still CPU-only, still on
the order of two minutes total, not a meaningful resource-cost increase
under the Feasibility criterion.

## Code

`experiments/iter38_seed_ensemble/driver.py`, raw results in
`experiments/iter38_seed_ensemble/results.json`.

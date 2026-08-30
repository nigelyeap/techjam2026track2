# iter37 — FM embedding dimension (model-capacity) sweep on iter27's fused config

## Idea

Every prior round tuned features, sampling, or formula constants — model
capacity (`k`, the FM embedding dimension, default 16 since iter1) has
never been swept on top of the fused config. Disjoint axis: capacity vs.
input/training-time changes.

## Method

`data_ext.py`/`train.py` are iter27's, unmodified (both already accept `k`
as a passthrough parameter). `driver.py` runs iter27's exact winning config
(`sampling_alpha=0.75, alpha=0.5, n_buckets=20, decay_halflife=3`,
ITER24_FEATS) at `k` ∈ {8, 24, 32}, 3 seeds each, vs the `k=16` baseline.

## Harness-fidelity check

`k=16`, 5 seeds: bit-exact match to iter27's published numbers (max abs
Δ=0.0, all 5 seeds, both splits).

## Sweep (3 seeds, valid-only selection)

| k | valid mean (3-seed) | test mean (3-seed) | Δ valid vs iter27 (0.63816) |
|---|---|---|---|
| 8 | 0.63750 | 0.63890 | −0.00065 |
| 16 (baseline) | 0.63816 | — | — |
| 24 | 0.63787 | 0.63826 | −0.00029 |
| 32 | 0.63731 | 0.63917 | −0.00085 |

All deltas are within seed-to-seed noise (compare iter27's own 5-seed std
of ~0.0007-0.0008) and none clears the 0.001 promotion margin in either
direction. Orchestrator independently verified all 4 tags' means by hand
against `results.json` (bit-exact match).

## Diagnosis

`k=16` is already at a flat capacity optimum for this fused config — neither
shrinking (k=8) nor growing (k=24, k=32) the embedding dimension moves valid
primary outside noise. This is a clean negative: the model is not
capacity-bottlenecked at its current feature/sampling/formula
configuration, so further gains are unlikely to come from this axis alone.

## Verdict: REJECT

iter27 remains current best, k=16 confirmed appropriate. No further capacity
tuning warranted without a change to the feature set that would justify
revisiting embedding width.

## Code

`experiments/iter37_embedding_dim/{data_ext.py,train.py,driver.py}`, raw
results in `experiments/iter37_embedding_dim/results.json`.

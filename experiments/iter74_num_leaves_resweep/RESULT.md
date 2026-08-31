# iter74 — num_leaves resweep on the current final feature set

## Provenance

Not a new feature — a structurally different lever from Round 22's now-closed
decayed-rate-generalization family (iter68-73, 4/4 REJECT). `num_leaves` was
last swept in iter52 (the `linear_tree` resweep), which chronologically
predates iter63's introduction of `decay_tab_rate_3` — the one real feature
gain found in this whole project. Round 17 declared the GBM hyperparameter
space "exhausted," but that declaration was made on a strictly weaker feature
set. This retests whether `num_leaves=2` is still optimal now that a
genuinely new proven feature has been added, using the same staleness
argument already validated for the iter15→iter68 retest earlier in the
project.

## Implementation

`experiments/iter74_num_leaves_resweep/train.py` reuses iter63's `run()`
unchanged (it already accepts `num_leaves` as a parameter). Features are
prepared once (`rate_only` variant) and shared across all `num_leaves`
values via the `_cache=` argument, so only the tree-capacity hyperparameter
varies; `learning_rate=0.10`, `n_estimators=500`, `min_child_samples=200`,
`reg_lambda=1.0`, `linear_tree=True` all held fixed at iter63's exact
winning config. Sweep grid: `{2, 3, 4, 5, 6, 7, 8, 10}`. Harness-fidelity
check at `num_leaves=2` reproduces iter63's exact baseline
(valid=0.67168, test=0.65353) before trusting any other number. No new
causal feature is introduced, so no causality verification applies.

## Result (seed 0, rate_only feature set)

| num_leaves | valid | test | Δvalid | Δtest |
|---|---|---|---|---|
| 2 | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| 3 | 0.66935 | 0.65218 | -0.00233 | -0.00134 |
| 4 | 0.66635 | 0.64978 | -0.00533 | -0.00375 |
| 5 | 0.66273 | 0.64719 | -0.00895 | -0.00633 |
| 6 | 0.65834 | 0.64493 | -0.01334 | -0.00859 |
| 7 | 0.65657 | 0.64254 | -0.01511 | -0.01098 |
| 8 | 0.65612 | 0.64279 | -0.01556 | -0.01074 |
| 10 | 0.64741 | 0.64574 | -0.02427 | -0.00779 |

## Diagnosis

Monotonic, substantial degradation as `num_leaves` increases from 2 — not
noise-level, and not close to a crossover at any point in the grid. `best_iteration`
also collapses hard (48 → 24 → 15 → 8 → 6...) as `num_leaves` grows, consistent
with each individual tree gaining enough capacity per split that the ensemble
starts overfitting far earlier. On this row-per-impression, heavily-categorical,
`linear_tree=True` setup, a 2-leaf (single-split) tree evidently remains the
right capacity ceiling — this is unchanged by adding `decay_tab_rate_3`. The
staleness concern motivating this retest is resolved: `num_leaves=2` is still
optimal on the current feature set, not just the pre-iter63 one.

## Verdict: REJECT (clean, no promotion)

No 5-seed confirmation needed — the effect is monotonic and an order of
magnitude larger than this project's "even look twice" threshold (~0.0003
valid), so seed noise cannot explain it. iter63's exact config
(`num_leaves=2`, `rate_only` decay-tab-rate feature set) remains the current
best. This closes the immediate hyperparameter-staleness question; no further
`num_leaves` retest is warranted unless the feature set changes again in a way
that plausibly shifts the bias/variance tradeoff (e.g. a much larger new
feature block).

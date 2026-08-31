# iter78 — learning_rate / reg_lambda / min_child_samples staleness resweep

## Provenance

Parallel staleness argument to iter74's `num_leaves` retest: `learning_rate`
(iter55/56), `reg_lambda` (iter57), and `min_child_samples` (iter58) were
all tuned before iter63's `decay_tab_rate_3`-replaces-`decay_tab_3` feature
swap and never re-checked against the actual current feature set. iter74
already closed this gap for `num_leaves`; this closes it for the three
remaining hyperparameters.

## Implementation

Staged coordinate-descent, reusing iter63's own `run()`/`prepare()`
unchanged (features cached once, no new feature = no causality check
needed): Stage 1 `learning_rate ∈ {0.03,0.05,0.07,0.10,0.13,0.16,0.20,0.25}`
→ pick best → Stage 2 `reg_lambda ∈ {0.1,0.3,1.0,3.0,10.0,30.0}` (at best
lr) → pick best → Stage 3 `min_child_samples ∈ {50,100,200,400,800,1600}`
(at best lr, reg_lambda). Harness-fidelity check reproduced iter63's exact
baseline (valid=0.67168, test=0.65353) before trusting the sweep.

## Result (seed 0)

| stage | winner | valid | Δvalid vs. iter63 default |
|---|---|---|---|
| learning_rate | 0.10 (unchanged) | 0.67168 | +0.00000 |
| reg_lambda | 10.0 (vs. default 1.0) | 0.67186 | +0.00018 |
| min_child_samples | 200 (unchanged, flat 0.67186 across entire grid 50-1600) | 0.67186 | — |

**Final best config** (`lr=0.10, reg_lambda=10.0, mcs=200`): valid=0.67186,
test=0.65359 — Δvalid **+0.00018**, Δtest **+0.00007** vs. iter63's exact
defaults.

## Diagnosis

`learning_rate=0.10` is reconfirmed as the exact optimum on the current
feature set — iter55's finding was not stale. `min_child_samples` remains
completely flat across a 32x range (50 to 1600), reconfirming iter58's
original flat/moot finding, unaffected by the feature swap. `reg_lambda`
shows a small, monotonically-then-decreasing bump at 10.0, but the margin
(+0.00018 valid) sits well below this project's single-seed noise floor
(prior 5-seed stds on this exact config have run 0.0002-0.0008) and far
below the ~0.001 threshold this project reserves 5-seed confirmation for.
Not a surprising or borderline result — a negligible, likely-noise-level
bump, not a real lever.

## Verdict: REJECT (no promotion; no 5-seed confirmation warranted)

iter63's exact defaults (`lr=0.10, reg_lambda=1.0, mcs=200`) remain the
current best. Combined with iter74 (`num_leaves`), this closes the
hyperparameter-staleness question completely: all four of the GBM's core
hyperparameters (`num_leaves`, `learning_rate`, `reg_lambda`,
`min_child_samples`) have now been independently reconfirmed optimal (or
negligibly improvable) on the actual current `rate_only` feature set, not
just on the pre-iter63 feature set they were originally tuned against.

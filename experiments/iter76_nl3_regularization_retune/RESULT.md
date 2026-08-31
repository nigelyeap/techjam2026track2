# iter76 — can num_leaves=3 be tamed by regularization it was never given?

## Provenance

iter74 found `num_leaves=3` regresses vs. `num_leaves=2` (Δvalid -0.00233,
`best_iteration` collapsing 48→24 — rapid overfitting). But `reg_lambda`
(iter57) and `min_child_samples` (iter58) were previously found flat/moot
**at `num_leaves=2` specifically** — iter57's own diagnosis was that at 2
leaves "nearly all model flexibility lives in the per-leaf linear fit,"
leaving the tree-structure regularizer nothing to do. `num_leaves=3`
introduces an actual split decision and less data per leaf (3-way split vs.
2's ~50/50), so `reg_lambda`, `min_child_samples`, and `linear_lambda`
(leaf-linear L2, tuned at `num_leaves=2` in iter53 where "each leaf gets
~half the training set, plenty of data") could plausibly bind differently.
None of the three had been swept at any `num_leaves` other than 2. This
combines three already-proven pieces (iter63's feature set, iter74's
capacity finding, Round 17's regularization-sweep methodology) rather than
introducing a new untested mechanism — directly the "combine methods that
work" instruction.

## Implementation

`experiments/iter76_nl3_regularization_retune/train.py`: staged
coordinate-descent at fixed `num_leaves=3` on the `rate_only` feature set —
Stage 1 sweeps `reg_lambda ∈ {1,3,10,30,100}`; Stage 2 fixes the Stage-1
winner and sweeps `min_child_samples ∈ {200,500,1000,2000,4000}`; Stage 3
fixes both and sweeps `linear_lambda ∈ {0,0.1,0.5,1,3}`. Harness-fidelity
check against iter63's exact `num_leaves=2` baseline passed before the sweep.

## Result (seed 0, num_leaves=3, rate_only feature set)

**Stage 1 (reg_lambda):**

| reg_lambda | valid | test | best_iter |
|---|---|---|---|
| 1.0 | 0.66935 | 0.65218 | 24 |
| 3.0 | 0.66929 | 0.65214 | 24 |
| **10.0** | **0.66991** | 0.65192 | 24 |
| 30.0 | 0.66942 | 0.65171 | 21 |
| 100.0 | 0.66960 | 0.65134 | 23 |

**Stage 2 (min_child_samples, reg_lambda=10.0 fixed):** bit-identical
(0.66991) across `{200, 500, 1000, 2000, 4000}` — completely flat, exactly
like iter58's finding at `num_leaves=2`; this constraint never binds at this
row/leaf ratio regardless of tree capacity.

**Stage 3 (linear_lambda, reg_lambda=10.0, min_child_samples=200 fixed):**

| linear_lambda | valid | test |
|---|---|---|
| **0.0** | **0.66991** | 0.65192 |
| 0.1 | 0.66986 | 0.65193 |
| 0.5 | 0.66920 | 0.65202 |
| 1.0 | 0.66918 | 0.65201 |
| 3.0 | 0.66950 | 0.65237 |

Default `linear_lambda=0.0` remains best; no value recovers or improves on it.

**Best num_leaves=3 config found**: `reg_lambda=10.0, min_child_samples=200,
linear_lambda=0.0` → valid=0.66991, test=0.65192 — **still Δvalid -0.00177
vs. num_leaves=2's 0.67168**. Regularization narrowed the gap slightly
(from iter74's -0.00233 at defaults to -0.00177 at the best-found config)
but did not close it, let alone beat `num_leaves=2`.

## Diagnosis

`num_leaves=3`'s regression is not a regularization deficiency that default
hyperparameters happened to miss — it persisted across a 15-point sweep over
the three most relevant regularizers, including one order-of-magnitude
tree-structure regularization (`reg_lambda=10-100`) far outside anything
previously tuned. `min_child_samples` reconfirms iter58's flat/moot finding
at a different `num_leaves`, generalizing that result. This is a genuine
capacity mismatch: for this row-per-impression, heavily-categorical,
linear-leaf-regression setup, a single global split (`num_leaves=2`) is
fundamentally better-matched than any 3-leaf structure, regardless of how
that extra structure is regularized.

## Verdict: REJECT (clean, no promotion)

Single seed sufficient — the gap is large and consistent across the entire
regularization grid, not a borderline/noisy result. `num_leaves=2` with
iter63's exact config remains the current best. This closes the GBM
capacity/regularization question decisively: not only is `num_leaves=2`
optimal at default regularization (iter74), it remains optimal across a wide
regularization sweep specifically designed to give `num_leaves=3` its best
possible chance (this iteration). No further `num_leaves`-adjacent
hyperparameter work is warranted without a structurally different feature
set change.

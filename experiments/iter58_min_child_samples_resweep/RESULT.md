# iter58 — min_child_samples resweep under linear_tree=True + learning_rate=0.10

## Motivation

`min_child_samples=200` was tuned in iter44/46 against the OLD constant-leaf
tree at `learning_rate=0.05`. Under `linear_tree=True`, this parameter
controls how much data each leaf's *linear* fit gets to train on, which is a
more direct interaction with the structural change (iter51) than
`reg_lambda` was — never re-checked after both `linear_tree=True` and
`learning_rate=0.10` (iter55) changed the training dynamics.

## Method

Sweep of `min_child_samples` over `{20, 50, 100, 150, 200, 300, 400, 600,
800, 1200}`, seed=0, all other hyperparameters at iter55's exact winning
config.

## Result

| min_child_samples | valid | test |
|---|---|---|
| 20 – 1200 | **0.67052** (identical) | **0.65277** (identical) |

The metric is bit-identical across the entire two-order-of-magnitude range.
**Verdict: REJECT** — with `num_leaves=2`, the tree has exactly two leaves
total, so with tens of thousands of training rows each leaf always holds
far more than even the largest tested `min_child_samples=1200`; the
constraint never binds at this capacity setting regardless of value.

## Consolidated conclusion: GBM tree-structure hyperparameters exhausted

Combined with iter53 (`linear_lambda`, REJECT — flat) and iter57
(`reg_lambda`, REJECT — flat), this closes the third and last
tree-structure-side regularization knob under the current
`linear_tree=True + learning_rate=0.10` config. All three are flat/moot at
`num_leaves=2`, because with only two leaves per tree essentially all model
flexibility lives in the per-leaf linear fit (governed by `linear_lambda`,
already at its confirmed-optimal default) and in `learning_rate` /
`n_estimators` (already resolved in iter55/iter56). The GBM side of the
hyperparameter space is now considered exhausted for this architecture;
further gains, if any, are more likely to come from the FM side, the blend
mechanism, or a genuinely new feature/architecture angle rather than from
further GBM knob resweeps.

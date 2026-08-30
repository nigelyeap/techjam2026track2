# iter57 — reg_lambda resweep under linear_tree=True + learning_rate=0.10

## Motivation

`reg_lambda=1.0` was tuned in iter44/46 against the old constant-leaf tree
at `learning_rate=0.05`. iter53 already checked `linear_lambda` (the
leaf-linear-model's own regularizer) under `linear_tree=True` and found
the default optimal, but `reg_lambda` (the tree-structure objective's
regularizer) was never re-checked after *both* `linear_tree=True`
(iter51) and `learning_rate=0.10` (iter55) changed the training dynamics.

## Method

Sweep of `reg_lambda` over `{0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0,
5.0}`, seed=0, all other hyperparameters at iter55's exact winning config.

## Result

| reg_lambda | valid | test |
|---|---|---|
| 0.00 – 3.00 | **0.67052** (identical) | **0.65277** (identical) |
| 5.00 | 0.67023 | 0.65256 |

The metric is bit-identical across the entire `{0.0..3.0}` range and only
degrades slightly at 5.0. **Verdict: REJECT** — `reg_lambda=1.0` sits in a
flat, wide plateau rather than a sharp optimum; the tree-structure
regularizer has essentially no effect at this hyperparameter combination
(`num_leaves=2` already leaves almost no room for the objective to
overfit via tree structure — nearly all model flexibility now lives in
the per-leaf linear fit instead, which is governed by `linear_lambda`,
already confirmed optimal at its default in iter53).

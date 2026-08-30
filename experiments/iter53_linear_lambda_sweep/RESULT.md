# iter53 — linear_lambda (leaf-linear-model regularization) sweep

## Motivation

`linear_tree=True` fits a regularized linear regression per leaf; LightGBM
exposes `linear_lambda` (default 0.0, unregularized) as the L2 penalty on
those per-leaf linear coefficients specifically — distinct from
`reg_lambda`, which regularizes the tree-structure objective overall and
was already tuned (as 1.0) in iter44/46 before `linear_tree` existed.
iter51 never touched `linear_lambda`, leaving it at the LightGBM default.
This sweeps it directly on top of iter51's exact winning config.

## Method

Single-axis sweep of `linear_lambda` over
`{0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0}`, seed=0, all other
hyperparameters unchanged from iter51 (`num_leaves=2, linear_tree=True`).

## Result

| linear_lambda | valid | test |
|---|---|---|
| **0.0 (default)** | **0.66932** | **0.65146** |
| 0.01 | 0.66932 | 0.65146 |
| 0.05 | 0.66932 | 0.65146 |
| 0.10 | 0.66932 | 0.65146 |
| 0.50 | 0.66920 | 0.65137 |
| 1.00 | 0.66876 | 0.65090 |
| 5.00 | 0.66872 | 0.65133 |
| 10.00 | 0.66877 | 0.65096 |

Small values (≤0.1) are indistinguishable from the unregularized default
(each leaf's linear model is fit on ~half the training set at
`num_leaves=2`, plenty of data to not need extra regularization);
values ≥0.5 monotonically hurt. **Verdict: REJECT.** The LightGBM default
(`linear_lambda=0.0`) is already at the optimum for this configuration —
consistent with iter51's other hyperparameters (`reg_lambda=1.0`) already
providing enough regularization pressure via the tree-structure term.

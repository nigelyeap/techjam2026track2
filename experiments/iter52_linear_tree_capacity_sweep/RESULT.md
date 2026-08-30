# iter52 — capacity (num_leaves) resweep under linear_tree=True

## Motivation

iter46's capacity sweep (num_leaves down to the floor of 2) was run
against the OLD constant-leaf GBM, where every extra leaf only buys
another flat constant. iter51 found that `linear_tree=True` turns each
leaf into a linear model instead — a structural change to what capacity
means, so the old "shrink to the floor" conclusion was not guaranteed to
transfer. This resweeps `num_leaves` with `linear_tree=True` fixed on,
everything else at iter51's exact winning hyperparameters.

## Method

Single-axis sweep of `num_leaves` over `{2, 3, 4, 5, 7, 10, 15, 20, 31}`,
`linear_tree=True` throughout, seed=0, all other hyperparameters
unchanged from iter51.

## Result

| num_leaves | valid | test |
|---|---|---|
| **2** | **0.66932** | **0.65146** |
| 3 | 0.66545 | 0.64903 |
| 4 | 0.63934 | 0.63080 |
| 5 | 0.63583 | 0.63171 |
| 7 | 0.63136 | 0.62566 |
| 10 | 0.63895 | 0.62607 |
| 15 | 0.63483 | 0.62479 |
| 20 | 0.63218 | 0.62381 |
| 31 | 0.62953 | 0.61943 |

`num_leaves=2` remains the clear optimum — the drop-off from 2→3 is
already -0.0039 valid, and by 4+ leaves the score has collapsed to
roughly the pre-iter44 FM-ensemble baseline level (~0.639), well below
even the old constant-leaf GBM's 0.66135. If anything, `linear_tree=True`
makes the model *more* sensitive to over-capacity than the constant-leaf
version was, not less: each additional leaf now gets its own linear
model fit on a shrinking, more overfit-prone data slice, compounding the
usual over-capacity problem with a higher-variance per-leaf estimator.
**Verdict: REJECT.** iter51's `num_leaves=2` stands confirmed as optimal
under both tree types.

# iter50 — GOSS boosting at num_leaves=2

## Motivation

iter46's boosting-type/sampling sweep covered `gbdt` (baseline), `dart`, and
row/column subsampling, but never GOSS (Gradient-based One-Side Sampling)
specifically — a distinct boosting algorithm (keeps all large-gradient
instances, randomly subsamples small-gradient ones), not a sampling-rate
variant of plain gbdt. Cheap, single-axis test on iter44's exact
pipeline/hyperparameters, `boosting_type` swapped only.

## Result

| | valid | test |
|---|---|---|
| `gbdt` (harness check, reproduces iter44 exactly) | 0.66135 | 0.64794 |
| `goss` | 0.64423 | 0.63413 |

Clearly worse (-0.017 valid) — below the 0.0003 look-threshold by a wide
margin, no seeds run. At `num_leaves=2`, each tree makes exactly one split;
GOSS's gradient-based instance subsampling reduces the *training set* each
split is chosen from, which only adds noise to that one split's threshold
when there's no capacity left to compensate across more splits — the same
class of finding as iter46's `subsample`/`colsample_bytree` results (both
also worse at this capacity for the same reason: stochastic sampling costs
more in split-quality than it buys in variance reduction when there's only
one split to get right).

**Verdict: REJECT.** Final model unchanged (iter44's blend, valid 0.66473 /
test 0.65197).

Code: `experiments/iter50_goss_boosting/train.py`.

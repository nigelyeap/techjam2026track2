# iter47 — stacking meta-learner over FM + GBM (+ CatBoost) scores

## Motivation

iter44's blend uses one scalar (`alpha=0.1`) found by a grid sweep on valid. This generalizes
that to a learned combiner: does a proper (multi-parameter) stacking model beat the single-alpha
blend, and does adding iter45's CatBoost as a third base model — despite being individually much
weaker (valid 0.62964 vs FM's 0.63988 and GBM's 0.66135) — add anything through diversity, the
way GBM's addition to FM did in iter44?

Two approaches tested: (1) a small logistic regression (intercept + one coefficient per
min-max-normalized base-model score) fit by full-batch gradient descent directly on valid
labels — a natural generalization of "sweep a parameter on valid," the same selection principle
iter44's alpha sweep already used, just via optimization instead of grid search; (2) a direct
grid search over blend weights, evaluated on the actual primary metric (not a proxy loss), as a
diagnostic for approach (1)'s result.

## Result

| Approach | valid | test |
|---|---|---|
| Baseline: iter44 fixed alpha=0.1 (FM+GBM only) | **0.66473** | 0.65197 |
| Logistic stack, FM+GBM (2-way) | 0.65426 | 0.65185 |
| Logistic stack, FM+GBM+CatBoost (3-way) | 0.57343 | 0.55369 |
| Direct grid search, FM+GBM+CatBoost (3-way, on primary metric) | **0.66473** | 0.65197 |

**Both logistic stacks underperform the simple fixed-alpha baseline on valid** — the 2-way stack
by a real margin (-0.0105 valid) despite having more free parameters and being fit directly on
valid data, and the 3-way stack catastrophically (-0.091 valid, with CatBoost's learned weight
going *negative*, -3.02 — a sign the optimizer found a spurious BCE-minimizing direction that
does not track ranking quality).

**Diagnosis, confirmed by the direct grid search:** binary cross-entropy over individual rows is
the wrong proxy objective for a query-grouped ranking metric (mean of GAUC and per-user nDCG@5).
Minimizing pointwise BCE does not necessarily improve within-group ranking, so a BCE-fit stack
can (and did) find weights that look better by its own loss but rank worse. This is the same
family of lesson as iter39's listwise-softmax REJECT: proxy-objective mismatch, not lack of
capacity. The **direct grid search — which evaluates the actual primary metric at every
candidate weight instead of a proxy loss — converges to exactly `w_fm=0.1, w_gbm=0.9, w_cb=0.0`,
reproducing iter44's original 2-model blend bit-for-bit (valid 0.66473, test 0.65197,
matching to 5 decimal places).** Two things this confirms simultaneously:

1. **CatBoost adds zero value to the blend even with a free weight and metric-aligned
   selection** — the grid search was free to assign it any weight and chose exactly 0. Its
   errors are not diverse enough from FM/GBM's to help, unlike GBM's addition to FM in iter44.
   This settles the open question from iter45's writeup.
2. **iter44's alpha=0.1 blend is already the true optimum** among all linear combinations of
   these three models' scores, evaluated on the real metric — there is no stacking gain
   available here, learned or grid-searched, beyond what the original 2-parameter sweep already
   found.

**Verdict: REJECT (no promotable finding) — confirms iter44's blend stands unchanged as the
final model.** Useful negative result on two fronts: BCE-based stacking is the wrong tool for
this task's grouped ranking metric (use metric-aligned search instead, as iter44 already did),
and CatBoost's weaker signal genuinely does not overlap with GBM/FM's blind spots.

Code: `experiments/iter47_stacking_meta/stack.py`. Full results in `results.json`.

# iter46 — extreme-low-capacity hyperparameter depth sweep at LightGBM's num_leaves=2 floor

## Motivation

iter44 swept `num_leaves` down to LightGBM's hard floor (2) and found the winner there, but every
*other* hyperparameter (learning_rate, n_estimators, min_child_samples, reg_lambda, subsampling,
boosting_type) was only ever tuned around `num_leaves=7` (`sweep2.py`'s last 5 rows) — never at
the actual winning capacity. This checks whether any of those axes still has headroom at
`num_leaves=2`, fixed at iter44's exact winning config as the baseline
(`num_leaves=2, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0` →
valid 0.66135, reproduced as config `[0]` — harness-fidelity confirmed before trusting the rest
of the sweep).

## Result: a clean, comprehensive null finding — num_leaves=2 with the original hyperparameters is a genuine local optimum

18 configs tested, one axis at a time, all fixed at `num_leaves=2`:

| Axis | Result |
|---|---|
| `learning_rate` (0.01/0.02/0.08/0.1 vs baseline 0.05) | **All worse.** 0.01→0.58777 (badly worse, early-stops too soon even with 4x more `n_estimators`), 0.02→0.65828, 0.08→0.65919, 0.1→0.65893 — 0.05 is the optimum |
| `min_child_samples` (50/100/400/800/1600 vs baseline 200) | **No effect at all** — every value ties the baseline exactly at 0.66135. With ~880K training rows split into only 2 leaves per tree, every tested threshold is far below the natural per-leaf sample count, so this constraint never binds |
| `reg_lambda` (0.1/0.3/3.0/10.0 vs baseline 1.0) | **All worse or flat.** Best alternate (0.1→0.66116) still trails baseline by 0.00019 |
| `subsample` (0.7/0.9, new axis, not swept anywhere before) | **Worse at both** — 0.7→0.64463, 0.9→0.65502. Stochastic row sampling only adds noise at this capacity |
| `colsample_bytree` (0.7/0.9, new axis) | **Worse at both** — 0.7→0.65760, 0.9→0.66094 |
| `boosting_type='dart'` (new axis) | **Worse** — 0.63066 vs 0.66135 gbdt |

**No single-axis change beat the baseline 0.66135 valid.** This is a genuine, comprehensively-
checked local optimum, not an under-tuned config — six independent axes (three previously
untested: subsample, colsample, dart) all point the same direction. `min_child_samples`'s total
lack of effect is itself informative: it confirms the earlier hyperparameter sensitivity seen
around `num_leaves=7` doesn't carry over to `num_leaves=2`, where the leaf-count constraint
dominates every other regularization knob.

**Verdict: REJECT (no promotable finding) — confirms iter44's config stands as the true optimum
along every hyperparameter axis tested.** This is a useful negative result: it closes off further
hyperparameter search on the current LightGBM-native model as a lever, redirecting remaining
effort toward genuinely different levers (new base models, new features, blending strategy —
see iter45/iter47).

Code: `experiments/iter46_extreme_capacity_depth/sweep.py`. Full grid in `sweep_results.json`.

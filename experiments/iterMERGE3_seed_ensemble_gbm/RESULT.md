# iterMERGE3: seed-ensembling LightGBM and XGBoost

## Verdict: REJECT — seed-ensembling LGB+XGB makes the blend worse, not better

## Hypothesis

The production blend (`make_submission.py`) already 5-seed-ensembles its FM
component (seeds 0-4, sigmoid-mean of `train_one_fm`), but trains LightGBM
and XGBoost at a single seed (seed 0) each. Untried by either research
track: does seed-ensembling LGB and XGB the same way (seeds 0-4, same
feature sets/configs, only `random_state` varies) reduce variance and lift
the 3-way within-user-percentile blend above the current best (valid
0.69943440 / test 0.68432260)?

## Harness-fidelity check (mandatory first step)

Reproduced, at seed 0, using yixi's exact `LGB_CONFIG`/`LGB_CANDIDATE_COLUMNS`/
`XGB_COLUMNS`/tuned XGB config and the production 10/52/38 weights — same
code path as `make_submission.py` and `iterMERGE1_verify_yixi10/verify.py`
(both loaded as modules and reused verbatim, not reimplemented):

| Check | Reference | Reproduced | Delta |
|---|---:|---:|---:|
| LightGBM valid (seed 0) | 0.68834144 | 0.68834144 | 0.0 |
| XGBoost valid (seed 0) | 0.66755420 | 0.66755420 | 0.0 |
| FM valid (5-seed sigmoid-mean ensemble, unchanged) | 0.63987792 | 0.63987792 | 0.0 |
| Full blend valid (10/52/38) | 0.69943440 | 0.69943440 | 0.0 |
| Full blend test | 0.68432260 | 0.68432260 | 0.0 |

All five checks passed at exact (`<=1e-6`) tolerance — see
`results.json` -> `harness_fidelity` (`all_pass: true`). Environment: python
3.14.6, lightgbm 4.7.0, xgboost 3.4.1.

## Method

`run.py` trains LightGBM at seeds 0-4 and XGBoost at seeds 0-4 (identical
feature columns/config to the production model, only `random_state`
varies), reusing `make_submission.py`'s own `_fit_lgb`/`_fit_xgb`/
`train_one_fm`/`within_user_percentile` functions loaded as a module (no
reimplementation). The FM component is trained once as its normal unchanged
5-seed sigmoid-mean ensemble. For each of LGB and XGB, both a plain mean of
raw scores and a sigmoid-mean of raw scores (the FM convention) were
computed and compared standalone on valid/test, then the better-performing
convention per model was carried into the blend.

## Per-model seed variance

| Model | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | spread |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM valid | 0.68834144 | 0.68762863 | 0.68746448 | 0.68714005 | 0.68850547 | 0.00137 |
| XGBoost valid | 0.66755420 | 0.66755420 | 0.66755420 | 0.66755420 | 0.66755420 | 0.0 |

**XGBoost is exactly seed-invariant** — all 5 seeds produce byte-identical
predictions and metrics. This is explained by its tuned config
(`iterYIXI5_xgboost_optimization/results.json`): `subsample=1.0`,
`colsample_bytree=1.0`, `max_depth=1`, `tree_method=hist`. With no row or
column subsampling and depth-1 stumps, there is no source of randomness
`random_state` could act on — histogram construction and the single split
per tree are fully deterministic given the data. Seed-ensembling XGBoost is
therefore a structural no-op under this config, not merely an empirical
null.

LightGBM does have real seed variance (spread 0.00137 across seeds 0-4,
~1.4x the +0.001 promotion-delta threshold), but **seed 0 happens to be the
best of the five** (0.68834144, only beaten by seed 4's 0.68850547 by
+0.00016). Averaging in the three weaker seeds (1/2/3) pulls the ensemble
mean down to ~0.68782 — worse than seed 0 alone by about -0.00052.

| LightGBM ensembling | valid | test |
|---|---:|---:|
| seed 0 only (production) | 0.68834144 | — |
| plain mean of 5 seeds | 0.68781745 | 0.67249596 |
| sigmoid mean of 5 seeds | 0.68782210 | 0.67249548 |

Sigmoid-mean edges out plain-mean by +0.0000047 valid for LGB (noise-level);
for XGB the two conventions are identical since all seeds are identical.
Full comparison in `results.json` -> `seed_ensemble_comparison`.

## Re-blend at production weights (10% FM / 52% LGB / 38% XGB)

Using seed-ensembled LGB (sigmoid-mean) + seed-ensembled XGB (plain-mean,
= single-seed since XGB is seed-invariant) + the unchanged FM ensemble:

| | valid | test |
|---|---:|---:|
| Reference (seed-0 LGB/XGB, current production) | 0.69943440 | 0.68432260 |
| Seed-ensembled LGB/XGB, same 10/52/38 weights | 0.69868320 | 0.68336004 |
| **Delta** | **-0.00075120** | **-0.00096256** |

## Local grid search around 10/52/38 (VALID ONLY, seed-ensembled LGB/XGB)

49-point grid (±0.06 in steps of 0.02 on FM/LGB weights, XGB = remainder,
non-negative only). Best point found:

| fm | lgb | xgb | valid | test (reported, not used for selection) |
|---:|---:|---:|---:|---:|
| 0.10 | 0.56 | 0.34 | 0.69878483 | 0.68323874 |
| 0.08 | 0.56 | 0.36 | 0.69874012 | — |
| 0.10 | 0.54 | 0.36 | 0.69873524 | — |
| 0.10 | 0.52 | 0.38 (production weights) | 0.69868320 | 0.68336004 |

Even the best re-weighted point (0.69878483) recovers only about +0.0001 of
the -0.00075 lost by seed-ensembling at fixed weights, and still sits
**-0.00065 below** the reference blend (0.69943440). Full 49-point grid in
`results.json` -> `grid_search`.

## Conclusion

| Stage | valid primary | test primary |
|---|---:|---:|
| Reference (current production, seed-0 LGB/XGB + 5-seed FM) | 0.69943440 | 0.68432260 |
| Seed-ensembled LGB+XGB, production weights | 0.69868320 | 0.68336004 |
| Seed-ensembled LGB+XGB, best re-swept weights (0.10/0.56/0.34) | 0.69878483 | 0.68323874 |
| **Final delta vs. reference (valid, selection metric)** | **-0.00064957** | (test not used for selection) |

Seed-ensembling does **not** help here, and the mechanism is now understood
rather than just empirically null:

1. **XGBoost is exactly seed-invariant** under its tuned config (no
   subsampling, depth-1 stumps, deterministic histogram splits) — there is
   no variance to reduce, so ensembling it is a pure no-op.
2. **LightGBM does have seed variance** (~0.0014 spread across 5 seeds),
   but the production seed (0) happens to sit near the top of that
   distribution. Averaging in weaker seeds regresses the ensemble toward a
   lower mean, costing more valid primary than it recovers from variance
   reduction. This is the opposite of the outcome hoped for — it is a
   genuine finding, not just "no effect."
3. A local weight re-sweep around the seed-ensembled components recovers
   only a small fraction of the loss and does not come close to closing the
   gap to the reference blend.

**No promotion action taken.** This does not touch `SUBMISSION.md`/
`make_submission.py`/`submission.csv` (orchestrator-owned). The current
best remains yixi's 10% FM / 52% LGB / 38% XGB blend at **valid 0.69943440
/ test 0.68432260**, unchanged by this experiment. A caveat for future
work: this REJECT is specific to seed 0 being an above-average LightGBM
seed under this exact feature/config combination — it does not imply
seed-ensembling would never help a *different* LGB config (e.g. one with
`bagging_fraction`/`feature_fraction` < 1.0, which would introduce more
seed variance and a different production-seed luck profile).

## Artifacts

- `run.py` — single script: harness-fidelity check, 5-seed LGB/XGB training
  (reusing `make_submission.py`'s own `_fit_lgb`/`_fit_xgb`/`train_one_fm`
  as an imported module, no reimplementation), plain-mean vs sigmoid-mean
  ensemble comparison, production-weight re-blend, and local grid search.
  All stages append to `results.json` incrementally as they complete.
- `results.json` — `harness_fidelity`, `seed_ensemble_comparison`,
  `production_weights_seed_ensembled`, `grid_search`, `summary` (verdict).
- `run.log` — full training log (~19 minutes total runtime).
- No files outside this directory were modified.

# iterYIXI11 — Causal user-history-regime adaptive blending

## Verdict: REJECT

The predeclared adaptive blend improved the official validation primary from
`0.69943440` to `0.69962597`, a gain of only `+0.00019157`. This is below the
project's `+0.0003` preliminary threshold and far below the `+0.001`
promotion threshold. It therefore did not qualify for seed confirmation,
shifted-split model fitting, or test evaluation.

The strongest reference remains the promoted YIXI10 fixed within-user
percentile blend:

```text
10% FM / 52% LightGBM / 38% XGBoost
valid primary = 0.69943440
test primary  = 0.68432260  (inherited published YIXI10 result)
```

The adaptive candidate's test score is **not evaluated**. Test data was not
used for regime construction, weight selection, or this rejection decision.

## Experiment question

Does the relative value of FM, LightGBM, and XGBoost vary enough with the
amount of causal user history to justify different blend weights for a small
number of user regimes?

This experiment froze the promoted YIXI10 base models and used the same
within-user percentile score representation. It changed only the blend
weights assigned to predeclared user-history regimes.

## Files and separation of conditions

Following the iter44 structure, each condition has a separate entry point:

- `harness.py`: reproduce and freeze the official validation predictions.
- `phase_a_regime_diagnostics.py`: inspect component performance by regime and
  apply the predeclared model-ranking-reversal gate.
- `phase_b_adaptive_weights.py`: run constrained validation-only local weight
  sweeps after the Phase A gate passes.
- `phase_c_shifted_robustness.py`: enforce the preliminary-gain gate before any
  shifted-split confirmation.
- `diagnose.py`: verify regime exclusivity, weight constraints, local plateaus,
  unique-score fractions, and metric sanity checks.
- `features.py`: immutable wrapper around the promoted YIXI10 feature builder.
- `regimes.py`: label-free causal regime construction.

Machine-readable outputs are retained in `harness_results.json`,
`phase_a_results.json`, `phase_b_results.json`, `phase_c_results.json`, and
`diagnostic_results.json`.

## Harness-fidelity check

The exact promoted YIXI10 configurations and weights were reproduced before
testing adaptive blending:

| Reference | Reproduced valid primary | Required valid primary | Difference |
|---|---:|---:|---:|
| LightGBM | 0.68834144 | 0.68834144 | < 5e-9 |
| XGBoost | 0.66755420 | 0.66755420 | < 5e-9 |
| Fixed percentile blend | 0.69943440 | 0.69943440 | < 5e-9 |

The FM prediction vector was loaded from the same established artifact used by
YIXI10 and its component primary was `0.63987792`. Row alignment checks passed
before the prediction vectors were frozen.

## Causal regime definition

For each user, the segmentation variable is the number of interactions
strictly before the validation boundary. It uses only `user_id` and
`date`—never labels, current/future outcomes, or test performance.

The low/high threshold was fixed once from the official training distribution:

```text
training median among users with positive history = 31 interactions

no_prior: prior count = 0
low:      prior count = 1..31
high:     prior count > 31
```

Every validation user is assigned exactly one regime. The same assignment is
used for all rows belonging to that user.

| Regime | Validation users | Validation rows | Prior-count range | Median count |
|---|---:|---:|---:|---:|
| no_prior | 422 | 1,990 | 0 | 0 |
| low | 9,933 | 36,747 | 1–31 | 15 |
| high | 12,022 | 86,172 | 32–809 | 61 |

The training distribution contained 26,210 users; its prior-count quartiles
were approximately 13, 31, and 59. No threshold was repeatedly searched on
validation.

## Phase A — component performance by regime

Scores below use each component's native score vector and evaluate only users
inside the stated regime.

| Regime | Model | GAUC | nDCG@5 | Primary |
|---|---|---:|---:|---:|
| no_prior | FM | 0.75252551 | 0.54895288 | 0.65073919 |
| no_prior | LightGBM | 0.75855291 | 0.56792712 | 0.66324002 |
| no_prior | XGBoost | 0.76524597 | 0.57629454 | **0.67077029** |
| low | FM | 0.73704469 | 0.56322062 | 0.65013266 |
| low | LightGBM | 0.80132037 | 0.58528459 | **0.69330251** |
| low | XGBoost | 0.77470648 | 0.58021748 | 0.67746198 |
| high | FM | 0.70910209 | 0.56032652 | 0.63471431 |
| high | LightGBM | 0.76904559 | 0.60950595 | **0.68927574** |
| high | XGBoost | 0.74145740 | 0.58573347 | 0.66359544 |

The predeclared Phase A gate required a pair of models to reverse relative
order between regimes with at least `0.001` separation on both sides. It
passed: XGBoost beat LightGBM by `0.00753027` for `no_prior`, while LightGBM
beat XGBoost by `0.01584053` for `low` and `0.02568030` for `high`.

This was sufficient evidence to run the constrained adaptive-weight sweep. It
was not evidence by itself for promotion.

## Phase B — constrained local regime weights

Each regime used the same predeclared local grid around the fixed reference
`0.10 / 0.52 / 0.38`:

```text
each coordinate: reference weight +/- 0.10
grid step:        0.02
minimum weight:   0.02
sum of weights:   exactly 1.00
```

There were 85 admissible candidates per regime. Selection used validation
primary only.

| Regime | Selected FM/LGB/XGB | Fixed-weight primary | Selected primary | Delta | Candidates within 0.0003 of winner |
|---|---|---:|---:|---:|---:|
| no_prior | 0.16 / 0.42 / 0.42 | 0.67483163 | 0.68155646 | +0.00672483 | 2 / 85 |
| low | 0.08 / 0.54 / 0.38 | 0.70180666 | 0.70183516 | +0.00002849 | 12 / 85 |
| high | 0.10 / 0.56 / 0.34 | 0.70187581 | 0.70200032 | +0.00012451 | 11 / 85 |

The sizeable local improvement for `no_prior` is real on this validation
slice, but the slice contains only 1,990 of 124,909 validation rows. The
selected changes in the two much larger regimes are below the noise threshold.

Combining the three disjoint regime choices produced:

| System | GAUC | nDCG@5 | Primary | Delta vs fixed |
|---|---:|---:|---:|---:|
| Fixed percentile blend | 0.79415059 | 0.60471827 | 0.69943440 | — |
| Adaptive regime blend | 0.79431188 | 0.60494012 | 0.69962597 | **+0.00019157** |

Both metric components moved upward, but the total change is below the
`+0.0003` preliminary threshold.

## Phase C — shifted temporal robustness gate

The shifted-split regime construction was audited label-free using the
project's established earlier split:

```text
train: 2022-04-05 through 2022-04-18
valid: 2022-04-19 through 2022-04-25
```

Its independently training-derived positive-history median is 29, producing:

| Shifted regime | Users | Rows |
|---|---:|---:|
| no_prior | 533 | 2,634 |
| low | 10,096 | 40,474 |
| high | 12,187 | 100,286 |

No shifted models were fit and no shifted adaptive weights were selected.
Section 3 requires stopping changes that fail the preliminary validation
threshold rather than spending additional comparisons on validation noise.
Because the official candidate gained only `+0.00019157`, it was ineligible
for shifted confirmation. Consequently it cannot be promoted under the
prompt's stricter robustness requirement.

## Diagnostics and confound checks

- Regime exclusivity: every user and row belongs to exactly one regime.
- Threshold provenance: 31 was derived only from the official training
  distribution; the shifted audit independently derived 29 from shifted
  training.
- Weight constraints: all selected weights sum to one, retain every model,
  and remain within `0.10` per coordinate of the global reference.
- Within-user unique-score fraction: fixed `0.99896403`; adaptive
  `0.99734369`. The result is not created by widespread tied predictions.
- Grid-spike check: the low/high winners have broad sub-noise plateaus; the
  no-history winner has one neighboring candidate within `0.0003`, but its
  aggregate effect is too small for confirmation.
- Metric sanity: constant scores gave primary `0.48367125`; seeded random
  scores gave `0.48265904`, well below all learned systems.
- Test isolation: no adaptive test vector or adaptive test metric was
  generated.

## Interpretation

Model complementarity does vary with available history: XGBoost is strongest
for completely unseen users, whereas LightGBM is strongest once user history
exists. The effect is actionable only for the small `no_prior` segment in this
run. Globally, the segment-specific system adds just `0.00019157`, so adaptive
weighting introduces extra validation-selected degrees of freedom without a
meaningful confirmed gain.

The clean conclusion is **REJECT**. Retain the fixed YIXI10 blend and do not
carry the adaptive weights forward.

## Reproduction

From the repository root:

```shell
python3 experiments/iterYIXI11_user_regime_blend/harness.py
python3 experiments/iterYIXI11_user_regime_blend/phase_a_regime_diagnostics.py
python3 experiments/iterYIXI11_user_regime_blend/phase_b_adaptive_weights.py
python3 experiments/iterYIXI11_user_regime_blend/phase_c_shifted_robustness.py
python3 experiments/iterYIXI11_user_regime_blend/diagnose.py
```

Only `experiments/iterYIXI11_user_regime_blend/` was created or modified for
this experiment. No prohibited shared file was changed.

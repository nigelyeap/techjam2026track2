# iterYIXI1 — standalone XGBoost and the promoted three-model blend

## Hypothesis

iter44 showed that LightGBM only became competitive after it stopped reusing
the FM's 20-bucket encoding and received the continuous causal signals as raw
floats. XGBoost had never been tested in this project. The hypothesis was that
an `XGBRanker` trained on exactly the same native representation could either
beat iter44's LightGBM or contribute sufficiently different ranking errors to
improve the existing FM + LightGBM score blend.

## Outcome and promotion target

The experiment has two deliberately separate results:

| runner | question | valid | test | verdict |
|---|---|---:|---:|---|
| `run_experiment.py` | Does standalone XGBoost replace iter44 LightGBM? | 0.65864 | 0.64512 | **REJECT** |
| `blend.py` | Does FM + LightGBM + XGBoost beat the current blend? | **0.67006** | **0.65646** | **PROMOTE** |

**Only the latter—the 16% FM / 8% LightGBM / 76% XGBoost three-model
blend—is promoted.** The standalone XGBoost model is a useful diversity
component, but it is not itself the promoted 6a score.

## Harness-fidelity gate

Before writing or running the XGBoost experiment, the unmodified
`make_submission.py` was run end-to-end with output directed to
`/tmp/submission_check.csv`. It reproduced all three required references
exactly:

```text
GBM standalone: valid=0.66135 test=0.64794
FM ensemble standalone: valid=0.63988 test=0.64187
iter44 blend: valid primary=0.66473 test primary=0.65197
submit.py format/alignment check: PASSED (170588 rows)
```

The setup-correctness gate therefore passed before any new result was trusted.

## Method and runner separation

Following iter44's separation between `train.py` and `blend.py`, 6a now has
two executable paths:

- `run_experiment.py` tests only standalone XGBoost objectives and depths,
  records the tie diagnostic, and performs the frozen standalone test check;
- `blend.py` imports the fixed standalone winner, retrains the published FM
  and LightGBM references, performs the validation-only three-way weight
  sweep and five-seed confirmation, and performs the frozen blend test check.

`run_experiment.py` imports iter44's `prepare()` directly and asserts that the
input columns are exactly iter44's existing feature set. `blend.py` reuses
that same loader and assertion:

- native categoricals: `user_id`, `video_id`, `author_id`, `tab`, `last1`
- raw numeric features: `duration_ms`, `decay_rate_2.5`, `decay_act_2.5`,
  `decay_tab_3`, `lastk_rate`, `gap`

This was a model-family-only change: there were no new features, alternate
labels, split changes, or encoding changes. Categorical vocabularies remained
fit on train only through iter44's encoder, with unseen validation/test values
represented as missing.

The sweep used XGBoost 3.4.1 with native categorical handling and the following
fixed parameters:

```text
n_estimators=500, learning_rate=0.05, early_stopping_rounds=30
min_child_weight=1.0, reg_lambda=1.0
subsample=1.0, colsample_bytree=1.0
tree_method='hist', max_bin=256, enable_categorical=True
```

Both requested objectives (`rank:pairwise`, `rank:ndcg`) were tested at
`max_depth` 1, 2, 3, 4, 5, and 7. The internal early-stopping metric was
`ndcg@5-`: XGBoost's `-` form assigns zero to a query with no positive row,
matching `evaluate.py`; ordinary XGBoost `ndcg@5` assigns such queries a
different default value. All candidate selection still used the project's
official `evaluate.py` **validation primary**, not XGBoost's internal metric.

The objective/depth winner was frozen on validation in `run_experiment.py`.
The separate `blend.py` then fixed that configuration and ran a
three-component simplex sweep on validation using the established score
treatment: sigmoid-mean FM scores plus separately min-max-normalized LightGBM
and XGBoost scores. A full 0.10 grid was followed by a local 0.02 refinement.
Each runner withheld test scoring until its own selection was frozen. This
keeps the standalone decision and the promoted blend decision operationally
and conceptually separate.

## `run_experiment.py` — validation-only objective and depth sweep

| objective | max depth | best iteration | GAUC | nDCG@5 | primary |
|---|---:|---:|---:|---:|---:|
| `rank:pairwise` | **1** | 186 | 0.74186 | 0.57192 | **0.65689** |
| `rank:pairwise` | 2 | 85 | 0.74023 | 0.57274 | 0.65648 |
| `rank:pairwise` | 3 | 67 | 0.73220 | 0.56997 | 0.65109 |
| `rank:pairwise` | 4 | 39 | 0.72737 | 0.56835 | 0.64786 |
| `rank:pairwise` | 5 | 49 | 0.72432 | 0.56618 | 0.64525 |
| `rank:pairwise` | 7 | 46 | 0.71585 | 0.56287 | 0.63936 |
| `rank:ndcg` | **1** | 333 | 0.73929 | 0.57799 | **0.65864** |
| `rank:ndcg` | 2 | 167 | 0.73354 | 0.57563 | 0.65459 |
| `rank:ndcg` | 3 | 304 | 0.72386 | 0.56913 | 0.64650 |
| `rank:ndcg` | 4 | 182 | 0.71759 | 0.56662 | 0.64211 |
| `rank:ndcg` | 5 | 175 | 0.71762 | 0.56572 | 0.64167 |
| `rank:ndcg` | 7 | 151 | 0.71016 | 0.56185 | 0.63600 |

The same inverted-capacity trend as iter44 appears strongly: decision stumps
(`max_depth=1`) win for both objectives, and performance declines rapidly as
depth increases. `rank:ndcg` depth 1 was selected because its official valid
primary, 0.65864, was the best in the sweep. Relative to depth-1 pairwise, it
trades some GAUC for a larger nDCG gain and improves primary by 0.00175.

## `run_experiment.py` — standalone result: REJECT

After the validation choice was frozen, the selected XGBoost model was checked
on test once:

| model | valid | test |
|---|---:|---:|
| FM five-seed ensemble reference | 0.63988 | 0.64187 |
| XGBoost `rank:ndcg`, depth 1 | **0.65864** | **0.64512** |
| LightGBM iter44 reference | 0.66135 | 0.64794 |

XGBoost is competitive and clearly stronger than the FM ensemble as a
standalone model, but it is below LightGBM by 0.00272 valid and 0.00282 test.
It is therefore **REJECTED as a standalone replacement** for iter44.

## `blend.py` — three-way result: PROMOTE

The fixed references reproduced their published validation numbers before the
weight sweep:

| component/reference | valid primary |
|---|---:|
| FM ensemble | 0.6398779 |
| native LightGBM | 0.6613549 |
| current 10% FM / 90% LightGBM blend | 0.6647331 |

The coarse grid winner was 20% FM / 10% LightGBM / 70% XGBoost at 0.66951.
The validation-only 0.02 refinement selected:

```text
16% FM / 8% LightGBM / 76% XGBoost
valid: GAUC=0.7549679, nDCG@5=0.5851620, primary=0.6700650
```

This is **+0.005332 valid** over the reproduced current blend. The gain is
balanced across both metric components: current/new GAUC is
0.749627/0.754968 and current/new nDCG@5 is 0.579839/0.585162.

After the weights were frozen, the single final test check produced:

| model | valid | test |
|---|---:|---:|
| current FM + LightGBM blend | 0.66473 | 0.65197 |
| **new FM + LightGBM + XGBoost blend** | **0.67006** | **0.65646** |
| delta | **+0.00533** | **+0.00449** |

## Verification 1 — tie/row-order artifact

The mandatory low-capacity tie diagnostic on validation gave:

```text
all-constant scores: primary=0.48367
random scores:       primary=0.48266
XGBoost winner:      mean within-user unique-score fraction=0.9090
                     10,139 unique scores over 124,909 rows
```

The all-tied baseline remains at the expected random floor, not at the model's
score. Although the stump ensemble has somewhat more ties than iter44's
LightGBM, more than 90% of scores within an average user's group are unique,
and both GAUC and nDCG are well above the trivial baselines. The gain is not an
incidental stable-sort inheritance from raw row order.

## Verification 2 — no silently-added confound

The runner asserts the DataFrame columns equal `iter44.CAT_COLS +
iter44.NUM_COLS` before training and imports iter44's train-fitted categorical
encoder rather than reimplementing it. No feature was added or removed. This
isolates the experimental axis to LightGBM versus XGBoost and rules out the
kind of `duration_ms`-style accidental representation confound investigated in
iter44.

## Verification 3 — five-seed promotion check

Because the blend gain cleared the required +0.001 valid threshold, `blend.py`
reran the chosen configuration and **fixed** weights for XGBoost seeds 0
through 4:

```text
seed 0: valid=0.67006499
seed 1: valid=0.67006499
seed 2: valid=0.67006499
seed 3: valid=0.67006499
seed 4: valid=0.67006499
mean:   valid=0.67006499, std=0.00000000
```

The identical results are expected because row and column subsampling are both
1.0, so the selected histogram learner has no active stochastic path. The
explicit five-seed check nevertheless confirms there is no seed-dependent
split or initialization behavior, and the +0.00533 gain clears the promotion
threshold on every seed.

The standalone model did not receive an unnecessary five-seed confirmation:
it was 0.00272 below, rather than at least 0.001 above, the LightGBM standalone
reference.

## Verification 4 — blend gain is a broad plateau, not a grid spike

Useful validation ablations from the same predeclared simplex sweep were:

| blend family | best weights (FM/LGB/XGB) | valid primary |
|---|---:|---:|
| FM + LightGBM | 0.10 / 0.90 / 0.00 | 0.66473 |
| LightGBM + XGBoost | 0.00 / 0.10 / 0.90 | 0.66563 |
| FM + XGBoost | 0.20 / 0.00 / 0.80 | 0.66929 |
| all three | 0.16 / 0.08 / 0.76 | **0.67006** |

The nearest refined weights form a smooth plateau: 0.16/0.06/0.78 scores
0.67004, 0.14/0.10/0.76 scores 0.67000, and 0.16/0.10/0.74 scores 0.66998.
Most of the improvement already exists in the independently useful FM +
XGBoost two-way blend, while the small LightGBM weight adds another 0.00077.
The result is therefore not dependent on one isolated weight tuple.

## Diagnosis

XGBoost does not beat LightGBM standalone because LightGBM's stump boosting
extracts slightly stronger total signal from this representation. The
objective comparison suggests the two implementations distribute ranking
quality differently: XGBoost `rank:ndcg` improves top-of-list nDCG relative to
its pairwise objective while conceding some GAUC.

That difference is valuable to the ensemble. The XGBoost model receives most
of the final blend weight even though it is weaker standalone, while the
bucketed FM contributes a smaller but strongly complementary component and
LightGBM supplies a final residual gain. In other words, XGBoost is not the
best replacement model; it is the best new diversity source tested here.

## Final verdict: PROMOTE the three-way blend

**PROMOTE** 16% FM / 8% native LightGBM / 76% native XGBoost:

- valid **0.67006**, up **0.00533** from 0.66473
- test **0.65646**, up **0.00449** from 0.65197
- fixed-weight valid gain reproduced across all five requested seeds
- tie artifact and feature-confound checks passed
- nearby weights and the FM + XGBoost ablation confirm a broad, real blend gain

Keep the verdict boundary explicit:

- `run_experiment.py`: standalone XGBoost is **REJECT**;
- `blend.py`: the FM + LightGBM + XGBoost combination is **PROMOTE** and
  establishes the new best result for this experiment.

The promotion applies to the **three-model blend only**, not to the standalone
XGBoost score.

## Artifacts

- `run_experiment.py`: standalone XGBoost validation sweep and frozen test
- `results.json`: standalone XGBoost metrics and explicit `REJECT` verdict
- `blend.py`: fixed-component validation weight sweep, confirmation, and
  frozen blend test
- `blend_results.json`: three-model blend metrics and explicit
  `PROMOTE_THREE_MODEL_BLEND` verdict

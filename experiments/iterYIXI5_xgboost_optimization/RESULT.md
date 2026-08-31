# iterYIXI5 — XGBoost optimization and ensemble calibration

## Final verdict: PROMOTE BOTH; use the percentile-normalized ensemble

Section 6e produced two independently promotable validation findings:

1. Lower-rate XGBoost training improves the post-6d standalone reference from
   **0.66649055** to **0.66755420** (**+0.00106364** valid). The paired gain
   repeats in all five requested seed fits.
2. The tuned XGBoost is most useful in a within-user percentile-normalized
   blend with the current FM and iter63 LightGBM components. The selected
   24% FM / 40% LightGBM / 36% XGBoost blend reaches **0.68237525**, a
   five-seed-confirmed **+0.00631076** over the locally reproduced current
   ensemble score of 0.67606449.

After every validation decision and confirmation was frozen, the standalone
model scored **0.65323293** on test and the selected ensemble scored
**0.66948622**. The ensemble test delta against the current 0.65955281 system
is **+0.00993341**. Test performance did not select, reject, rescue, or reorder
any configuration.

The primary promotion target is the percentile-normalized three-model
ensemble. The lower-rate standalone XGBoost configuration is also a confirmed
finding, but its score is not the overall project score.

## Runner separation and scope

Following iter44's training/blending separation, this directory contains two
independent executable paths:

- `run_experiment.py` reproduces the post-6d XGBoost reference, performs only
  the ordered model-parameter phases, confirms the frozen standalone winner,
  and then performs its one final standalone test evaluation.
- `blend.py` reads the frozen tuning result, rebuilds the current FM and
  iter63 LightGBM components, sweeps global and within-user-normalized blend
  weights on validation, confirms fixed weights/normalization, and then
  performs its final test evaluation.

No shared experiment, submission, ledger, or write-up file was modified.

## Fixed feature representation

All XGBoost fits use exactly iterYIXI4's promoted 5-day user-decay
replacement. The 11 columns are:

```text
user_id, video_id, author_id, tab, last1,
duration_ms, decay_tab_3, lastk_rate, gap,
decay_rate_5, decay_act_5
```

Relative to the original 6a representation, `decay_rate_2.5` and
`decay_act_2.5` are removed and the unchanged 6b definitions of
`decay_rate_5` and `decay_act_5` are added. The feature width remains 11.
No rejected 6d cross, author/video popularity, user-tab timescale, or
cross-family composition feature is in the model matrix.

The builder is imported from
`experiments/iterYIXI2_feature_depth/features.py`. Its independent causal
checks passed again with global maximum absolute error **1.48e-12**, including
explicit exclusion of same-date matching rows. Label and user alignment are
asserted on every split.

## Harness-fidelity gates

Before writing the new runner, the unmodified mandatory harness was run with
output directed to `/tmp/yixi6e_submission_check.csv`. It passed the submission
format/alignment check and reproduced the current iter63 system:

```text
LightGBM standalone: valid=0.67168 test=0.65353
FM standalone:       valid=0.63988 test=0.64187
iter63 blend:        valid=0.67606 test=0.65955
```

The standalone runner then reproduced the exact selected 6d XGBoost score
before tuning:

| metric | iterYIXI4 published | 6e reproduction |
|---|---:|---:|
| GAUC | 0.7504962087 | 0.7504962087 |
| nDCG@5 | 0.5824848413 | 0.5824848413 |
| primary | **0.6664905548** | **0.6664905548** |

The blend runner independently reproduced the tuned XGBoost, current iter63
LightGBM, and FM scores exactly. Recombining FM and LightGBM locally gives
0.67606449 rather than iter63's published 0.67606294, a difference of only
1.55e-6 caused by floating-point ranking ties across environments. This is
within the instruction's allowed numeric noise. To be conservative, every 6e
ensemble delta and confirmation uses the slightly stronger local
**0.67606449** reference.

## Validation-only sequential tuning protocol

The initial configuration was the exact post-6d XGBoost ranker:

```text
objective='rank:ndcg', eval_metric='ndcg@5-'
max_depth=1, learning_rate=0.05, n_estimators=500
early_stopping_rounds=30
min_child_weight=1, gamma=0, reg_lambda=1, reg_alpha=0
subsample=1, colsample_bytree=1
tree_method='hist', max_bin=256
enable_categorical=True, max_cat_to_onehot=4
```

The phases were run in the prompt's order. Every candidate in a phase was
created from that phase's starting incumbent; there was no Cartesian grid.
The raw validation winner was carried only if its official validation-primary
gain was at least 0.0003. Otherwise the axis was abandoned and the incumbent
was kept.

| phase | best tested alternative | valid | delta vs phase start | decision |
|---|---|---:|---:|---|
| tree capacity | `max_depth=2` | 0.65240633 | -0.01408422 | retain depth 1 |
| rate × trees | lr 0.025, 1000 trees, patience 120 | **0.66755420** | **+0.00106364** | **carry** |
| `min_child_weight` | 0.5 | 0.66755420 | +0.00000000 | exact null; abandon |
| `gamma` | 0.01 | 0.66755420 | +0.00000000 | exact null; abandon |
| `reg_lambda` | 0.25 | 0.66707814 | -0.00047606 | abandon |
| `reg_alpha` | 0.01 | 0.66737962 | -0.00017458 | abandon |
| local L1 diagnostic | 0.075 | 0.66715097 | -0.00040323 | abandon |
| row stochasticity | `subsample=0.9` | 0.66172969 | -0.00582451 | abandon |
| feature stochasticity | `colsample_bytree=0.9` | 0.66753006 | -0.00002414 | noise; abandon |

### Boosting-rate diagnosis

The rate/tree phase exposed a real training-path confound that was diagnosed
before selection was finalized:

| learning rate | tree cap | patience | best iteration | valid |
|---:|---:|---:|---:|---:|
| 0.05 reference | 500 | 30 | 340 | 0.66649055 |
| 0.10 | 250 | 30 | 248 | 0.66420919 |
| 0.075 | 400 | 30 | 309 | 0.66724879 |
| 0.025 | 1000 | 30 | 148 | 0.62846732 |
| **0.025** | **1000** | **120** | **791** | **0.66755420** |

The first lower-rate run stopped after only 148 rounds and was badly
underfit. It would have been incorrect to interpret that score as a failure of
lower learning rates. Scaling patience allowed the intended proportionally
longer training path and produced the phase winner. The selected change is
therefore exactly:

```text
learning_rate:          0.05 -> 0.025
n_estimators:            500 -> 1000
early_stopping_rounds:    30 -> 120
```

Every other model parameter and every feature remains at the post-6d
reference. The later L1 check also showed why sequential selection matters:
`reg_alpha=0.1` helped the old 0.05 path in an initial diagnostic, but became
negative after the lower-rate schedule was carried. It was not composed into
the final model.

## Standalone XGBoost result and confirmation

| model | GAUC | nDCG@5 | primary | delta |
|---|---:|---:|---:|---:|
| post-6d h5 reference | 0.75049621 | 0.58248484 | 0.66649055 | — |
| tuned lower-rate XGBoost | **0.75199854** | **0.58310986** | **0.66755420** | **+0.00106364** |

Both metric components improve: +0.00150234 GAUC and +0.00062501 nDCG@5.
The result is not carried by one side of the official average.

The required paired validation confirmation was:

| seed | reference | tuned | paired delta |
|---:|---:|---:|---:|
| 0 | 0.66649055 | 0.66755420 | +0.00106364 |
| 1 | 0.66649055 | 0.66755420 | +0.00106364 |
| 2 | 0.66649055 | 0.66755420 | +0.00106364 |
| 3 | 0.66649055 | 0.66755420 | +0.00106364 |
| 4 | 0.66649055 | 0.66755420 | +0.00106364 |

Mean/min delta is +0.00106364 and the standard deviation is zero. Identical
seeds are expected because full row and column sampling remain enabled, but
all five requested refits were run explicitly.

The selected model emits 15,776 distinct validation scores and has mean
within-user unique-score fraction 0.9363. Constant and seeded-random valid
rankings score only 0.48367125 and 0.48265904. Together with the exact column
audit and balanced metric movement, this rules out stable-sort ties or a
silently added feature as the source of the gain.

## Ensemble reference and validation-only calibration

The ensemble runner uses three independently aligned components:

| component | valid primary |
|---|---:|
| unchanged iter38 FM five-seed ensemble | 0.63987792 |
| current iter63 `rate_only` LightGBM | 0.67167872 |
| tuned 6e XGBoost | 0.66755420 |
| current 14% FM / 86% LightGBM reference | **0.67606449** |

Two normalization families were predeclared and evaluated independently:

- **Global:** FM retains its sigmoid probability; LightGBM and XGBoost are
  min-max normalized independently over the split, matching the established
  project blend treatment.
- **Within-user percentile:** each component is independently converted to
  average-tie percentile ranks within the current user's candidate rows.
  This transformation reads only model scores and user IDs—not labels—and is
  performed separately on validation and test. It cannot move rows between
  users or use future/history labels.

Each family used a full 0.10 validation simplex followed by a 0.02 local
refinement around its best coarse three-model region. All three weights were
required to be positive for the selected three-model candidate; zero-weight
points were retained as diagnostic ablations.

| normalization | weights FM / LGB / XGB | valid | delta vs 0.67606449 |
|---|---:|---:|---:|
| global | 0.06 / 0.78 / 0.16 | 0.67706347 | +0.00099897 |
| **within-user percentile** | **0.24 / 0.40 / 0.36** | **0.68237525** | **+0.00631076** |

The global result is approximately +0.001 but misses the exact local
confirmation trigger by 1.03e-6; it is not rounded into a separate promotion.
The percentile result is validation-selected and clears the threshold
unambiguously.

## Ensemble artifact, ablation, and confound checks

### XGBoost is necessary

The large gain is not merely a normalization-only improvement to the current
two-model system:

| within-user-percentile ablation | best weights | valid |
|---|---:|---:|
| FM + LightGBM, no XGBoost | 0.30 / 0.70 / 0.00 | 0.67457372 |
| LightGBM + XGBoost, no FM | 0.00 / 0.40 / 0.60 | 0.67907828 |
| FM + XGBoost, no LightGBM | 0.30 / 0.00 / 0.70 | 0.67554665 |
| **all three** | **0.24 / 0.40 / 0.36** | **0.68237525** |

The no-XGBoost ablation is below the existing 0.67606449 system. LightGBM +
XGBoost is independently positive, and FM supplies a further +0.00330 when
all three are composed. The selected result therefore genuinely depends on
the XGBoost diversity source and on three-model composition.

### Broad weight plateau

The winner is not an isolated grid spike. Nearby fixed weights include:

| FM / LGB / XGB | valid |
|---:|---:|
| 0.24 / 0.40 / 0.36 | **0.68237525** |
| 0.24 / 0.34 / 0.42 | 0.68237007 |
| 0.24 / 0.42 / 0.34 | 0.68232292 |
| 0.22 / 0.42 / 0.36 | 0.68229949 |
| 0.26 / 0.36 / 0.38 | 0.68225193 |

### Ties and metric balance

The selected blend's mean within-user unique-score fraction is **0.9988**.
Although percentile conversion intentionally creates a common component
scale, weighted composition almost completely resolves component ties rather
than inheriting input row order.

Against the current ensemble, validation GAUC improves from 0.76506764 to
0.77264076 (**+0.00757313**) and nDCG@5 improves from 0.58706141 to
0.59210974 (**+0.00504833**). The primary gain is balanced across both official
metrics.

## Five-seed fixed-calibration confirmation

Normalization and the 0.24 / 0.40 / 0.36 weights were selected once at seed
0 and then held fixed:

| XGBoost seed | blend valid | delta vs current reference |
|---:|---:|---:|
| 0 | 0.68237525 | +0.00631076 |
| 1 | 0.68237525 | +0.00631076 |
| 2 | 0.68237525 | +0.00631076 |
| 3 | 0.68237525 | +0.00631076 |
| 4 | 0.68237525 | +0.00631076 |

Mean/min delta is +0.00631076, standard deviation is zero, and every seed
clears +0.001. This satisfies Section 3 without recalibrating weights per
seed.

## Frozen one-time test evaluation

Only after standalone/ensemble choices, confirmations, diagnostics, and
verdicts were frozen were test predictions computed:

| system | valid primary | test primary |
|---|---:|---:|
| post-6d standalone XGBoost | 0.66649055 | 0.65197098 |
| tuned standalone XGBoost | **0.66755420** | **0.65323293** |
| current FM + LightGBM | 0.67606449 | 0.65955281 |
| selected percentile FM + LightGBM + XGBoost | **0.68237525** | **0.66948622** |

The frozen standalone test delta is +0.00126195. The frozen ensemble test
delta is +0.00993341, with GAUC +0.00945598 and nDCG@5 +0.01041079. These test
movements support the validation finding but did not participate in model,
normalization, weight, or promotion selection.

## Conclusion

**PROMOTE** the lower-rate standalone XGBoost configuration as a confirmed
model improvement, and **PROMOTE as the final system** the within-user
percentile blend:

```text
24% unchanged FM five-seed ensemble
40% current iter63 rate_only LightGBM
36% tuned h5 XGBoost
```

The main optimization lesson is that the 6a/6d XGBoost was close on structural
parameters—depth, split thresholds, L2/L1 regularization, and sampling did not
help—but its lower-rate schedule required proportionally longer early-stopping
patience. The main ensemble lesson is stronger: global score calibration is
only borderline, while within-user percentile calibration exposes substantial
and robust complementary ordering among all three models.

## Artifacts

- `run_experiment.py`: fixed-feature sequential tuning, paired confirmation,
  diagnostics, and frozen standalone test stage
- `results.json`: exact configs, every phase candidate, feature provenance,
  causal report, confirmations, diagnostics, and standalone metrics
- `blend.py`: current-reference reproduction, two normalization sweeps,
  fixed-weight confirmation, ablations, and frozen ensemble test stage
- `blend_results.json`: every weight candidate, components, confirmations,
  diagnostics, selected ensemble, and final metrics
- `RESULT.md`: this report

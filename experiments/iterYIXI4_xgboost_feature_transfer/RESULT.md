# iterYIXI4 — transfer of 6b feature families to XGBoost-native

## Final verdict: PROMOTE the 5-day user-decay replacement

The Section 6b null result does **not** fully generalize from native LightGBM
to native XGBoost. With the promoted 6a XGBoost configuration held fixed, two
unchanged 6b families independently produced confirmed validation gains:

- replacing the 2.5-day user decay pair with the 5-day pair improved valid
  primary from **0.65863872** to **0.66649055** (**+0.00785184**);
- adding `decay_rate_2.5 * log1p(decay_act_2.5)` improved valid primary to
  **0.66386497** (**+0.00522625**).

Both deltas repeated in every one of five requested seed fits. The families
did not compose: their union scored only 0.65881252 (+0.00017381), below the
preliminary threshold. Validation therefore selected the 5-day user-decay
replacement. After selection was frozen, its one test evaluation scored
**0.65197098**, a **+0.00684977** gain over the 6a standalone XGBoost test
reference.

Author/video popularity-decay remains a clean rejection. Thus feature utility
is model-family dependent for user timescale and cross features, but not for
the global popularity features.

## Hypothesis and controlled change

Section 6b tested additional decay timescales, causal author/video popularity,
and three predeclared crosses on iter44's native LightGBM representation. It
found no promotable result after verification. Section 6a subsequently
promoted an XGBoost-native ranker as part of the best blend, leaving open
whether the 6b features were intrinsically unhelpful or interacted differently
with XGBoost.

This experiment changed only the learner. `run_experiment.py` imports:

- the XGBoost constructor directly from
  `experiments/iterYIXI1_xgboost_native/run_experiment.py`;
- the feature builder directly from
  `experiments/iterYIXI2_feature_depth/features.py`.

No 6b feature definition was copied, redesigned, or edited. The recorded
SHA-256 hashes were:

```text
6a model builder: aca1c2d2b1ff409ce3a63b234bfd74825fa15c1c2f8c0bccc033d5b185e2dbea
6b feature builder: bef9dab34fc7c159f561ed95f1ca2542627857ec0d54104f9934378d68a90b14
```

## Harness-fidelity gate

Before the transfer sweep, a validation-only harness loaded the exact 6a data
path, native columns, grouping logic, and promoted XGBoost constructor. It
reproduced the published 6a standalone winner exactly:

| metric | 6a published | 6d reproduced |
|---|---:|---:|
| GAUC | 0.7392888665 | 0.7392888665 |
| nDCG@5 | 0.5779885054 | 0.5779885054 |
| primary | **0.6586387157** | **0.6586387157** |
| best iteration | 333 | 333 |

The complete 6d runner repeated that gate before fitting any feature
candidate and asserted absolute primary drift below `1e-8`. It passed.

## Fixed XGBoost configuration

Every reference, candidate, and confirmation fit used the exact 6a winner:

```text
XGBRanker(
    objective='rank:ndcg', eval_metric='ndcg@5-',
    max_depth=1, n_estimators=500, learning_rate=0.05,
    min_child_weight=1.0, reg_lambda=1.0, reg_alpha=0.0,
    subsample=1.0, colsample_bytree=1.0,
    tree_method='hist', max_bin=256,
    enable_categorical=True, max_cat_to_onehot=4,
    early_stopping_rounds=30, random_state=seed,
    n_jobs=-1
)
```

The training and validation rows were stably sorted by user for XGBoost's
query groups. Predictions were evaluated in original row order with the
official `evaluate.py`. No hyperparameter was retuned for any feature family.

## Unchanged feature definitions and causal verification

The three families were the exact 6b implementations:

1. User rate/activity at half-lives 1, 5, 7, and 14 days, as replacements for
   the established 2.5-day pair and as parallel additions; user-tab positive
   decay at 1, 5, 7, and 14 days, as replacements for 3 days and in parallel.
2. Author/video 2.5-day historical rate and activity. Rate retained the 6b
   Laplace form `(decayed_pos + 0.5) / (decayed_total + 1)`.
3. `decay_rate / (duration_ms + 1)`,
   `decay_rate * log1p(decay_activity)`, and the train-fitted user-activity
   tier × tab categorical interaction.

The imported feature builder repeated its independent direct-sum checks on
three validation rows for every historical family. Each direct calculation
used matching rows from strictly earlier dates only.

| history family | checked half-lives | maximum absolute error |
|---|---|---:|
| user decay rate/activity | 1, 5, 7, 14 | 1.78e-14 |
| user-tab positive decay | 1, 5, 7, 14 | 5.33e-15 |
| author decay rate/activity | 2.5 | 1.48e-12 |
| video decay rate/activity | 2.5 | 1.48e-12 |

All errors were below `1e-10`. Across the checks, 626 matching same-date rows
were present and explicitly excluded. The causal gate passed with no leakage
or row-alignment failure.

## Validation-only selection and verification protocol

- Every predeclared candidate was first fit at seed 0 and ranked only by
  official validation primary.
- A family winner needed at least **+0.0003** to be revisited.
- Every preliminary-positive family winner was refit against the paired
  reference at seeds 0 through 4. Promotion required a validation gain of at
  least **+0.001 across all five seeds** and in the mean.
- Within-family unions were attempted only for individually preliminary-positive
  members. Cross-family composition was allowed only after two families had
  independently passed full confirmation.
- Test performance was neither computed nor inspected during selection. One
  selected test score was computed only after confirmations, composition, and
  diagnostics had frozen the choice.

Because the 6a configuration uses full row and column sampling, its five seed
fits are deterministic. The five requested fits were nevertheless run rather
than inferred.

## Family 1 — additional decay timescales

Each user candidate adds both the smoothed rate and decayed activity at the
stated half-life. `Replacement` removes the original 2.5-day pair;
`parallel` retains it.

| user half-life | replacement valid | delta | parallel valid | delta |
|---:|---:|---:|---:|---:|
| 1 day | 0.65351117 | -0.00512755 | 0.65863872 | +0.00000000 |
| **5 days** | **0.66649055** | **+0.00785184** | **0.66649055** | **+0.00785184** |
| 7 days | 0.66541404 | +0.00677532 | 0.66541404 | +0.00677532 |
| 14 days | 0.65772629 | -0.00091243 | 0.65772629 | -0.00091243 |

The preliminary-positive h5+h7 parallel union scored 0.66541404
(+0.00677532), no better than h7 alone. In the h5 and h7 parallel models,
XGBoost assigned zero gain to the original 2.5-day rate/activity and produced
the same model metrics as replacement. This isolates the useful change as
the longer user rate rather than extra dimensionality. For h5 replacement,
`decay_rate_5` received 11.02% of total gain and `decay_act_5` received zero.

The user-tab results were all null or negative:

| tab half-life | replacement valid | delta | parallel valid | delta |
|---:|---:|---:|---:|---:|
| 1 day | 0.65227520 | -0.00636351 | 0.65890974 | +0.00027102 |
| 5 days | 0.65117151 | -0.00746721 | 0.65131754 | -0.00732118 |
| 7 days | 0.63829863 | -0.02034008 | 0.63851380 | -0.02012491 |
| 14 days | 0.63792646 | -0.02071226 | 0.63815236 | -0.02048635 |

The best tab result, 1 day in parallel, missed the +0.0003 gate by 0.00002898,
so no tab union or user+tab union was allowed. The family winner was the h5
user replacement at **+0.00785184**.

## Family 2 — author/video popularity-decay

| added feature(s) | valid | delta | added rate/activity gain |
|---|---:|---:|---:|
| author rate | 0.60862124 | -0.05001748 | 26.79% / — |
| author activity | 0.65863872 | +0.00000000 | — / 0.00% |
| author rate + activity | 0.60862124 | -0.05001748 | 26.79% / 0.00% |
| video rate | 0.60784876 | -0.05078995 | 24.28% / — |
| video activity | 0.65863872 | +0.00000000 | — / 0.00% |
| video rate + activity | 0.60784876 | -0.05078995 | 24.28% / 0.00% |

Activity alone was ignored and exactly reproduced the reference. Global rate
attracted roughly a quarter of all split gain but badly damaged both ranking
components. This strengthens 6b's diagnosis: low-depth rankers can find global
popularity tempting even though it conflicts with the task's within-user
ordering. Neither author nor video passed the preliminary gate, so they were
not combined and this family was rejected without confirmation.

## Family 3 — pairwise/cross features

| added cross | GAUC | nDCG@5 | primary | delta | gain share |
|---|---:|---:|---:|---:|---:|
| `decay_rate / (duration_ms + 1)` | 0.74203748 | 0.57944137 | 0.66073942 | +0.00210071 | 13.20% |
| **`decay_rate * log1p(decay_activity)`** | **0.74330729** | **0.58442259** | **0.66386497** | **+0.00522625** | **15.47%** |
| activity tier × tab | 0.74228722 | 0.57958221 | 0.66093469 | +0.00229597 | 22.19% |
| all three preliminary-positive crosses | 0.73707074 | 0.58166176 | 0.65936625 | +0.00072753 | — |

All three individual crosses passed the preliminary gate under XGBoost, but
their union did not preserve the gains. The rate×log-activity cross was the
family winner and entered five-seed confirmation unchanged.

## Required paired five-seed confirmations

| family | seed | reference valid | candidate valid | paired delta |
|---|---:|---:|---:|---:|
| 5-day user decay | 0 | 0.65863872 | 0.66649055 | +0.00785184 |
| 5-day user decay | 1 | 0.65863872 | 0.66649055 | +0.00785184 |
| 5-day user decay | 2 | 0.65863872 | 0.66649055 | +0.00785184 |
| 5-day user decay | 3 | 0.65863872 | 0.66649055 | +0.00785184 |
| 5-day user decay | 4 | 0.65863872 | 0.66649055 | +0.00785184 |
| rate×log-activity cross | 0 | 0.65863872 | 0.66386497 | +0.00522625 |
| rate×log-activity cross | 1 | 0.65863872 | 0.66386497 | +0.00522625 |
| rate×log-activity cross | 2 | 0.65863872 | 0.66386497 | +0.00522625 |
| rate×log-activity cross | 3 | 0.65863872 | 0.66386497 | +0.00522625 |
| rate×log-activity cross | 4 | 0.65863872 | 0.66386497 | +0.00522625 |

For h5, mean/min delta was +0.00785184 with standard deviation 0. For the
cross, mean/min delta was +0.00522625 with standard deviation 0. Both exceed
+0.001 in every requested seed and are confirmed promotable findings.

This contrasts directly with 6b LightGBM: its h5 user feature was negative,
while its rate×log-activity seed-0 gain failed confirmation because three of
five paired deltas were negative. XGBoost's fixed full-sampling stump path is
both stronger on these features and stable across the requested seeds.

## Allowed cross-family composition

Only after both families had independently confirmed was their union tested:

| configuration | valid | delta | decision |
|---|---:|---:|---|
| h5 user replacement + rate×log-activity | 0.65881252 | +0.00017381 | below preliminary gate |

Both added rates received substantial split gain in the union (12.56% for
`decay_rate_5`, 13.99% for the cross), but the official metric did not retain
either standalone gain. The likely diagnosis is redundant/competing
expressions of user confidence consuming the capacity of a one-split-per-tree
ranker. The union was not confirmed or selected.

## Artifact and confound checks

The h5 gain is large enough to require extra scrutiny:

- **Exact column audit:** replacement has the same 11-column width as the 6a
  reference. Its only changes are removal of `decay_rate_2.5` and
  `decay_act_2.5`, and addition of the unchanged 6b `decay_rate_5` and
  `decay_act_5`. The parallel form reaches exactly the same metrics while
  assigning zero gain to the old pair. No hidden feature was added.
- **Causality:** independent direct sums passed, including explicit exclusion
  of same-date matching rows.
- **Seed check:** all five paired refits reproduced the full gain exactly.
- **Tie check:** all-constant valid scores produced primary 0.48367125 and a
  seeded random ranking produced 0.48265904. The selected model produced
  13,236 distinct scores across 124,909 validation rows and a mean within-user
  unique-score fraction of 0.9324. Its score is not inherited from stable-sort
  row order.
- **Metric balance:** versus reference, h5 improves validation GAUC by
  +0.01120734 and nDCG@5 by +0.00449634; the primary gain is not a one-metric
  artifact.

These checks support a genuine model-family interaction rather than leakage,
row-order ties, seed luck, model retuning, or an accidental feature confound.

## Frozen one-time test evaluation

The validation winner and PROMOTE decision were frozen before test scoring.
The selected seed-0 h5 replacement was then evaluated on test exactly once:

| model | split | GAUC | nDCG@5 | primary |
|---|---|---:|---:|---:|
| 6a XGBoost reference | valid | 0.73928887 | 0.57798851 | 0.65863872 |
| selected h5 replacement | valid | 0.75049621 | 0.58248484 | **0.66649055** |
| 6a XGBoost reference | test | 0.71106535 | 0.57917708 | 0.64512122 |
| selected h5 replacement | test | 0.71995062 | 0.58399141 | **0.65197098** |

The frozen test delta is +0.00888526 GAUC, +0.00481433 nDCG@5, and
**+0.00684977 primary**. This is reported only as an out-of-sample check; it
did not select, reject, rescue, or reorder any candidate.

## Conclusion

**PROMOTE the unchanged 6b 5-day user-decay replacement for the standalone
6a XGBoost-native ranker.** The unchanged rate×log-activity cross is also an
independently confirmed XGBoost-positive finding, but it is not selected
because h5 is stronger and their union fails to compose.

The precise transfer conclusion is:

- the 6b user-timescale null was LightGBM-specific;
- the 6b cross-family non-promotion was also LightGBM-specific;
- the 6b author/video popularity null generalizes to both native tree-ranker
  families;
- user-tab timescales remain non-promotable;
- independently useful low-depth features are not necessarily additive.

## Artifacts

- `run_experiment.py`: fixed-model, validation-only sweep, confirmation,
  diagnostics, optional composition, and frozen final test stage
- `results.json`: exact feature provenance, causality records, every candidate,
  gain fractions, seed confirmations, diagnostics, and final metrics
- `RESULT.md`: this report

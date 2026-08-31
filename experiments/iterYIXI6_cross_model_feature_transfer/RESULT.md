# iterYIXI6 — cross-model feature transfer

## Final verdict: REJECT for the final ensemble

The feature transfers improve both tree models standalone, but they do not
improve the current YIXI5 within-user-percentile ensemble:

| result | current valid | transferred valid | delta | standalone decision |
|---|---:|---:|---:|---|
| XGBoost: tab count → tab rate | 0.66755420 | **0.66976142** | **+0.00220722** | confirmed positive |
| LightGBM: 2.5-day → 5-day user pair | 0.67167872 | **0.67311722** | **+0.00143850** | confirmed positive |
| LightGBM: add unchanged rate×log-activity cross | 0.67167872 | 0.67058140 | -0.00109732 | reject |
| final FM + LightGBM + XGBoost | **0.68237525** | 0.68191737 | **-0.00045788** | **REJECT** |

Phase A and Phase B each pass the full paired five-seed standalone criterion.
Nevertheless, the final ensemble is the prompt's promotion boundary, and its
validation score is lower even after local weight re-optimization. The current
YIXI5 24% FM / 40% LightGBM / 36% XGBoost system therefore remains the best
validated system.

After all validation choices and the REJECT verdict were frozen, the 6f
candidate scored **0.67228580** on test versus YIXI5's 0.66948622
(+0.00279957). That favorable test movement is recorded, but validation-only
discipline forbids using it to rescue or promote the candidate.

## File and condition separation

Following iter44's separation of training and blending conditions, 6f uses:

- `harness.py`: YIXI5 component and final-blend validation fidelity gate;
- `phase_a_xgb_tab_rate.py`: XGBoost tab-rate transfer only;
- `phase_b_lgb_h5.py`: LightGBM user-timescale transfer only;
- `phase_c_lgb_cross.py`: LightGBM cross-feature retest only;
- `composition.py`: gated representation composition/selection;
- `blend.py`: frozen-representation percentile calibration, final verdict,
  and the sole 6f candidate test stage;
- `diagnose_diversity.py`: post-selection validation-only explanation of the
  ensemble result, with no effect on selection.

No existing experiment, submission, ledger, or shared write-up file was
modified.

## Mandatory and experiment-level harness gates

Before new code was written, the unmodified repository harness was run as:

```text
python3 make_submission.py /tmp/yixi6f_submission_check.csv
```

It reproduced the current iter63 path and passed submission alignment:

```text
LightGBM standalone: valid=0.67168 test=0.65353
FM standalone:       valid=0.63988 test=0.64187
iter63 blend:        valid=0.67606 test=0.65955
submit.py format/alignment check: PASSED
```

Before any 6f transfer was selected, `harness.py` then rebuilt every YIXI5
validation component through the new unified feature frame:

| component | published YIXI5 | 6f reproduction |
|---|---:|---:|
| tuned XGBoost | 0.66755420 | 0.66755420 |
| iter63 LightGBM | 0.67167872 | 0.67167872 |
| unchanged FM ensemble | 0.63987792 | 0.63987792 |
| 24/40/36 percentile blend | **0.68237525** | **0.68237525** |

All four scores reproduced within `1e-8`. Test predictions were not computed
by this gate.

## Exact feature provenance and causality

`features.py` imports iter63's data module and calls its established causal
functions directly. It does not modify or redesign either discovery:

- `decay_tab_3` is iter63's 3-day decayed positive count;
- `decay_tab_rate_3` is iter63's exact Laplace-smoothed
  `(decayed_tab_pos + 0.5) / (decayed_tab_total + 1.0)`;
- `decay_rate_5` and `decay_act_5` are produced by the same imported
  date-grouped user-decay function used for the YIXI4 5-day finding;
- the Phase C cross is exactly
  `decay_rate_2.5 * log1p(decay_act_2.5)`.

The unified builder rebuilt iter63 rows without writing a cache into an
existing experiment directory. A regeneratable cache lives only inside the
new 6f directory.

Independent direct sums checked three validation rows for both the 5-day user
positive/total histories and 3-day user-tab positive/total histories. Every
sum used matching rows from strictly earlier dates only.

| causal check | result |
|---|---:|
| maximum absolute error | **7.11e-15** |
| same-date matching rows explicitly excluded | 39 |
| label/user/split alignment | passed |

This independently verifies both historical transfer features rather than
trusting their implementation by inspection alone.

## Validation-only selection and verification policy

- A0/A1/A2, B0/B1/B2, and C0/C1 were selected only by official standalone
  validation primary.
- A change needed at least +0.0003 to survive its independent phase.
- A standalone gain of at least +0.001 triggered paired five-seed
  confirmation; confirmation required mean and minimum paired deltas of at
  least +0.001.
- The B+C union was allowed only if B and C each independently cleared
  +0.0003. There was no blind union.
- Standalone-selected representations were frozen before ensemble weight
  calibration. Ensemble diagnostics could not change them.
- The final ensemble required a confirmed validation improvement of at least
  +0.001 over 0.68237525. A negative seed-0 delta correctly skipped
  confirmation and produced REJECT.
- No phase runner accessed test predictions. `blend.py` first accessed test
  only after representation/weight selection, diagnostics, confirmation
  eligibility, and the verdict were frozen.

## Phase A — iter63 tab rate transferred to tuned XGBoost

Every Phase A fit held the full YIXI5 XGBoost configuration fixed:

```text
rank:ndcg, max_depth=1
learning_rate=0.025, n_estimators=1000
early_stopping_rounds=120
min_child_weight=1, gamma=0, lambda=1, alpha=0
subsample=1, colsample_bytree=1
5-day user-decay replacement
```

| candidate | tab feature(s) | best iteration | GAUC | nDCG@5 | primary | delta |
|---|---|---:|---:|---:|---:|---:|
| A0 | `decay_tab_3` | 791 | 0.75199854 | 0.58310986 | 0.66755420 | — |
| **A1** | **`decay_tab_rate_3` replaces count** | **971** | **0.75475472** | **0.58476818** | **0.66976142** | **+0.00220722** |
| A2 | count + rate | 791 | 0.75199854 | 0.58310986 | 0.66755420 | +0.00000000 |

A1 improves both GAUC (+0.00275618) and nDCG@5 (+0.00165832). The selected
tab rate receives 14.86% of XGBoost gain. In A2, however, the added tab rate
receives exactly zero gain and the result is bit-identical to A0. This mirrors
iter63's earlier observation: replacement exposes the normalized signal,
whereas the learner retains the older count when both compete.

### Phase A paired confirmation

| seed | A0 reference | A1 candidate | paired delta |
|---:|---:|---:|---:|
| 0 | 0.66755420 | 0.66976142 | +0.00220722 |
| 1 | 0.66755420 | 0.66976142 | +0.00220722 |
| 2 | 0.66755420 | 0.66976142 | +0.00220722 |
| 3 | 0.66755420 | 0.66976142 | +0.00220722 |
| 4 | 0.66755420 | 0.66976142 | +0.00220722 |

Mean/min delta is +0.00220722 with zero standard deviation. Full sampling
makes exact seed equality expected, but all five refits were run. A1 emits
24,984 distinct validation scores with mean within-user unique-score fraction
0.9418; constant and seeded-random rankings score only 0.48367125 and
0.48265904. The standalone transfer is real and not a stable-sort tie artifact.

**Phase A finding: confirmed positive standalone.**

## Phase B — YIXI4 5-day user pair transferred to current LightGBM

Every Phase B fit held iter63's current architecture and tab rate fixed:

```text
linear_tree=True, num_leaves=2
learning_rate=0.10, n_estimators=500
min_child_samples=200, reg_lambda=1
decay_tab_rate_3
```

| candidate | user timescale | best iteration | GAUC | nDCG@5 | primary | delta |
|---|---|---:|---:|---:|---:|---:|
| B0 | current 2.5-day pair | 48 | 0.75887352 | 0.58448392 | 0.67167872 | — |
| **B1** | **replace with 5-day pair** | **47** | **0.76094317** | **0.58529127** | **0.67311722** | **+0.00143850** |
| B2 | retain 2.5-day + add 5-day | 47 | 0.76094317 | 0.58529127 | 0.67311722 | +0.00143850 |

B1 improves GAUC by +0.00206965 and nDCG@5 by +0.00080734. B1 and B2 are
exactly identical. In B2, the old 2.5-day rate/activity and both activity
features receive zero gain, while `decay_rate_5` receives 16.06%. The useful
change is the longer continuous rate, not feature width; validation therefore
selects the simpler B1 replacement.

### Phase B paired confirmation

| seed | B0 reference | B1 candidate | paired delta |
|---:|---:|---:|---:|
| 0 | 0.67167872 | 0.67311722 | +0.00143850 |
| 1 | 0.67105150 | 0.67416626 | +0.00311476 |
| 2 | 0.67104304 | 0.67415625 | +0.00311321 |
| 3 | 0.67104304 | 0.67415625 | +0.00311321 |
| 4 | 0.67105150 | 0.67415625 | +0.00310475 |

Mean paired delta is +0.00277689, minimum is +0.00143850, and all five seeds
clear the required threshold. This updates the earlier LightGBM-specific
conclusion: the 5-day pair was unhelpful under older LightGBM conditions, but
does transfer after the move to linear leaves, learning rate 0.10, and the
tab-rate representation.

**Phase B finding: confirmed positive standalone.**

## Phase C — unchanged cross on current LightGBM

| candidate | GAUC | nDCG@5 | primary | delta | cross gain share |
|---|---:|---:|---:|---:|---:|
| C0 current LightGBM | 0.75887352 | 0.58448392 | **0.67167872** | — | — |
| C1 add exact rate×log-activity | 0.75553596 | 0.58562684 | 0.67058140 | **-0.00109732** | 13.13% |

The cross attracts substantial split gain and slightly improves nDCG@5
(+0.00114292), but damages GAUC more (-0.00333756), producing a negative
official-primary result. The current linear-tree architecture does not reverse
the earlier LightGBM null.

**Phase C: REJECT.** It fails the +0.0003 gate and receives no confirmation.

## Gated composition

The only possible same-model union was Phase B + Phase C. Phase C was
independently negative, so `composition.py` correctly skipped this union.
There was no blind combination of all transferred features.

The frozen standalone-selected representations passed to the ensemble were:

- XGBoost: A1, 5-day user pair plus `decay_tab_rate_3` replacing
  `decay_tab_3`;
- LightGBM: B1, `decay_tab_rate_3` plus the 5-day pair replacing the 2.5-day
  pair;
- no Phase C cross.

## Final within-user-percentile ensemble

The current YIXI5 reference was reconstructed again immediately before the
new sweep:

```text
24% FM / 40% current LightGBM / 36% current XGBoost
valid primary = 0.68237525
```

All representations were frozen from standalone validation. The ensemble
runner reused YIXI5's exact average-tie within-user percentile transformation,
ran a 0.10 simplex, and refined the best local three-model region at 0.02.

### Attribution at the current 24/40/36 weights

| tree representation(s) changed | valid | delta vs YIXI5 |
|---|---:|---:|
| none | **0.68237525** | — |
| XGBoost A1 only | 0.68133044 | -0.00104481 |
| LightGBM B1 only | 0.68219340 | -0.00018185 |
| both A1 + B1 | 0.68176603 | -0.00060922 |

Both standalone improvements reduce ensemble validation score at the existing
weights. This is direct evidence that model accuracy alone is insufficient;
the current ensemble depends on complementary errors/orderings.

### Locally re-optimized weights

| eligible diagnostic | best FM / LGB / XGB | valid | delta vs YIXI5 |
|---|---:|---:|---:|
| XGBoost A1 only | 0.18 / 0.30 / 0.52 | 0.68167442 | -0.00070083 |
| LightGBM B1 only | 0.24 / 0.40 / 0.36 | 0.68219340 | -0.00018185 |
| **frozen A1 + B1 candidate** | **0.22 / 0.36 / 0.42** | **0.68191737** | **-0.00045788** |

Reweighting recovers only +0.00015134 over the both-transfer fixed-weight
score. It cannot recover the YIXI5 reference. The final candidate also loses
both official validation components: GAUC -0.00061005 and nDCG@5 -0.00030571.
Its mean within-user unique-score fraction is 0.9994, so the loss is not a tie
artifact.

Because the final validation delta is negative, no five-seed ensemble
confirmation was run. Confirmation is a verification gate for a qualifying
gain, not a mechanism to search for a lucky rescue.

## Accuracy versus diversity diagnosis

`diagnose_diversity.py` was run after the representations, weights, test
stage, and REJECT verdict were frozen. It did not participate in selection and
used validation only.

The diagnostic converts each tree's scores to the same within-user percentile
values used by the ensemble and measures pooled Pearson rank-score
correlation:

| tree pair | rank correlation |
|---|---:|
| current iter63 LGB vs current YIXI5 XGB | 0.78685857 |
| transferred B1 LGB vs transferred A1 XGB | **0.81338526** |
| change | **+0.02652669** |

The transferred XGBoost retains correlation 0.92849 with its current version;
the transferred LightGBM retains correlation 0.95569 with its current
version. Yet the two selected tree models become materially more correlated
with each other.

This gives a concrete diagnosis:

- **model accuracy improves:** both tree standalones gain and confirm;
- **model diversity worsens:** their percentile rankings converge;
- **final blend declines:** the lost complementarity outweighs the standalone
  accuracy gains on validation.

## Frozen one-time test evaluation

Only after all validation decisions were frozen did `blend.py` predict test:

| model/system | current test | transferred test | delta |
|---|---:|---:|---:|
| XGBoost standalone | 0.65323293 | 0.65550005 | +0.00226712 |
| LightGBM standalone | 0.65352744 | 0.66269004 | +0.00916260 |
| percentile ensemble | 0.66948622 | **0.67228580** | +0.00279957 |

The positive test movements are notable but cannot override the negative final
validation delta. Reporting them does not change the decision.

## Conclusion

6f answers the cross-transfer question in two parts:

1. **The feature discoveries do transfer at standalone level.** XGBoost
   benefits from replacing the tab count with iter63's tab rate, and current
   LightGBM benefits from replacing the 2.5-day pair with YIXI4's 5-day pair.
   Both findings are causal and five-seed confirmed.
2. **They should not replace the current YIXI5 ensemble components.** The
   transferred tree rankings are more correlated, and the best recalibrated
   blend is 0.00045788 below YIXI5 on validation.

Therefore the overall verdict is **REJECT** and the current YIXI5 system
remains unchanged:

```text
24% FM
40% iter63 LightGBM with 2.5-day user pair + decay_tab_rate_3
36% tuned YIXI5 XGBoost with 5-day user pair + decay_tab_3
within-user percentile normalization
valid 0.68237525 / test 0.66948622
```

The asymmetry in the current ensemble is useful diversity rather than merely
technical debt: making the two tree representations individually stronger and
more alike reduces the validation-selected combination.

## Artifacts

- `features.py`: exact imported feature union, row alignment, cache, and
  independent causality checks
- `common.py`: fixed model configurations, fit helpers, thresholds, and exact
  column sets
- `harness.py`, `harness_results.json`: YIXI5 validation fidelity gate
- `phase_a_xgb_tab_rate.py`, `phase_a_results.json`: A0/A1/A2 and confirmation
- `phase_b_lgb_h5.py`, `phase_b_results.json`: B0/B1/B2 and confirmation
- `phase_c_lgb_cross.py`, `phase_c_results.json`: C0/C1 result
- `composition.py`, `representation_results.json`: gated composition decision
  and frozen representation selection
- `blend.py`, `blend_results.json`: percentile sweeps, attribution,
  diagnostics, verdict, and frozen test metrics
- `diagnose_diversity.py`, `diversity_results.json`: post-selection rank
  correlation diagnosis
- `RESULT.md`: this report

# iterYIXI9 — Causal historical watch-depth features

## Verdict: PROMOTE

The 5-day exponentially decayed mean of strictly historical, clipped watch
fraction is a confirmed LightGBM feature and produces a new best three-model
ensemble:

| system | FM / LGB / XGB weights | valid GAUC | valid nDCG@5 | valid primary | test GAUC | test nDCG@5 | test primary |
|---|---:|---:|---:|---:|---:|---:|---:|
| post-6h reference | 0.24 / 0.42 / 0.34 | 0.77445179 | 0.59385890 | 0.68415534 | 0.74980152 | 0.59894353 | 0.67437255 |
| **selected 6i ensemble** | **0.18 / 0.46 / 0.36** | **0.78442186** | **0.59734720** | **0.69088453** | **0.75533485** | **0.60150570** | **0.67842031** |
| delta | — | **+0.00997007** | **+0.00348830** | **+0.00672919** | **+0.00553334** | **+0.00256217** | **+0.00404775** |

The branch and weights were selected entirely on validation. Test prediction
and evaluation were first performed after the selection, confirmation,
diagnostics, and `PROMOTE` verdict had been frozen to `ensemble_results.json`.

The promoted system changes only the LightGBM component: it adds
`hist_watch_decay_mean_5`. The final ensemble intentionally retains the
post-6h A0 XGBoost representation (`decay_tab_3`), because either confirmed
watch-depth XGBoost branch reduced the ensemble relative to the LightGBM-only
candidate.

## Harness fidelity

Before writing experiment code, the mandatory repository harness was run:

```text
python3 make_submission.py /tmp/yixi6i_submission_check.csv

LightGBM standalone       valid 0.67168 / test 0.65353
FM ensemble standalone    valid 0.63988 / test 0.64187
iter63 blend              valid 0.67606 / test 0.65955
submission alignment      PASSED (170588 rows)
```

`harness.py` then reproduced all exact post-6h validation references at a
`1e-8` tolerance before the feature comparisons:

| reference | role | valid primary | result |
|---|---|---:|---|
| YIXI7 LightGBM B1 | strongest current LightGBM | 0.67689133 | exact |
| YIXI6 XGBoost A1 | strongest current standalone XGBoost | 0.66976142 | exact |
| YIXI7 percentile ensemble | current final system | 0.68415534 | exact |

The distinction between XGBoost references is important. A1
(`decay_tab_rate_3`) is the strongest standalone representation, while the
current final ensemble retained A0 (`decay_tab_3`, standalone 0.66755420) for
better diversity. Independent XGBoost feature tests therefore use A1, while
the exact final-ensemble baseline uses A0.

## Watch-fraction policy, frozen before model scores

The raw distribution was inspected before choosing any clipping policy:

| statistic | raw `play_time_ms / duration_ms` |
|---|---:|
| total rows | 1,436,609 |
| rows with `duration_ms > 0` | 1,407,735 |
| rows with `duration_ms == 0` | 28,874 |
| median / p75 / p90 | 0.10042 / 0.63661 / 1.03666 |
| p95 / p99 / p99.9 | 1.15102 / 2.08974 / 5.04243 |
| maximum | 68.31013 |
| fraction above 1 | 15.2528% |

Completion is naturally saturated at 1 and the raw distribution has a real
upper tail, so the predeclared value is:

```text
watch_fraction = clip(play_time_ms / duration_ms, 0, 1)
```

Rows with zero duration remain interactions for temporal/window ordering but
do not update a watch-value numerator or denominator. There are no negative
play-time or duration values. The four candidates remain raw continuous
features:

| feature | exact definition | valid coverage |
|---|---|---:|
| `hist_watch_decay_mean_2.5` | user historical decayed mean, 2.5-day half-life | 99.6285% |
| `hist_watch_decay_mean_5` | user historical decayed mean, 5-day half-life | 99.6285% |
| `hist_watch_last5_mean` | mean of defined values among last 5 prior interactions | 99.6285% |
| `hist_watch_tab_decay_mean_3` | matching user/tab historical decayed mean, established 3-day tab half-life | 95.7969% |

## Causality and leakage verification

The implementation orders each user's events by `time_ms`. Every feature for
a timestamp is read from the state first; only after every row at that exact
timestamp has read the snapshot is the state updated. Therefore:

- current-row `play_time_ms` never reaches its own feature;
- interactions with smaller timestamps on the same date are included;
- equal-timestamp interactions are mutually excluded;
- later same-date interactions are excluded;
- exact-timestamp ties use `orig_idx` only to determine their position in
  future last-five windows, never to let tied rows see one another.

An independent brute-force implementation recomputed all four features using
only rows satisfying `prior.time_ms < current.time_ms` for 16 randomly selected
and edge-case validation rows. This explicitly exercised same-date histories
and real timestamp ties. Maximum absolute error was `2.78e-16`.

The data contain 90,386 real exact-timestamp user groups covering 187,073
rows. Four such validation cases were explicitly checked. A synthetic tie
stress case also passed: two tied first interactions both received missing
history, and the later interaction received their mean. These checks are
recorded row by row in `feature_results.json`.

## Independent LightGBM results

All candidates add exactly one column to the current YIXI7 rank-specific
LightGBM and keep every model parameter fixed (`linear_tree=True`,
`num_leaves=2`, learning rate 0.10, up to 500 trees,
`lambdarank_truncation_level=50`, `sigmoid=2`).

| candidate | valid GAUC | valid nDCG@5 | valid primary | delta | best iteration | candidate gain fraction |
|---|---:|---:|---:|---:|---:|---:|
| reference | 0.76465243 | 0.58913028 | 0.67689133 | — | 40 | — |
| 2.5-day decayed mean | 0.76090950 | 0.58638626 | 0.67364788 | -0.00324345 | 59 | 15.93% |
| **5-day decayed mean** | **0.76751000** | 0.58893478 | **0.67822242** | **+0.00133109** | 40 | 15.93% |
| last-5 mean | 0.76404095 | 0.58793080 | 0.67598587 | -0.00090545 | 36 | 0.00% |
| per-tab 3-day mean | 0.75722301 | 0.58365875 | 0.67044091 | -0.00645041 | 45 | 9.68% |

Only the 5-day feature clears confirmation. Its five paired seed deltas are:

```text
+0.00133109, +0.00144023, +0.00183827, +0.00140047, +0.00158358
mean = +0.00151873; minimum = +0.00133109
```

It primarily improves standalone GAUC; standalone nDCG@5 is nearly flat. On
the one post-selection test evaluation, LightGBM primary moves from
`0.66578388` to `0.66622883` (`+0.00044495`): GAUC rises by `+0.00165939`
while nDCG@5 falls by `-0.00076956`. The much larger final-ensemble gain is
therefore a complementarity result, not merely the standalone test delta.

## Independent XGBoost results

Every candidate adds one column to the strongest standalone A1
representation and holds the tuned YIXI5 configuration fixed:
`rank:ndcg`, `max_depth=1`, learning rate 0.025, 1000 trees,
early stopping 120, and full row/column sampling.

| candidate | valid GAUC | valid nDCG@5 | valid primary | delta | best iteration | candidate gain fraction |
|---|---:|---:|---:|---:|---:|---:|
| reference A1 | 0.75475472 | 0.58476818 | 0.66976142 | — | 999 | — |
| **2.5-day decayed mean** | 0.75418729 | **0.58765006** | **0.67091870** | **+0.00115728** | 910 | 15.27% |
| **5-day decayed mean** | 0.75448924 | **0.58770168** | **0.67109549** | **+0.00133407** | 999 | 14.66% |
| last-5 mean | 0.75207561 | 0.58279061 | 0.66743314 | -0.00232828 | 998 | 11.03% |
| per-tab 3-day mean | 0.74682599 | 0.58253109 | 0.66467857 | -0.00508285 | 995 | 15.34% |

Both user-level decayed means clear standalone confirmation. The XGBoost
configuration is deterministic across `random_state` because it uses full
row and feature sampling, so all five prescribed refits reproduce the exact
same paired deltas (`+0.00115728` and `+0.00133407`, respectively). This is
documented as deterministic repetition, not independent stochastic evidence.

Unlike LightGBM, both XGBoost gains come from nDCG@5 while GAUC decreases
slightly. Test was not used to choose between these branches and neither
unselected XGBoost candidate was evaluated on test.

## Ensemble selection

Only independently confirmed branches entered composition. Every eligible
branch used within-user average-tie percentiles, a coarse 0.10 validation
simplex, then a local 0.02 refinement. The XGBoost A1 representation without
any watch feature was also scored as a diagnostic and was negative
(`0.68394625`, delta `-0.00020909`); it was not an eligible new branch.

| eligible branch | primary at old weights | old-weight delta | locally selected weights FM/LGB/XGB | best valid primary | delta |
|---|---:|---:|---:|---:|---:|
| **LGB 5-day only, keep XGB A0** | **0.69043803** | **+0.00628269** | **0.18 / 0.46 / 0.36** | **0.69088453** | **+0.00672919** |
| XGB A1 + 2.5-day only | 0.68572736 | +0.00157201 | 0.16 / 0.46 / 0.38 | 0.68580210 | +0.00164676 |
| XGB A1 + 5-day only | 0.68483388 | +0.00067854 | 0.18 / 0.42 / 0.40 | 0.68521154 | +0.00105619 |
| LGB 5-day + XGB A1 2.5-day | 0.68784535 | +0.00369000 | 0.20 / 0.44 / 0.36 | 0.68846184 | +0.00430650 |
| LGB 5-day + XGB A1 5-day | 0.68791533 | +0.00375998 | 0.20 / 0.44 / 0.36 | 0.68840587 | +0.00425053 |

The LightGBM-only branch wins. The two independently positive model changes
do not compose additively: replacing ensemble XGBoost A0 with A1 plus a watch
feature reduces useful diversity relative to retaining A0.

The selected weight point is not an isolated spike. Seventeen searched points
are within `0.0003` of the winner; examples include:

```text
FM/LGB/XGB       valid primary      distance from best
0.18/0.46/0.36   0.69088453         0
0.20/0.44/0.36   0.69085121        -0.00003332
0.16/0.44/0.40   0.69083047        -0.00005406
0.20/0.46/0.34   0.69082147        -0.00006306
0.18/0.44/0.38   0.69079214        -0.00009239
```

## Promotion verification

### Five-seed paired confirmation

The branch and weights were frozen at seed 0. The same fixed weights were
then compared with paired reference/candidate LightGBM seeds:

| seed | reference valid | candidate valid | delta |
|---:|---:|---:|---:|
| 0 | 0.68415534 | 0.69088453 | +0.00672919 |
| 1 | 0.68425941 | 0.69095004 | +0.00669062 |
| 2 | 0.68395257 | 0.69096386 | +0.00701129 |
| 3 | 0.68418711 | 0.69081426 | +0.00662714 |
| 4 | 0.68430072 | 0.69083291 | +0.00653219 |

Mean delta is `+0.00671809`, standard deviation `0.00016104`, and minimum
delta `+0.00653219`.

### Tie and stable-sort artifact check

| scores | overall unique values | mean within-user unique fraction |
|---|---:|---:|
| reference LightGBM | 122,609 / 124,909 | 98.64% |
| candidate LightGBM | 122,030 / 124,909 | 98.07% |
| reference ensemble | 18,588 / 124,909 | 99.94% |
| candidate ensemble | 14,267 / 124,909 | 99.94% |

The all-constant baseline scores `0.48367125` primary and uniform random seed
0 scores `0.48265904`, confirming original row order is at the trivial floor.
The candidate is not heavily tied, and both final GAUC and nDCG improve.

### Confound check

- Candidate versus reference LightGBM columns differ by exactly one column:
  `hist_watch_decay_mean_5`.
- No `play_time_ms`, `long_view`, label, or target column reaches either model.
- All model parameters, training rows, labels, and other feature columns are
  identical.
- The gain is already `+0.00628269` at the unchanged reference weights; local
  weight calibration contributes only another `+0.00044650`.
- The selected feature accounts for 15.93% of LightGBM split gain, so it is
  actively used rather than changing behavior through a missing-value or
  column-order accident.
- The independent timestamp-causality audit passes before any model fitting.

## Interpretation

Long-view history discards engagement depth. A smoothed 5-day historical
completion signal adds substantial ranking information, especially for
LightGBM's whole-user ordering. The shorter or noisier summaries are not
universally useful: 2.5-day LightGBM, last-five, and per-tab features all
regress, while XGBoost converts both decayed user features mainly into
top-of-list nDCG gains.

The final result is model- and ensemble-specific. The promoted change is not
"add watch history everywhere"; it is:

```text
LightGBM: add 5-day historical clipped-watch-fraction mean
XGBoost:  keep the existing ensemble A0 representation
Blend:    within-user percentile, 0.18 FM / 0.46 LGB / 0.36 XGB
```

## Reproduction

Run from the repository root, in order:

```bash
python3 experiments/iterYIXI9_watch_depth_history/inspect_features.py
python3 experiments/iterYIXI9_watch_depth_history/harness.py
python3 experiments/iterYIXI9_watch_depth_history/phase_lgb.py
python3 experiments/iterYIXI9_watch_depth_history/phase_xgb.py
python3 experiments/iterYIXI9_watch_depth_history/ensemble.py
python3 experiments/iterYIXI9_watch_depth_history/diagnose_artifacts.py
```

Runners are split by condition, following the iter44 structure. Machine
readable outputs are `feature_results.json`, `harness_results.json`,
`phase_lgb_results.json`, `phase_xgb_results.json`, `ensemble_results.json`,
and `artifact_results.json`.

All implementation and outputs are confined to
`experiments/iterYIXI9_watch_depth_history/`. No prohibited shared file was
modified.

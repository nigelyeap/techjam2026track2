# iterXUXIA3 — reopening multi-task learning under the GBM-native representation

## Hypothesis
Multi-task learning (auxiliary heads on `is_like`/`is_follow`/`is_comment`/
`is_forward`) was rejected twice under FM's shared-embedding architecture
(iter31, iter36), diagnosed as a **shared absolute-score conflict**
between the rank-invariant BPR loss and base-rate-calibrated pointwise
auxiliary losses. That diagnosis is specific to architectures where
multiple losses share one absolute score — it doesn't obviously apply to
the GBM-native ranker, which has no shared embedding table. Tested
approach 1 from the instructions (auxiliary **features** via stacking, not
auxiliary **losses**): train small LightGBM models to predict
`is_like`/`is_follow`/`is_comment`/`is_forward`, feed their leakage-free
out-of-fold predictions back into the main `long_view` GBM ranker as 4
new input columns.

## Baseline note
Same as iterXUXIA1/2: baseline to beat is iter63's GBM standalone
(valid=0.67168/test=0.65353), this clone's actual `HEAD`, not iter44's.

## Method
- `aux_features.py`: for each of the 4 auxiliary tasks, trained a
  `LGBMClassifier(objective='binary', num_leaves=2, linear_tree=True,
  lr=0.10, n_estimators=300)` on iter63's exact native feature set
  (categoricals + `decay_rate_2.5`/`decay_act_2.5`/`decay_tab_rate_3`/
  `last1`/`lastk_rate`/`gap`/`duration_ms`).
  - **Train rows**: 5-fold out-of-fold predictions (a row's aux-feature
    value comes from a fold model that never saw that row).
  - **Valid/test rows**: predictions from a model fit on 100% of train
    (never touches valid/test in training).
- `train_with_aux.py`: injected the resulting 4 columns
  (`aux_is_like`/`aux_is_follow`/`aux_is_comment`/`aux_is_forward`) into
  iter63's unmodified `train.py`/`run()` code path via its `_cache=`
  parameter — identical LightGBM hyperparameters, only the input
  DataFrame's column set differs between baseline and aux-augmented runs.

## Causality / leakage argument
- Inputs to the 4 auxiliary models are the same native features the main
  model already uses. None of them encode the *current row's own* label
  (long_view or otherwise) — the decay features are strictly decayed
  counts of *past* long_view events, already causal by iter24/44's
  construction — so reusing them to predict a *different* same-row label
  introduces no new leakage.
- Train-row aux features are 5-fold OOF (leak-free by construction: no
  fold model is ever asked to score a row it was trained on).
- Valid/test aux features come from a model that never saw valid/test
  during training — no leakage path exists.
- Alignment (do `load_aux_labels`' per-split row order and iter63's own
  row order actually match, not just by docstring argument?) was verified
  with an independent spot-check: re-parsed the raw CSVs from scratch and
  compared `video_id`/`tab` at 30 random row positions per split against
  `dfs[...]` at the same position. All matched exactly. (`user_id` also
  checked, with one expected exception: a valid-set row whose `user_id`
  never appeared in train legitimately becomes `NaN` under the existing
  `pd.CategoricalDtype(categories=train.unique())` encoding — a
  pre-existing artifact of iter63's own encoding, not a row-order bug;
  confirmed by `video_id`/`tab` still matching exactly at that row.)
  ```
  alignment spot-check PASSED (90 rows across train/valid/test, exact video_id/tab match;
  1 rows had an out-of-vocabulary user_id -> NaN, expected pre-existing encoding behavior,
  user_id matched exactly on the rest)
  ```

## Harness-fidelity check
```
=== harness-fidelity check (baseline, no aux features, seed=0) ===
[rate_only] best_iteration=48  valid=0.67168  test=0.65353
baseline standalone: valid=0.67168 test=0.65353  [expect ~0.67168/0.65353, matching iter63 rate_only]
```

## Result (seed 0)
```
=== seed=0 single-run comparison: baseline vs. +aux features ===
[rate_only] best_iteration=48  valid=0.67192  test=0.65396
baseline: valid=0.67168 test=0.65353
+aux:     valid=0.67192 test=0.65396  (delta valid=+0.00024)
```

## Feature importance (aux-augmented model, seed 0; `best_iteration=48`, so 48 total splits)
```
  tab                      15
  decay_rate_2.5           11
  author_id                 9
  video_id                  8
  lastk_rate                3
  duration_ms               1
  aux_is_like               1  <-- aux
  user_id                   0
  last1                     0
  decay_act_2.5             0
  gap                       0
  decay_tab_rate_3          0
  aux_is_follow             0  <-- aux
  aux_is_comment            0  <-- aux
  aux_is_forward            0  <-- aux
```

## Diagnosis
The single-seed gain (+0.00024 valid) is below Section 3's 0.0003
"even look twice" threshold, so per protocol no 5-seed confirmation was
run — this is the correct, protocol-consistent stopping point, not a
shortcut. Feature importance independently confirms the same conclusion
from a different angle: across all 48 trees actually built (early
stopping triggered at `best_iteration=48`, well short of the
`n_estimators=500` cap), `aux_is_like` was chosen as a split feature
exactly once, and `aux_is_follow`/`aux_is_comment`/`aux_is_forward` were
never chosen at all — the model had access to all 4 auxiliary signals and
essentially declined to use 3 of them, and used the 4th only marginally.
This is consistent with the auxiliary engagement signals carrying little
incremental information beyond what `decay_rate_2.5`/`tab`/`author_id`/
`video_id` already capture about `long_view` likelihood, at least at this
model's capacity (`num_leaves=2`, so only 48 total split decisions to
allocate across 11 candidate features).

The original REJECT's specific mechanism (shared-absolute-score conflict
between a rank-invariant loss and base-rate-calibrated pointwise losses)
does **not** reproduce here — there is no shared loss or shared score at
all in this stacking design, so that diagnosis correctly does not apply
under the GBM-native representation. But the direction's outcome is the
same: no usable signal, just via more mundane feature-redundancy rather
than gradient conflict. Both diagnoses can be true — the auxiliary
engagement signals may simply not carry independent predictive value for
`long_view` in this dataset, regardless of architecture.

## Verdict: REJECT

+0.00024 valid, single seed, below the promotion-look threshold; feature
importance shows the main model makes almost no use of the 4 new columns.
This is a genuinely useful result for the direction as a whole: it rules
out "shared embedding gradient conflict" as the *only* reason multi-task
signals haven't helped in this project — under an architecture with no
shared embedding table at all, the auxiliary engagement signals still add
essentially nothing, pointing instead to the signals themselves (or their
current featurization) being low-information for this task rather than
an architecture-specific artifact. iter63's GBM standalone (valid=0.67168)
remains best; no further work on this specific approach (auxiliary
features from is_like/is_follow/is_comment/is_forward) is recommended.
Approach 2 (true joint multi-task training via a custom LightGBM
objective) was not attempted, per the instructions' own guidance to only
pursue it if approach 1 showed a promising signal — it did not.

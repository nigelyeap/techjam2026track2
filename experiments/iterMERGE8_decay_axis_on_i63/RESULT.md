# iterMERGE8: decay-axis transfer onto iter63's own minimal `rate_only` set

## Hypothesis

`iterMERGE2_decay_axis_transfer` tested the same 3 causal decay-rate feature
axes (author-popularity, duration-bucket, hour-of-day; Laplace-smoothed
half-life-3 rate, sourced from `iter71`/`iter72`/`iter73`) against yixi's
**rich** `LGB_CANDIDATE_COLUMNS`/`XGB_COLUMNS` reference sets (30+ columns
each) and found a clean 4/4 null (with the earlier `iter69` tag-rate axis):
neither model ever split on any of the new columns. That test was never run
against iter63's own **minimal** `rate_only` feature set -- only 6 columns
(`decay_rate_2.5`, `decay_act_2.5`, `decay_tab_rate_3`, `last1`,
`lastk_rate`, `gap`, plus categoricals), `num_leaves=2` linear-tree. A tree
choosing splits among only 6-7 numeric competitors faces far less
competition than one choosing among 30+; a weak feature that never wins a
split among many strong candidates might still win among few. This
experiment tests whether that changes the outcome, reusing
`iterMERGE2_decay_axis_transfer/common.py`'s `attach_axis_rate` (feature
construction + alignment checks) and `iter63_decay_tab_rate/train.py`'s own
`prepare()`/`run()` directly, plus `iterMERGE5_four_model_blend/run.py`'s
verified `fit_lgb`/`fit_xgb`/`within_user_percentile`/`blend4` machinery for
any blend-level retest.

## Status

Complete. Total elapsed 478.7s.

## Harness-fidelity check (must pass before trusting anything else)

Four checks, reproduced via imported modules only (no re-derivation of
feature construction or row alignment) -- all pass at exact (`<=1e-6`)
tolerance, deltas all exactly `0.0`:

| Check | Reference | Reproduced | Delta | Pass |
|---|---|---|---|---|
| iter63 `rate_only` standalone, seed 0 (valid) | 0.6716787219047546 | 0.6716787219047546 | 0.0 | YES |
| iter63 `rate_only` 5-seed sigmoid-mean ensemble, valid | 0.6714167594909668 | 0.6714167594909668 | 0.0 | YES |
| iter63 `rate_only` 5-seed sigmoid-mean ensemble, test | 0.653362512588501 | 0.653362512588501 | 0.0 | YES |
| 3-model reference blend (FM 10%/LGB 52%/XGB 38%), valid | 0.69943440 | 0.6994343996047974 | 0.0 | YES |
| 3-model reference blend, test | 0.68432260 | 0.6843225955963135 | 0.0 | YES |
| 4-model raw blend at iterMERGE5 best weights (fm=0.072/lgb=0.5184/xgb=0.3296/i63=0.08), valid | 0.6998183727264404 | 0.6998183727264404 | 0.0 | YES |
| 4-model raw blend, test | 0.6836708784103394 | 0.6836708784103394 | 0.0 | YES |

Component sub-checks (part of the 3-model reproduction, also exact):
LGB candidate valid 0.6883414387702942, XGB valid 0.6675541996955872, FM
5-seed ensemble valid 0.6398779153823853 -- all match iterMERGE1/iterMERGE5's
established references exactly.

Row-alignment check (yixi's `iterYIXI10` frames vs. iter63's own
`data_ext.py` splits, needed to combine the 4th component position-wise):
0 `user_id` mismatches, 0 `video_id` mismatches (excluding known yixi
NaN-video rows), 0 label mismatches on both valid (124,909 rows) and test
(170,588 rows) -- reproducing iterMERGE5's already-established finding,
using the same trusted (uncast) identity arrays rather than iter63's
train-vocabulary-restricted categorical `user_id` column (see
`iterMERGE5_four_model_blend/RESULT.md` for why the naive categorical
comparison is a false-alarm trap).

All 7 harness numbers reproduced at exact `0.0` delta -- no
nondeterminism tolerance was actually needed in this run (unlike
iterMERGE5's `confirm.py`, which saw ~1e-4-scale cross-run float noise
between two separate script invocations).

## Method

1. Load iter63's own `rate_only` dfs/labels/user-ids via
   `iter63_decay_tab_rate/train.py`'s `prepare()` (through `run()`'s own
   `_cache=None` path at seed 0).
2. For each axis (author, durbucket, hour): take a shallow dict-copy of
   iter63's dfs (so the original 6-column dfs are untouched), attach the
   axis's Laplace-smoothed decay-rate column via
   `iterMERGE2_decay_axis_transfer/common.py`'s `attach_axis_rate` (which
   independently re-verifies row/user/label alignment against iter63's own
   arrays before attaching -- not assumed), then retrain iter63's exact
   `run()` model config (`num_leaves=2`, `linear_tree=True`,
   `learning_rate=0.10`, `n_estimators=500`, `min_child_samples=200`,
   `reg_lambda=1.0`) at seed 0 via `_cache` override (no reimplementation of
   the fitting call).
3. Check `model.booster_.feature_importance()` (both `gain` and `split`
   count) for the new column, and the standalone valid delta vs. iter63's
   unmodified seed-0 baseline (0.6716787219047546).
4. Gate: if any axis shows nonzero split count or delta beyond exact-zero
   noise (`>1e-9`), retrain that axis at 5 seeds (sigmoid-mean ensemble,
   matching the project's iter63/FM convention) and re-run the 4-model
   blend weight search (coarse sweep over the i63 weight + local fine grid
   refinement around the best point, identical procedure to
   `iterMERGE5_four_model_blend/run.py`) with the enhanced i63 replacing
   the original i63 as the 4th component.
5. If all 3 axes show zero split usage / exact-zero delta, stop at the
   standalone gate -- clean REJECT, no blend-level retest (matches
   iterMERGE2's own scoping and this round's time-budget instruction).

## Results

Standalone ablations, single seed = 0, iter63's own `rate_only` feature set
(`decay_rate_2.5`, `decay_act_2.5`, `decay_tab_rate_3`, `lastk_rate`, `gap`,
`duration_ms` numeric + `user_id`/`video_id`/`author_id`/`tab`/`last1`
categorical -- only 6-7 numeric split competitors) plus one new axis column:

| Axis | Column | Valid | Delta vs. seed-0 baseline | New-col split count | New-col gain fraction | Clears standalone gate? |
|---|---|---|---|---|---|---|
| author (popularity) | `decay_author_rate_3` | 0.6716787219047546 | +0.0000000000 | 0 | 0.000000 | No |
| duration-bucket | `decay_durbucket_rate_3` | 0.6716787219047546 | +0.0000000000 | 0 | 0.000000 | No |
| hour-of-day | `decay_hour_rate_3` | 0.6716787219047546 | +0.0000000000 | 0 | 0.000000 | No |

All three deltas are exact-zero at full float64 precision (not rounded --
`valid_primary` for all three variants is byte-identical to the unmodified
seed-0 baseline, `0.6716787219047546`; test primary likewise identical at
`0.6535274386405945`). `model.booster_.feature_importance()` confirms **zero
split usage and zero gain** for every one of the three new columns in every
one of the three fitted models -- the total gain across the model
(59552.996...) is identical with or without the new column present, meaning
LightGBM's `num_leaves=2` linear tree never once chose to split on any of
these features, even with only 6-7 numeric competitors (far less split
competition than yixi's 30+-column reference sets in `iterMERGE2`).

This is not merely "no measurable effect" -- it is the same exact-zero,
zero-split-usage signature `iterMERGE2` found against yixi's much richer
harness, now reproduced a 3rd time against the smallest, lowest-competition
harness in the project. The reduced-competition hypothesis (fewer competing
features -> more likely a weak feature wins a split) does not hold here:
these three engineered rate features carry literally no exploitable signal
beyond what `author_id`/`tab`/`duration_ms`/`decay_tab_rate_3` already
encode, regardless of how much competition they face for splits.

Per protocol (step 4/5 of the task), since none of the 3 axes cleared the
standalone signal gate (nonzero split usage or delta beyond `1e-9`), no
5-seed retrain or blend-level 4-model weight-search retest was performed --
attempting one would be noise on features both trees refuse to split on
with 100% consistency across three independent harnesses now.

## Verdict

| Axis | Verdict |
|---|---|
| author-popularity decay rate (on i63's minimal set) | REJECT |
| duration-bucket decay rate (on i63's minimal set) | REJECT |
| hour-of-day decay rate (on i63's minimal set) | REJECT |

**Overall: REJECT (clean null, all 3 axes, matching iterMERGE2's outcome
exactly).** The "less split competition might let a weak feature win"
hypothesis motivating this round is falsified for this specific feature
family: iter63's own minimal 6-column `rate_only` LightGBM (`num_leaves=2`,
linear-tree) shows the identical zero-split, exact-zero-delta signature
that yixi's much richer 30+-column reference sets showed in `iterMERGE2`.
Combined with `iterMERGE2`'s prior 4/4 nulls (tag from `iter69`, author,
duration-bucket, hour) against yixi's harness, and the original `iter71`/
`iter72`/`iter73` nulls against the old iter63 harness, this closes the
decayed-rate-axis-generalization family at effectively **3/3 harnesses
tested for the author/durbucket/hour trio** (old minimal iter63 harness,
yixi's rich harness, and now the current-production minimal iter63 harness
used as the 4th blend member) -- every harness tried, regardless of feature
count or model capacity, finds these three axes contribute nothing beyond
what the existing categoricals and `decay_tab_rate_3`/`decay_rate_*`
features already encode. No further work on this feature family is
recommended without a genuinely different construction (different
half-life, different smoothing scheme, or a grouping key not already
implicit in the categoricals) -- the "try it against a smaller/different
model" angle specifically is now exhausted for this feature family.

Best-known promotable number is unaffected and remains valid 0.69943440 /
test 0.68432260 (yixi10 blend). Best PRELIMINARY (not promoted) number also
unaffected, remains iterMERGE5's 4-model blend (valid 0.69981837 / test
0.68367088, delta +0.00038397).

## Artifacts

- `run.py` -- single end-to-end script: harness-fidelity reproduction (7
  checks: iter63 seed-0 standalone, iter63 5-seed ensemble valid+test,
  3-model reference valid+test, 4-model raw blend at iterMERGE5's best
  weights valid+test, plus row-alignment verification), standalone per-axis
  ablation (seed 0, all 3 axes), and a conditional 5-seed retrain + 4-model
  blend re-search path for any axis clearing the standalone signal gate
  (not triggered this run -- all 3 axes were clean nulls). Writes
  `results.json` incrementally after each stage. Ran in 478.7s.
- `results.json` -- full numeric record (harness fidelity, row alignment,
  standalone ablations with per-axis `feature_importance()` gain/split
  breakdown, verdict).
- `run.log` -- full stdout of the run.
- No files outside this directory were modified;
  `iterMERGE2_decay_axis_transfer/common.py` (specifically its
  `attach_axis_rate` helper), `iterMERGE5_four_model_blend/run.py`
  (`fit_lgb`/`fit_xgb`/`within_user_percentile`/`blend4`/`PRODUCTION_WEIGHTS`/
  `LGB_CANDIDATE_COLUMNS`/`XGB_COLUMNS`), and `iter63_decay_tab_rate/train.py`
  (`prepare`/`run`/`_sort_by_user`/`_de`) were imported directly as modules
  (read-only), not copied or edited.

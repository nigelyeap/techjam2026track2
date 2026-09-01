# iterMERGE5: iter63's own rate_only GBM as a 4th ensemble member

## Hypothesis

Two rejected rounds (`iterMERGE3_seed_ensemble_gbm`, `iterMERGE4_multitask_aux_3model`)
tried transferring individual columns/techniques between existing
components. This round tries a structurally different direction: add
`iter63_decay_tab_rate`'s own `rate_only` LightGBM -- a genuinely
different, smaller model (`num_leaves=2` linear-tree on
`decay_rate_2.5`/`decay_act_2.5`/`decay_tab_rate_3`/`last1`/`lastk_rate`/`gap`,
vs. yixi's richer `LGB_CANDIDATE_COLUMNS`) -- as a **4th ensemble member**
alongside the current-best 3-model FM/LGB/XGB blend (valid primary
0.69943440 / test 0.68432260,
`experiments/iterYIXI10_video_metadata/RESULT.md`,
`experiments/iterMERGE1_verify_yixi10/RESULT.md`), rather than transferring
a feature or technique between the existing components.

## Harness-fidelity check (must pass before trusting anything else)

**Part A -- 3-model reference**, reproduced from scratch, seed 0, same code
path as `iterMERGE1_verify_yixi10/verify.py`:

| Check | Reference | Reproduced | Delta |
|---|---|---|---|
| LightGBM valid (`LGB_CANDIDATE_COLUMNS`) | 0.68834144 | 0.68834144 | 0.0 |
| XGBoost valid (`XGB_COLUMNS`) | 0.66755420 | 0.66755420 | 0.0 |
| FM valid (5-seed ensemble) | 0.63987792 | 0.63987792 | 0.0 |
| 3-way blend valid (10/52/38, within-user-percentile) | 0.69943440 | 0.69943440 | 0.0 |
| 3-way blend test | 0.68432260 | 0.68432260 | 0.0 |

All five checks passed at exact (`<=1e-6`) tolerance (`results.json`'s
`harness_fidelity_3model.all_pass_1e-6: true`, every delta exactly `0.0`).

**Part B -- iter63's own `rate_only` GBM standalone**, via
`iter63_decay_tab_rate/train.py`'s own `run()`/`prepare()` (not
reimplemented). The dispatch note's stated reference
(`0.6768913269042969`) turned out on inspection to be a *different*
constant: `iterYIXI9_watch_depth_history/common.py`'s
`LGB_REFERENCE_VALID`, which belongs to a richer intermediate model in
yixi's own chain (`decay_rate_5`/`decay_act_5`/`hist_watch_decay_mean_5`),
not iter63's `rate_only` model at all. The correct, code-verified
reference for iter63's own model is `0.6716787219047546` -- confirmed by
directly running `iter63_decay_tab_rate/train.py` (`valid=0.67168` in its
own `run.log`) and independently corroborated by
`iterYIXI6_cross_model_feature_transfer/common.py`'s
`LGB_REFERENCE_VALID = 0.6716787219047546`, which that file's `harness.py`
explicitly labels "iter63 LightGBM". Reproduced exactly against the
corrected reference (`delta_vs_code_verified_reference: 0.0`,
`pass_1e-6: true`).

## Row-alignment investigation (a genuine finding, resolved)

An initial version of this script compared `frames[name]["user_id"]`
(yixi's feature frame) against `i63_dfs[name]["user_id"]`
(iter63's own DataFrame) row-for-row and found large mismatches: 1,990/124,909
on valid, 6,171/170,588 on test -- while `video_id` matched exactly once
yixi's ~17/14 known NaN-video rows were excluded. This looked like a
genuine cross-pipeline row-order misalignment and the script correctly
aborted rather than silently combining misaligned scores.

Root-cause investigation found the actual cause: `i63_dfs[name]["user_id"]`
is a **categorical column whose categories are restricted to values seen
in the train split** (`iter63_decay_tab_rate/train.py:88`,
`pd.CategoricalDtype(categories=dfs['train'][c].unique())`, applied via
`.astype(cats[c])` at line 91). Any `user_id` appearing in valid/test but
not in train becomes `NaN` under that cast -- a legitimate cold-start
feature-encoding choice in iter63's own modeling pipeline (LightGBM treats
it as a missing category), **not a row-order bug**. Comparing that lossy
categorical column against a string made the row look "misaligned" even
though its position was correct.

The fix: compare the **trusted, uncast identity arrays** instead --
`users[name]` (yixi's own array, checked internally by her
`build_frames()`'s own alignment assertion) against `i63_u[name]`
(returned separately by iter63's `prepare()`, sourced from `splits[name]`
before any categorical restriction -- `train.py:93`,
`u = {name: [x[_de.IDX['user_id']] for x in splits[name]] ...}`), plus a
freshly-derived raw `video_id` array pulled directly from iter63's own
`data_ext.load_ext()` splits (same source, no categorical cast). Verified
exhaustively for **every row of valid and test** (not spot-sampled):

| Split | rows | user_id mismatches (trusted arrays) | video_id mismatches (excl. known NaN) | label mismatches |
|---|---|---|---|---|
| valid | 124,909 | **0** | **0** | **0** |
| test | 170,588 | **0** | **0** | **0** |

(`results.json`'s `row_alignment_check` block.) Position-based combination
is confirmed safe -- yixi's `iterYIXI10` feature-frame pipeline and
iter63's own `data_ext.py` pipeline both scan the same two raw CSV files
sequentially with no reordering, so their row order agrees exactly. No
`orig_idx`-based key join was needed once the comparison used the correct
(uncast) arrays; the earlier alarm was a false positive caused by
comparing against a lossy model-input feature column rather than the raw
identity, not a real alignment defect.

## iter63 rate_only GBM: 5-seed ensemble

Sigmoid-mean ensemble across seeds 0-4, matching the FM ensembling
convention used elsewhere in this project:

| Seed | valid primary | test primary |
|---|---|---|
| 0 | 0.6716787 | 0.6535274 |
| 1 | 0.6710515 | 0.6532335 |
| 2 | 0.6710430 | 0.6532010 |
| 3 | 0.6710430 | 0.6532010 |
| 4 | 0.6710515 | 0.6532335 |
| **5-seed sigmoid-mean ensemble** | **0.6714168** | **0.6533625** |

Consistent with `iter63_decay_tab_rate/RESULT.md`'s own prior finding that
seed variance for this model is small (std ~0.0002); ensembling doesn't
change the standalone level meaningfully.

## 4-way blend weight search (valid only)

Components within-user-percentile normalized identically (`groupby(user).rank(pct=True)`),
then combined by weighted sum. Selection is valid-only; test reported for
the record.

**Stage A -- coarse sweep** over the iter63 weight, rescaling FM/LGB/XGB
proportionally down from the production 10/52/38 point:

| w_i63 | fm | lgb | xgb | valid primary |
|---|---|---|---|---|
| 0.00 (reference) | 0.100 | 0.520 | 0.380 | 0.69943440 |
| 0.02 | 0.098 | 0.510 | 0.372 | 0.69946498 |
| 0.05 | 0.095 | 0.494 | 0.361 | 0.69931996 |
| **0.08** | **0.092** | **0.478** | **0.350** | **0.69952005** |
| 0.10 | 0.090 | 0.468 | 0.342 | 0.69943666 |
| 0.12 | 0.088 | 0.458 | 0.334 | 0.69939601 |
| 0.15 | 0.085 | 0.442 | 0.323 | 0.69892651 |
| 0.20 | 0.080 | 0.416 | 0.304 | 0.69837856 |
| 0.25 | 0.075 | 0.390 | 0.285 | 0.69748712 |
| 0.30 | 0.070 | 0.364 | 0.266 | 0.69555748 |

Best coarse point: `w_i63=0.08` (proportional rescale), valid 0.69952005.

**Stage B -- local grid refinement** around the coarse best (125 combos
evaluated, +/-0.02/+/-0.04 steps on i63/fm/lgb, xgb=remainder). Top result:

| fm | lgb | xgb | i63 | valid primary | test primary |
|---|---|---|---|---|---|
| **0.072** | **0.5184** | **0.3296** | **0.08** | **0.69981754** | 0.68366760 |
| 0.052 | 0.5184 | 0.3496 | 0.08 | 0.69980049 | 0.68345177 |
| 0.072 | 0.5184 | 0.3496 | 0.06 | 0.69979024 | 0.68392932 |
| 0.092 | 0.5184 | 0.3296 | 0.06 | 0.69973350 | 0.68398631 |
| 0.072 | 0.4984 | 0.3496 | 0.08 | 0.69970179 | 0.68367648 |

Full top-30 grid in `results.json`'s `weight_search_fine`.

## Confirmation pass (`confirm.py`)

The weight search found the best point via a 135-combo grid on valid only
-- a real risk of a narrow grid-cell spike rather than a genuine, stable
gain. Two checks, run against a fresh, independently-refit FM/LGB/XGB
(sanity-checked back to the exact 0.69943440 reference) plus fresh raw
per-seed iter63 scores:

**1. Per-seed robustness** -- blend at the best weight point
(`fm=0.072, lgb=0.5184, xgb=0.3296, i63=0.08`) using iter63's rate_only
score from each of its 5 *individual* seeds (not the ensemble), against
the same fixed FM/LGB/XGB components:

| iter63 seed | valid primary | delta vs. reference | test primary |
|---|---|---|---|
| 0 | 0.69988716 | +0.00045276 | 0.68372762 |
| 1 | 0.69975996 | +0.00032556 | 0.68370277 |
| 2 | 0.69977546 | +0.00034106 | 0.68369925 |
| 3 | 0.69977546 | +0.00034106 | 0.68369925 |
| 4 | 0.69975996 | +0.00032556 | 0.68370277 |
| **5-seed ensemble** | **0.69981837** | **+0.00038397** | 0.68367088 |

All 5 individual seeds clear `PRELIMINARY_DELTA=0.0003` on their own
(range +0.00033 to +0.00045) -- the gain is not an artifact of the
sigmoid-mean ensembling step, it holds up seed-by-seed.

**2. Plateau check** -- re-evaluated the top-3 fine-grid points (using the
established 5-seed iter63 ensemble):

| Point | weights | valid delta | test delta |
|---|---|---|---|
| best | fm=0.072, lgb=0.5184, xgb=0.3296, i63=0.08 | +0.00038397 | -0.00065172 |
| 2nd | fm=0.052, lgb=0.5184, xgb=0.3496, i63=0.08 | +0.00036609 | -0.00087190 |
| 3rd | fm=0.072, lgb=0.5184, xgb=0.3496, i63=0.06 | +0.00035810 | -0.00039768 |

All three neighboring grid points independently clear the preliminary bar
on valid (`is_plateau_across_top3_grid_points: true`) and all three show
the same small, consistent test regression (-0.0004 to -0.0009). This is
a genuine, if small, **plateau** -- not a single overfit spike in the
search grid -- but the pattern (valid up ~0.0004, test down ~0.0004 to
0.0009) is directionally opposite between the two splits at every point
checked, which is exactly the signature of "real, small, valid-only
effect" rather than "found and confirmed on both splits."

## Verdict

**PRELIMINARY.** Best 4-model point: `fm=0.072, lgb=0.5184, xgb=0.3296,
i63=0.08`, valid primary **0.69981837** vs. 3-model reference
**0.69943440** -- delta **+0.00038397** (confirmed by an independent
refit in `confirm.py`, consistent with `run.py`'s original +0.00038314).
This clears `PRELIMINARY_DELTA=0.0003` but falls well short of
`PROMOTION_DELTA=0.001` (~38% of the way to the promotion bar), and the
gain is now confirmed robust: it holds across all 5 individual iter63
seeds (not just the ensemble) and across the top-3 neighboring grid
points (a plateau, not a search-grid spike). What keeps this at
PRELIMINARY rather than PROMOTE is that test primary is consistently
**below** the 3-model reference at every point checked (-0.0004 to
-0.0009, reported for the record only, never used for selection) -- the
valid-side gain, while real and stable under both robustness checks
performed, does not visibly transfer to test, so it is best read as a
small, genuine but low-confidence improvement rather than a promotion-grade
one.

Given the improvement clears preliminary but the valid/test directional
mismatch persists across every seed and grid point tested, and given the
declining prior after 4 consecutive REJECTs on this project going into
this round, this is left as PRELIMINARY rather than escalated to PROMOTE.
No submission-affecting files were touched; the current production blend
(`experiments/iterYIXI10_video_metadata/RESULT.md`, valid 0.69943440 /
test 0.68432260) remains the best confirmed result.

## Artifacts

- `run.py` -- single end-to-end script: 3-model harness-fidelity
  reproduction, iter63 standalone harness-fidelity reproduction (with the
  dispatch-note reference-constant discrepancy documented in code), row
  identity verification (trusted-array method, see above), 5-seed iter63
  `rate_only` training with sigmoid-mean ensembling, and the coarse+fine
  4-way weight search. Writes `results.json` incrementally after each
  stage (`harness_fidelity_3model` -> `harness_fidelity_iter63_standalone`
  -> `row_alignment_check` -> `iter63_5seed_ensemble` ->
  `weight_search_coarse` -> `weight_search_fine`/`summary`).
- `results.json` -- full numeric record of every stage above, including
  the complete coarse grid and top-30 fine grid.
- `run.log` -- full stdout of the run (elapsed 258.7s).
- `confirm.py` -- confirmation pass: independent refit of FM/LGB/XGB
  (sanity-checked back to the exact 3-model reference) and iter63's 5
  seeds with raw per-seed scores retained, then (1) per-seed blend at the
  best weight point and (2) re-evaluation of the top-3 fine-grid points.
  Appends `seed_confirmation`, `plateau_check`, and `final_summary` to
  the same `results.json`.
- `confirm.log` -- full stdout of the confirmation run (elapsed 206.8s).
- `results.json` -- full numeric record of every stage from both scripts.
- No files outside this directory were modified; `iter63_decay_tab_rate/train.py`
  and `make_submission.py` were imported directly as modules (read-only),
  not copied or edited.

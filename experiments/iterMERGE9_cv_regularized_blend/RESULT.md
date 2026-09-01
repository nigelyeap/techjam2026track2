# iterMERGE9: CV-regularized weight selection for the 4-model blend

## Hypothesis

`iterMERGE5_four_model_blend`'s fine-grid single-split-valid optimum
(`fm=0.072, lgb=0.5184, xgb=0.3296, i63=0.08`) clears `PRELIMINARY_DELTA`
on valid (+0.00038397 vs. the 3-model reference) but REGRESSES on test
(-0.00065172) -- a valid/test crossover consistent with a 135-point fine
grid search overfitting its weight choice to the single 124,909-row valid
split. `iterMERGE7_stacked_meta_blend`'s k-fold user-level out-of-fold (OOF)
stacking tested a genuinely different, nonlinear combiner of the same 4
components and also found nothing promotable, but never directly asked
whether a *coarser, CV-regularized* choice within the same *linear* weight
family closes the valid/test gap. This experiment tests exactly that: same
coarse-then-fine grid search structure as `iterMERGE5`, over the same 4
frozen within-user-percentile component scores (FM 5-seed ensemble, LGB
seed 0, XGB seed 0, i63 5-seed ensemble), but each grid point is scored by
5-fold user-level cross-validated MEAN valid score (fold-splitting adapted
from `iterMERGE7`'s exact `KFold`-on-unique-users technique) instead of
single-split valid. No underlying model is retrained per fold -- the 4
components are already frozen scores, so per-fold scoring is pure held-out
evaluation of the same globally-computed within-user-percentile scores
restricted to that fold's rows (no leakage: a user's own percentile rank
depends only on that user's own rows, and fold membership is at the user
level).

Question: does CV-selecting the weight point find one that (a) still clears
the preliminary/promotion bar on single-split valid, comparably to
`iterMERGE5`, AND (b) does not regress on test vs. the 3-model reference
(0.68432260) -- i.e., does regularizing the selection actually close the
crossover, or does CV selection just land on the same point (or a worse
one)?

## Status

Complete. Total elapsed 506.2s.

## Harness-fidelity check (must pass before trusting anything else)

Seven checks, reproduced via imported modules only (no re-derivation of
feature construction or row alignment) — all pass at exact `0.0` delta:

| Check | Reference | Reproduced | Delta | Pass |
|---|---|---|---|---|
| LightGBM candidate valid | 0.68834144 | 0.68834144 | 0.0 | YES |
| XGBoost valid | 0.66755420 | 0.66755420 | 0.0 | YES |
| FM valid (5-seed ensemble) | 0.63987792 | 0.63987792 | 0.0 | YES |
| 3-model reference blend, valid | 0.69943440 | 0.69943440 | 0.0 | YES |
| 3-model reference blend, test | 0.68432260 | 0.68432260 | 0.0 | YES |
| iter63 `rate_only` standalone, seed 0 | 0.6716787219047546 | 0.6716787219 | 0.0 | YES |
| iter63 `rate_only` 5-seed ensemble, valid | 0.67141676 | 0.67141676 | 0.0 | YES |
| iter63 `rate_only` 5-seed ensemble, test | 0.65336251 | 0.65336251 | 0.0 | YES |
| 4-model raw blend at iterMERGE5's best weights, valid | 0.69981837 | 0.69981837 | 0.0 | YES |
| 4-model raw blend at iterMERGE5's best weights, test | 0.68367088 | 0.68367088 | 0.0 | YES |

Row alignment (yixi's frames vs. iter63's own splits, trusted uncast
identity arrays): 0 `user_id`/`video_id`/label mismatches on both valid
(124,909 rows) and test (170,588 rows). All checks passed at exact
tolerance before the CV search began.

## Method

1. Reuse `iterMERGE5_four_model_blend/run.py`'s verified functions
   (`fit_lgb`, `fit_xgb`, `sigmoid`, `within_user_percentile`,
   `stable_user_order`, `blend4`, `LGB_CANDIDATE_COLUMNS`, `XGB_COLUMNS`,
   `PRODUCTION_WEIGHTS`, `ITER63_RATE_ONLY_REFERENCE_VALID`) as an imported
   module -- no re-derivation of feature construction or row alignment.
2. Reproduce the 3-model reference blend, iter63 `rate_only` standalone
   seed-0 valid, iter63 5-seed sigmoid-mean ensemble (valid+test), and the
   4-model raw blend at `iterMERGE5`'s exact best weights (valid+test) --
   all checked at `<=1e-6` tolerance before trusting anything downstream.
3. Build 5-fold, user-level CV fold masks over valid's unique users
   (`sklearn.model_selection.KFold`, `shuffle=True`, `random_state=0` --
   matching `iterMERGE7`'s exact convention), partitioning all of valid's
   rows with no overlap and no user split across folds.
4. Coarse sweep over the i63 weight (rescaling FM/LGB/XGB proportionally
   from the production 10/52/38 point, identical grid to `iterMERGE5`),
   then local fine-grid refinement around the best coarse point (same
   `+-0.04` step-`0.02` grid as `iterMERGE5`) -- but selecting at each stage
   by highest 5-fold CV MEAN valid primary, not single-split valid.
5. Report the CV-selected point's CV mean valid, single-split valid (for
   direct comparability to `iterMERGE5`/the 3-model reference), and test
   (record only, never for selection). Also report the CV mean valid AT
   `iterMERGE5`'s original best point, and the point that single-split
   selection would have chosen from this identical fine grid, for a direct
   side-by-side.
6. Decision gate: `PRELIMINARY_DELTA=0.0003`, `PROMOTION_DELTA=0.001`
   (vs. the 3-model reference's 0.69943440 valid), applied to the
   CV-selected point's single-split valid, consistent with every prior
   round in `MERGE_LEDGER.md`. Additionally check whether the CV-selected
   point's test score regresses vs. the 3-model reference test
   (0.68432260) -- the key question this round asks.

## Results

**5-fold user-level CV fold construction**: 22,377 unique valid users split
into 5 folds of ~4,475-4,476 users each, covering all 124,909 valid rows
with no user split across folds (fold row sizes: 25085/25094/24840/24982/
24908).

**CV mean valid AT iterMERGE5's original best point**
(`fm=0.072, lgb=0.5184, xgb=0.3296, i63=0.08`):

| | Value |
|---|---|
| Single-split valid | 0.69981837 |
| CV mean valid | 0.69979823 |
| Per-fold primaries | 0.702313 / 0.699951 / 0.696766 / 0.695340 / 0.704623 |
| Test (record only) | 0.68367088 |

The CV mean (0.69979823) is within 0.00002 of the single-split valid
(0.69981837) — iterMERGE5's original point was already CV-robust, not an
artifact of single-split overfitting, even before any CV-based reselection.

**Coarse sweep over the i63 weight, scored by CV mean valid**: 10 points
swept (`w_i63` from 0.00 to 0.30); best coarse point by CV mean was
`w_i63=0.08` (`fm=0.092, lgb=0.4784, xgb=0.3496, i63=0.08`,
`cv_mean_valid=0.69949651`) — the same `w_i63` iterMERGE5's own coarse
sweep had already identified as best.

**Fine grid refinement** (125 combos around the coarse optimum, scored by
CV mean valid): best point —

| | Value |
|---|---|
| Weights | `fm=0.072, lgb=0.5184, xgb=0.3296, i63=0.08` |
| CV mean valid | 0.69979733 |
| Single-split valid | 0.69981754 |
| Test (record only) | 0.68366760 |

This is, to 4 decimal places, **the identical weight point** iterMERGE5's
single-split fine grid search already found — CV-based selection did not
move the optimum. The next-best fine-grid points by CV mean (weights
`fm=0.052/xgb=0.3496`, `xgb=0.3496/i63=0.06`, `fm=0.092/i63=0.06`,
`lgb=0.4984`) all score lower on CV mean valid and show test scores
scattered both above (0.68392932, 0.68398631) and below
(0.68345177, 0.68367648) the winning point's test — no CV-adjacent
point systematically resolves the crossover either.

## Verdict

| | Single-split valid | CV mean valid | Test (record only) |
|---|---|---|---|
| 3-model reference | 0.69943440 | — | 0.68432260 |
| iterMERGE5 original point | 0.69981837 | 0.69979823 | 0.68367088 |
| **iterMERGE9 CV-selected point** | **0.69981754** | **0.69979733** | **0.68366760** |

Delta vs. 3-model reference (CV-selected point): single-split valid
**+0.00038314** (clears `PRELIMINARY_DELTA=0.0003`), test **-0.00065500**
(regresses). CV-selected point is the same point as iterMERGE5's original
(to 4 decimal places). Crossover resolved (test ≥ 3-model reference test)?
**No.**

**PRELIMINARY, not actionable — same substantive finding as iterMERGE5,
now with an important negative confirmation added.** This round directly
answers its motivating question: the valid/test crossover iterMERGE5 first
surfaced is **not an artifact of the fine grid search overfitting its
weight choice to the single 124,909-row valid split**. Regularizing
selection via 5-fold user-level cross-validation converges to essentially
the same weight point, whose CV mean valid tracks its single-split valid to
within 0.00002 — there was no overfitting gap to regularize away in the
first place. The valid/test crossover is therefore better explained as a
genuine distributional difference between the valid and test splits
themselves (consistent with `iterMERGE7`'s independent finding that a
completely different, nonlinear combiner produces the same crossover
pattern under honest out-of-fold evaluation) rather than a selection
methodology problem — closing off "try a more robust selection procedure"
as a promising direction for resolving the crossover specifically.

Best-known promotable number is unaffected and remains valid 0.69943440 /
test 0.68432260 (yixi10 3-model blend, currently submitted). Best
PRELIMINARY (not promoted) finding remains iterMERGE5's original 4-model
blend point — now corroborated as CV-robust by this round, but still not
promotable given its test-side regression and the 0.001 promotion bar it
falls well short of.

## Artifacts

- `run.py` -- single end-to-end script: harness-fidelity reproduction (7
  checks), 5-fold user-level CV fold construction, coarse+fine grid weight
  search scored by CV mean valid, comparison against `iterMERGE5`'s
  original point and against single-split selection on the identical grid.
  Writes `results.json` incrementally after each stage.
- `results.json` -- full numeric record.
- `run.log` -- full stdout of the run.
- No files outside this directory were modified;
  `iterMERGE5_four_model_blend/run.py` (`fit_lgb`/`fit_xgb`/`sigmoid`/
  `within_user_percentile`/`stable_user_order`/`blend4`/
  `LGB_CANDIDATE_COLUMNS`/`XGB_COLUMNS`/`PRODUCTION_WEIGHTS`/
  `ITER63_RATE_ONLY_REFERENCE_VALID`) and `iter63_decay_tab_rate/train.py`
  (`prepare`/`run`/`_de`) were imported directly as modules (read-only), not
  copied or edited. `iterMERGE7_stacked_meta_blend/run.py`'s fold-splitting
  technique was adapted (not imported directly, since its function is
  train/holdout-oriented for meta-learner fitting -- here it's simplified
  to pure held-out scoring of frozen components) into this file's
  `make_cv_fold_masks`.

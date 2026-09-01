# iterMERGE6: isotonic calibration applied to the 4-model FM/LGB/XGB/i63 blend

## Hypothesis

`iterMERGE5_four_model_blend` found a real but small valid-only gain from
adding iter63's own `rate_only` GBM as a 4th ensemble member (best point
`fm=0.072, lgb=0.5184, xgb=0.3296, i63=0.08`, valid **0.69981837** vs.
3-model reference **0.69943440**, delta **+0.00038397**) -- but test was
consistently *worse* than the 3-model reference at every point checked
(-0.0004 to -0.0009). This is exactly the valid/test-crossover pattern that
`iter66_calibrated_blend`'s isotonic-calibration technique was designed to
address (correcting for a component's score not being well-calibrated,
rather than a pure weight-search overfit). This experiment asks: does
fitting per-component isotonic maps and calibrating before blending, on
this specific 4-model combination, produce a blend that is **both** a
genuine valid improvement over 0.69943440 **and** doesn't regress test
below 0.68432260 -- the two-split-consistent gain that iterMERGE5 lacked?

**Important prior directly on point:** `iter66_calibrated_blend/RESULT.md`
already tested isotonic calibration on iter63's own `rate_only` GBM (in a
2-model alpha=0.14/global-minmax blend) and found a catastrophic REJECT:
isotonic regression pooled 122,613 unique raw valid scores into only **37**
distinct calibrated levels, destroying within-user ranking (GBM standalone
valid 0.67168 -> 0.54189, blend 0.67606 -> 0.63877). This experiment
re-tests the same mechanism against all 4 different, generally richer
components used in the current production blend, to check whether that
collapse is specific to iter63's small `num_leaves=2` GBM or a general
property of isotonic calibration on this task's raw score distributions.

## Harness-fidelity check (must pass before trusting anything else)

Reused `iterMERGE5_four_model_blend/run.py` directly as an imported module
(`fit_lgb`, `fit_xgb`, `sigmoid`, `within_user_percentile`,
`stable_user_order`, `LGB_CANDIDATE_COLUMNS`, `XGB_COLUMNS`) rather than
re-deriving any of it, per the operational instructions for this round.

**Part A -- 3-model reference**, reproduced from scratch, seed 0:

| Check | Reference | Reproduced | Delta |
|---|---|---|---|
| LightGBM valid (`LGB_CANDIDATE_COLUMNS`) | 0.68834144 | 0.68834144 | 0.0 |
| XGBoost valid (`XGB_COLUMNS`) | 0.66755420 | 0.66755420 | 0.0 |
| FM valid (5-seed ensemble) | 0.63987792 | 0.63987792 | 0.0 |
| 3-way blend valid (10/52/38, within-user-percentile) | 0.69943440 | 0.69943440 | 0.0 |
| 3-way blend test | 0.68432260 | 0.68432260 | 0.0 |

All five checks passed at exact (`<=1e-6`) tolerance
(`harness_fidelity_3model.all_pass_1e-6: true`).

**Part B -- iter63's own `rate_only` GBM standalone.** Used the
code-verified reference from `iterMERGE5/RESULT.md`
(**0.6716787219047546**), *not* the dispatch-note trap constant
(0.6768913269042969, which is actually `iterYIXI9`'s
`LGB_REFERENCE_VALID` for a different, richer intermediate model in yixi's
chain -- documented in `iterMERGE5/RESULT.md`). Reproduced exactly:
seed-0 valid = 0.6716787219, delta 0.0, `pass_1e-6: true`.

**Row-alignment check** (reused method: trusted uncast identity arrays, not
iter63's train-only-categories `user_id` column -- the false-alarm trap
`iterMERGE5` documented). Checked on **all three splits** (train included,
since this round needs train-split raw scores for isotonic fitting that
iterMERGE5 didn't need):

| Split | rows | user_id mismatches | video_id mismatches (excl. known NaN) | label mismatches |
|---|---|---|---|---|
| train | 1,141,112 | **0** | **0** | **0** |
| valid | 124,909 | **0** | **0** | **0** |
| test | 170,588 | **0** | **0** | **0** |

Position-based combination confirmed safe on all 3 splits.

**Sanity check**: the raw (uncalibrated) 4-model blend at iterMERGE5's
exact best weight point reproduced iterMERGE5's exact numbers in this
independent run: valid 0.69981837 (delta 0.0), test 0.68367088 (delta 0.0).
This confirms the two experiments' pipelines agree before introducing
calibration as the only new variable.

## Method

Per-component isotonic calibration, following `iter66_calibrated_blend`'s
actual code (`run_isotonic.py`), which fits `IsotonicRegression
(out_of_bounds='clip')` on `(component_raw_train_score, y_train)` and
applies the fitted map to valid/test. Train is fully held out from both
evaluation splits, so this avoids leakage into valid (used for weight
search) or test (reported for the record). This differs slightly from the
dispatch note's "held-out slice of valid" phrasing, but matches iter66's
actual, already-validated precedent exactly rather than inventing a new
calibration-fitting protocol; noted explicitly here for auditability.

"Raw" score per component = whatever score iterMERGE5 fed into
`within_user_percentile()` for that component (LGB/XGB: `model.predict()`;
FM/i63: the existing 5-seed sigmoid-mean ensemble score) -- i.e., the same
processing level at which each component enters the uncalibrated blend, so
calibration is a clean substitution rather than a change in what "the
component's score" means.

Two blending variants, both weight-searched on **valid only**
(coarse-then-fine grid, search centers at both iterMERGE5's found optimum
`fm=0.072/lgb=0.5184/xgb=0.3296/i63=0.08` and the production 10/52/38/0
point, per the dispatch instructions):

- **(a) percentile**: calibrated score -> `within_user_percentile()` (same
  normalization convention as the current production blend) -> weighted
  sum.
- **(b) direct**: calibrated score used directly (isotonic output is
  already probability-scaled) -> weighted sum, no percentile step --
  mirrors `iter66_calibrated_blend`'s own original substitution design.

## Results

### Calibration diagnostics (the key finding)

| Component | unique raw (train / valid) | unique calibrated (valid / test) | standalone valid: raw -> calibrated | delta |
|---|---|---|---|---|
| FM | 788,541 / 117,422 | **576 / 631** | 0.63988 -> 0.63991 | **+0.00003** |
| LightGBM | 1,016,423 / 120,557 | **35 / 42** | 0.68834 -> 0.53970 | **-0.14865** |
| XGBoost | 71,165 / 15,776 | **68 / 71** | 0.66755 -> 0.53774 | **-0.12981** |
| iter63 i63 | 955,658 / 122,612 | **55 / 52** | 0.67142 -> 0.54159 | **-0.12983** |

`iter66`'s collapse mechanism reproduces almost exactly on 3 of the 4
components (iter63's own model collapsed to 37 levels in iter66 vs. 52-55
levels here -- same mechanism, same order of magnitude, small difference
plausibly from the different blend/ensembling context). It is **not**
specific to iter63's small `num_leaves=2` GBM: yixi's richer LightGBM
(`num_leaves=2` linear-tree, same family) and XGBoost (a fundamentally
different library/config) both collapse just as catastrophically. Only FM
(a continuous embedding-dot-product + sigmoid model, not tree-based)
survives calibration essentially unchanged. This is consistent with
isotonic regression's pool-adjacent-violators mechanism: any component
whose raw score has runs of ties or near-ties in local label rate as score
increases gets step-function-collapsed, and tree-model outputs (a small
number of achievable leaf values / leaf-value combinations, even under
linear-tree leaves) are exactly this shape, while FM's dense per-example
embedding interactions are not.

### Blend weight search (valid only)

**Variant (a) -- calibrated + percentile.** All values catastrophic and far
below the reference:

| weights | valid | test |
|---|---|---|
| best coarse (`fm=.055,lgb=.394,xgb=.251,i63=.30`) | 0.60932672 | 0.60603249 |
| best fine (`fm=.115,lgb=.454,xgb=.071,i63=.36`) | 0.63370478 | 0.63300258 |

**Variant (b) -- calibrated, direct (no percentile).** Better than variant
(a) but still far below the reference, and notably the search converges to
**`i63=0.0`** -- the weight search actively drives the worst-collapsed,
least-useful-post-calibration component's weight to zero, effectively
falling back toward a 3-component calibrated blend:

| weights | valid | test |
|---|---|---|
| best (`fm=0.078, lgb=0.563, xgb=0.358, i63=0.0`) | **0.63862032** | 0.63966358 |

### Overall best (across both variants)

| | valid | test |
|---|---|---|
| 3-model reference (production) | 0.69943440 | 0.68432260 |
| iterMERGE5 raw 4-model best (uncalibrated, reproduced here) | 0.69981837 | 0.68367088 |
| **iterMERGE6 best calibrated 4-model (variant b)** | **0.63862032** | **0.63966358** |
| delta vs. 3-model reference | **-0.06081408** | **-0.04465902** |

## Verdict

**REJECT.** Isotonic calibration does not close or improve on iterMERGE5's
valid/test crossover -- it makes both splits dramatically worse. Best
calibrated 4-model point (variant b, direct calibrated-score blend,
`fm=0.078, lgb=0.563, xgb=0.358, i63=0.0`): valid **0.63862032**, a delta
of **-0.06081408** vs. the 3-model reference (**-0.06508** vs. the raw
4-model iterMERGE5 best) -- nowhere near `PRELIMINARY_DELTA=0.0003`, let
alone `PROMOTION_DELTA=0.001`, and in the wrong direction by nearly two
orders of magnitude relative to that bar. Test is equally catastrophic
(-0.04465902 vs. reference), so this is not a valid/test-crossover case
either -- calibration hurts both splits together, consistently.

Root cause is now confirmed to generalize beyond `iter66`'s original
single-model finding: isotonic calibration's pool-adjacent-violators
mechanism collapses **any of this task's tree-based model scores**
(LightGBM, XGBoost, iter63's GBM -- 3 of 4 components) into 35-71 distinct
levels, which is catastrophic for a within-user ranking task where GAUC and
nDCG@5 depend on fine-grained relative ordering within each user's small
impression set -- tied scores get averaged ranks, destroying exactly the
signal the linear blend relies on. Only FM, a continuous (non-tree)
similarity-style score, survives calibration intact (576-631 levels,
±0.00003 standalone delta) -- but FM alone isn't enough to carry a useful
blend, and the weight search confirms this by driving the calibrated i63
weight to exactly 0 (the search prefers to drop the worst-collapsed
component rather than use it, but still can't recover anywhere near the
reference level from the two collapsed tree-model components it must keep).

This is a clean, mechanistic, two-split-consistent REJECT, not a
borderline call -- there is no calibrated-blend candidate anywhere in
either variant's coarse+fine grids (85+ points total) that comes remotely
close to the 3-model reference, let alone the iterMERGE5 4-model point this
experiment was trying to fix. The specific hypothesis (does calibration fix
iterMERGE5's valid/test crossover) is answered: no -- calibration is
strictly worse on both splits, so it does not offer a path to a genuine
two-split-consistent gain for this blend. No submission-affecting files
were touched; current production blend
(`experiments/iterYIXI10_video_metadata/RESULT.md`, valid 0.69943440 / test
0.68432260) remains the best confirmed and only promotable result.

## Artifacts

- `run.py` -- single end-to-end script: imports `iterMERGE5_four_model_
  blend/run.py` as a module and reuses its verified `fit_lgb`/`fit_xgb`/
  `sigmoid`/`within_user_percentile` functions directly; reproduces the
  3-model and iter63-standalone harness-fidelity checks; row-alignment
  check extended to all 3 splits (train included, needed for isotonic
  fitting); trains all 4 components with train-split raw scores retained;
  fits per-component `sklearn.isotonic.IsotonicRegression` on
  `(train_raw, y_train)`; runs both calibrated-blend weight-search variants
  (percentile and direct) with coarse-then-fine grids centered at both the
  iterMERGE5 optimum and the production weights. Writes `results.json`
  incrementally after each of 8 stages.
- `results.json` -- full numeric record: `harness_fidelity_3model`,
  `harness_fidelity_iter63_standalone`, `row_alignment_check`,
  `iter63_5seed_ensemble`, `raw_4model_at_merge5_best_weights` (sanity
  check), `calibration_fit` (per-component diagnostics), both
  `weight_search_variant_*` grids (coarse + top-30 fine), and `verdict`.
- `run.log` -- full stdout of the run (elapsed 772.6s).
- No files outside this directory were modified;
  `iterMERGE5_four_model_blend/run.py`, `iter63_decay_tab_rate/train.py`,
  `iterYIXI10_video_metadata/features.py`, and `make_submission.py` were
  imported directly as modules (read-only), not copied or edited.

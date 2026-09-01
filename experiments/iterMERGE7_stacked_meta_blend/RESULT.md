# iterMERGE7: stacked nonlinear meta-learner over the 4 within-user-percentile component scores

## Hypothesis

Every merge round so far (2 through 6) only tried **linear** combinations of
component scores: fixed-weight weighted sums (raw or grid-searched:
`iterMERGE2`-`iterMERGE5`), or per-component monotonic recalibration before a
linear sum (`iterMERGE6`, catastrophic REJECT). None has tried a genuinely
**nonlinear meta-learner** that can capture interactions between components
(e.g. "when FM and i63 disagree strongly, trust LGB more"). This experiment
tests whether a stacked meta-learner over the 4 within-user-percentile
component scores (FM, LGB, XGB, i63) beats the best linear weight point
found in `iterMERGE5` (`fm=0.072/lgb=0.5184/xgb=0.3296/i63=0.08`, valid
**0.69981837**), using rigorous k-fold out-of-fold (OOF) evaluation to
avoid trivially overfitting the meta-learner to a single 124,909-row valid
split.

## Harness-fidelity check (must pass before trusting anything else)

Reused `iterMERGE5_four_model_blend/run.py` directly as an imported module
(`fit_lgb`, `fit_xgb`, `sigmoid`, `within_user_percentile`,
`stable_user_order`, `blend4`, `LGB_CANDIDATE_COLUMNS`, `XGB_COLUMNS`,
`PRODUCTION_WEIGHTS`, `ITER63_RATE_ONLY_REFERENCE_VALID`, and the row-
alignment method), per the operational instructions for this round.

**Part A — 3-model reference**, reproduced from scratch, seed 0:

| Check | Reference | Reproduced | Delta |
|---|---|---|---|
| LightGBM valid (`LGB_CANDIDATE_COLUMNS`) | 0.68834144 | 0.68834144 | 0.0 |
| XGBoost valid (`XGB_COLUMNS`) | 0.66755420 | 0.66755420 | 0.0 |
| FM valid (5-seed ensemble) | 0.63987792 | 0.63987792 | 0.0 |
| 3-way blend valid (10/52/38, within-user-percentile) | 0.69943440 | 0.69943440 | 0.0 |
| 3-way blend test | 0.68432260 | 0.68432260 | 0.0 |

All five checks passed at exact (`<=1e-6`) tolerance
(`harness_fidelity_3model.all_pass_1e-6: true`, every delta exactly `0.0`).

**Part B — iter63's own `rate_only` GBM standalone.** Used the code-verified
reference from `iterMERGE5/RESULT.md`: **0.6716787219047546**, NOT the
dispatch-note-stated `0.6768913269042969` (that constant is
`iterYIXI9_watch_depth_history/common.py`'s `LGB_REFERENCE_VALID`, a
different, richer intermediate model in yixi's own chain — see
`iterMERGE5/RESULT.md` for the full explanation). Reproduced exactly:
seed-0 valid = 0.6716787219, delta 0.0, `pass_1e-6: true`.

**5-seed iter63 ensemble**: valid=0.67141676, test=0.65336251 — matches
`iterMERGE5`'s own 5-seed ensemble numbers to the reported precision.

**Row-alignment check** (reused method: trusted uncast identity arrays, not
iter63's train-only-categories `user_id` column): 0 user_id mismatches, 0
video_id mismatches (excl. known NaN), 0 label mismatches on both valid
(124,909 rows) and test (170,588 rows). Position-based combination
confirmed safe.

**Sanity check — raw 4-model blend at iterMERGE5's exact best weights**
(`fm=0.072, lgb=0.5184, xgb=0.3296, i63=0.08`), reproduced independently in
this script:

| | Claimed (iterMERGE5) | Reproduced (this run) | Delta |
|---|---|---|---|
| valid | 0.69981837 | 0.69981837 | 0.0 |
| test | 0.68367088 | 0.68367088 | 0.0 |

Exact match. All harness-fidelity checks pass before any new method is
introduced.

## Method

**K-fold (5-fold) OOF stacking**, folded at the **user level** (not row
level) — no user's rows are split across the meta-learner's train/holdout
folds within a fold iteration, consistent with the within-user-ranking
nature of the primary metric. For each of 5 folds: fit the meta-learner on
the other 4 folds' `(4 percentile-normalized component scores, label)`
pairs, predict on the held-out fold. This produces OOF meta-predictions
covering all 124,909 valid rows — evaluated as the valid-only selection
number (never the optimistic in-sample-fit valid).

Two meta-learner families tried, both over the same 4 features
`[fm, lgb, xgb, i63]` (each already within-user-percentile normalized,
matching what the linear blend consumes):

- **(a) Logistic regression**, `sklearn.linear_model.LogisticRegression`,
  swept over inverse-regularization strength `C ∈ {0.001, 0.003, 0.01,
  0.03, 0.1, 0.3, 1.0, 3.0}` (small `C` = heavier regularization).
- **(b) Shallow GBM**, `LGBMClassifier` with `num_leaves ∈ {2, 3, 4}`
  (mirroring the production LGB/i63 configs' shallow-tree convention),
  `learning_rate=0.05`, `n_estimators=200`, `min_child_samples=200`,
  `reg_lambda=1.0`, `subsample=0.8`.

For the final reported model: refit the winning meta-learner on **all** of
valid, apply to test, report test for the record only (never for
selection).

## Results

### OOF valid sweep

**Logistic regression** (heavier regularization = lower `C`):

| C | OOF valid primary |
|---|---|
| 0.001 | 0.69561803 |
| 0.003 | 0.69753742 |
| 0.01 | 0.69812423 |
| 0.03 | 0.69837332 |
| 0.1 | 0.69845629 |
| **0.3** | **0.69856703** (best) |
| 1.0 | 0.69851267 |
| 3.0 | 0.69848919 |

Best logistic regression OOF valid: **0.69856703** at `C=0.3` — below the
3-model reference (0.69943440) by -0.00086737.

**Shallow GBM**:

| num_leaves | OOF valid primary |
|---|---|
| **2** | **0.69904566** (best) |
| 3 | 0.69789273 |
| 4 | 0.69783026 |

Best GBM OOF valid: **0.69904566** at `num_leaves=2` — better than the best
logistic regression, but still below the 3-model reference (0.69943440) by
-0.00038874, and below iterMERGE5's linear optimum (0.69981837) by
-0.00077271.

**Best overall meta-learner: shallow GBM, `num_leaves=2`, OOF valid
0.69904566.**

### Refit on all valid, apply to test

Refitting the winning meta-learner (GBM, `num_leaves=2`) on all of valid
and scoring test:

| | valid | test |
|---|---|---|
| 3-model reference (production) | 0.69943440 | 0.68432260 |
| iterMERGE5 linear optimum (4-model, raw) | 0.69981837 | 0.68367088 |
| **iterMERGE7 stacked meta-blend (OOF selection number)** | **0.69904566** | — |
| iterMERGE7 stacked meta-blend, refit-on-all-valid (in-sample, NOT the selection number) | 0.69951206 | **0.68481672** |

Feature importances of the final refit GBM: `lgb=77, fm=58, xgb=54, i63=11`
(gain-based split counts) — LGB dominates, consistent with it being the
strongest single component; i63 (the weakest, most recently added
component) contributes least, consistent with `iterMERGE5`/`iterMERGE6`'s
findings that i63's marginal value is small.

**Notable side finding**: test primary from the refit-on-all-valid model
(0.68481672) is *higher* than both the 3-model reference test (0.68432260,
delta **+0.00049**) and iterMERGE5's linear-optimum test (0.68367088, delta
**+0.00115**) — i.e., on the in-sample-refit number, the nonlinear blend
*does* appear to close (and reverse) the valid/test crossover that
iterMERGE5 and iterMERGE6 both showed. However, this test number is
reported for the record only and carries no weight in the verdict: (1) the
refit-on-all-valid valid number (0.69951206) is itself optimistic/in-sample
and is not the number used for selection — the correct, non-overfit
selection number is the **OOF** valid score (0.69904566), which is already
below the 3-model reference; (2) a single test-side observation on one
model is not a robustness check (no seed/fold variation was run on the
test side, unlike iterMERGE5's per-seed confirmation pass) and the primary
governing discipline of this project is valid-only selection — this result
does not change the verdict.

## Verdict

**REJECT.** The best OOF (k-fold, user-level, non-overfit) valid score
across both meta-learner families — shallow GBM at `num_leaves=2`,
**0.69904566** — is *below* both the 3-model reference (0.69943440, delta
**-0.00038874**) and iterMERGE5's linear weight-search optimum (0.69981837,
delta **-0.00077271**). Logistic regression is worse still (best OOF
0.69856703). Neither meta-learner family clears even `PRELIMINARY_DELTA
=0.0003` against the 3-model reference, let alone beats iterMERGE5's
already-PRELIMINARY linear point.

This answers the round's central question directly: a nonlinear/
interaction-aware combination of these 4 specific components does **not**
outperform the best linear weight point once evaluated honestly via
out-of-fold cross-validation — the valid/test crossover iterMERGE5 and
iterMERGE6 both surfaced appears to be **intrinsic to combining these 4
components** (a genuine generalization gap between iterMERGE5's linear
grid-search optimum and test, likely from searching a fine 135-combo grid
on a single valid split), not an artifact of using linear weights that a
smarter combiner could fix. The apparent shallow-GBM interaction structure
that the meta-learner tries to fit (LGB/FM/XGB dominant, i63 marginal,
gain-based importances 77/58/54/11) does not generalize better
out-of-fold than a simple fixed weighted sum — likely because 4 input
features with only ~125K training rows and heavy multicollinearity between
components (all 4 are within-user-percentile scores predicting the same
label) leave little genuine nonlinear signal beyond what linear weighting
already captures, while adding meta-learner variance that the k-fold
evaluation correctly penalizes. The refit-on-all-valid test number
(0.68481672, nominally the project's best-ever test score if it could be
trusted) is an interesting directional signal but is not evidence of a
real, non-overfit gain given the OOF valid check it must pass first does
not clear the bar.

Best-known promotable number remains unchanged at valid 0.69943440 / test
0.68432260 (yixi10 3-model blend). Best PRELIMINARY (not promoted) finding
remains iterMERGE5's raw 4-model blend (valid 0.69981837/test 0.68367088).

## Artifacts

- `run.py` — single end-to-end script: imports `iterMERGE5_four_model_blend/run.py`
  as a module and reuses its verified `fit_lgb`/`fit_xgb`/`sigmoid`/
  `within_user_percentile`/`stable_user_order`/`blend4`/column lists/
  row-alignment method directly (no re-derivation); reproduces the 3-model
  and iter63-standalone harness-fidelity checks plus a fresh sanity check
  against iterMERGE5's exact raw-4-model-blend numbers; trains all 4
  components (iter63 at 5 seeds, sigmoid-mean ensemble); runs 5-fold
  user-level OOF stacking for logistic regression (8-point `C` sweep) and
  shallow GBM (`num_leaves` 2/3/4 sweep); refits the winning meta-learner on
  all of valid and scores test. Writes `results.json` incrementally after
  each stage (`harness_fidelity_3model` → `harness_fidelity_iter63_standalone`
  → `row_alignment_check` → `harness_fidelity_merge5_raw4model_sanity` →
  `oof_stacking_logreg_sweep` → `oof_stacking_gbm_sweep` →
  `oof_stacking_results` → `refit_and_test` → `summary`).
- `results.json` — full numeric record of every stage above.
- `run.log` — full stdout of the run (elapsed 427.9s).
- No files outside this directory were modified;
  `iterMERGE5_four_model_blend/run.py`, `iter63_decay_tab_rate/train.py`,
  `iterYIXI10_video_metadata/features.py`, and `make_submission.py` were
  imported directly as modules (read-only), not copied or edited.

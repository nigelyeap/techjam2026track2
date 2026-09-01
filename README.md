# KuaiRand-Pure Within-User Ranking: TikTok TechJam 2026, Track 2

**Test primary 0.68432260 against the organizer FM baseline's 0.5946: +0.08972 absolute, +15.09%
relative.** Reached through 103 logged research iterations across four tracks, most of it driven
autonomously by Claude Code with a human setting scope and approving each promotion, and every
claimed number independently re-verified from raw CSVs before being trusted.

Autonomous ML research submission for Track 2 (AutoML Research Agent for RecSys). The task is
within-user ranking on [KuaiRand-Pure](https://kuairand.com): each user's own logged exposures
get ranked against each other, scored by `mean(GAUC, nDCG@5)` ("primary") on the `long_view`
label. GAUC is per-user AUC averaged across users, the standard grouped-ranking metric in this
literature (not a global AUC across all users' exposures pooled together).

Validation primary (used for all model selection) is 0.69943440. The submitted model is a
within-user-percentile blend of three components: a factorization machine trained with pairwise
BPR (10% weight), a LightGBM `LGBMRanker` with `num_leaves=2`, `linear_tree=True`, lambdarank
objective (52%), and a tuned `XGBRanker` (38%). The 103 iterations split across four tracks: 80
main-track, 11 on one teammate's independent branch, 9 in a final merge-and-verify track, and 3
in a second teammate's independent verification lane. See
[`SUBMISSION.md`](SUBMISSION.md) for the full numeric history and [`DEVPOST.md`](DEVPOST.md) for
the project narrative.

## Results

| | GAUC | nDCG@5 | primary (test) |
|---|---|---|---|
| FM baseline (organizer-provided) | 0.6610 | 0.5282 | 0.5946 |
| **Final blend (FM 10% / LightGBM 52% / XGBoost 38%)** | n/a | n/a | **0.68432260** |

GAUC/nDCG@5 are marked n/a for the final blend because test labels aren't public; only the
composite primary score comes back from the submission check, so it's the only test-set number
we can report for the blend without fabricating a breakdown.

The blend was reached in two phases: a primary track of 60+ iterations (FM with pairwise BPR
loss, recency-decay features, activity-weighted sampling, then a seed-ensemble and a
`linear_tree=True` LightGBM layered on top by blend, converging at test 0.65955), then a final
merge with a teammate's independently-developed LightGBM/XGBoost feature and objective tuning,
which added the XGBoost component and pushed test to 0.68432. Full detail, including every
rejected direction and why, is in [`SUBMISSION.md`](SUBMISSION.md) and
[`experiments/LEDGER.md`](experiments/LEDGER.md).

Reproduction commands to check these numbers yourself are directly below.

## Setup and installation

Python 3.9+. The base FM/BPR line (`baseline.py`, `train_bpr.py`) needs only `numpy`, no torch,
pandas, or sklearn. Reproducing the **final submitted model** additionally needs `pandas`,
`lightgbm`, and `xgboost` for its two GBM components:

```bash
pip install numpy pandas lightgbm xgboost
```

There's no `requirements.txt` in the repo; the above four packages are the complete set actually
imported anywhere in the reproduction path (`make_submission.py`, `data.py`, `baseline.py`,
`evaluate.py`, `submit.py`). A handful of individual experiment scripts under `experiments/`
(the calibration and stacking ones, e.g. `iterMERGE6_calibrated_4model`, `iterMERGE7_stacked_meta_blend`)
also import `scikit-learn`, but none of those are on the critical path to reproducing the final
submission.

Data: KuaiRand-Pure, downloaded from Zenodo (no registration needed):

```bash
# from the repo root; extracts to ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

`KuaiRand-Pure/data/` is gitignored (it's ~200MB of CSVs), so this step is required after a
fresh clone. Every script here defaults to `./KuaiRand-Pure/data` and takes `--data_dir` to
override it.

## Reproducing the final result

From a clean checkout, with the data downloaded as above:

```bash
python3 make_submission.py submission.csv
```

This rebuilds the causal feature frames (5-day user-decay, causal watch-depth history,
`upload_type`), trains the tuned XGBoost ranker and LightGBM ranker (a few seconds each on CPU),
trains the FM/BPR ensemble at 5 seeds (~25s each, one core, about 2 minutes total), blends all
three via within-user-percentile normalization at the confirmed weights (10% FM / 52% LightGBM /
38% XGBoost), evaluates on valid and test, writes `submission.csv` in the format `submit.py`
requires, and self-checks the written file with `submit.py`'s own alignment logic before exiting.
`make_submission.py` takes one optional positional argument, the output path (default
`submission.csv`).

Validate the output independently:

```bash
python3 submit.py --check submission.csv
```

This was run against the committed `submission.csv` before writing this README: **passes, 170,588
rows, split=test.** `submit.py` also supports `--make` (generate a sample submission from the
official FM baseline) and `--score` (validate and score against `valid` locally, since test
labels aren't available for local scoring):

```bash
python3 submit.py --make  --split test  some_baseline.csv   # sanity: regenerate the FM baseline
python3 submit.py --score --split valid submission.csv       # score against valid
```

To reproduce the FM/BPR line's own 5-seed result directly against its experiment harness (no
GBMs, numpy only):

```bash
cd experiments/iter27_triple_fusion
python3 driver.py   # or see results.json for the already-recorded run
```

## Repo structure

```
data.py                     official data loading + split + baseline feature encoding
baseline.py                 FM / item-popularity / random baselines (evaluate.py is the
                             scoring source of truth, not this file)
evaluate.py                 GAUC / nDCG@5 / primary metric implementation (do not modify)
submit.py                   generate/validate/score submission CSVs
train_bpr.py                standalone BPR training script
ablation_features.py        feature ablation experiments (organizer-provided groundwork)
make_submission.py          end-to-end reproduction of the final submitted model
submission.csv              the submitted predictions (test split, validated)
baseline_scores.json        officially published baseline scores + seed variance

SUBMISSION.md                final numeric writeup: exact scores, model description, and repro
                              commands
DEVPOST.md                   project narrative for the Devpost submission
YIXI_INSTRUCTIONS.md / XUXIA_INSTRUCTIONS.md / YIXI_SUMMARY.md
                             teammate coordination docs from the collaborative research phase

experiments/                 one directory per research iteration (iter1 .. iter81,
                             iterYIXI1..11, iterMERGE1..9, iterXUXIA1..3), each with its own
                             script(s) and a RESULT.md
experiments/LEDGER.md        full iteration log for the primary GBM/feature-engineering track
                             (2449 lines, 60+ iterations)
experiments/MERGE_LEDGER.md  log of the final merge/verification track: combining this
                             project's FM+GBM line with a teammate's independent LightGBM/
                             XGBoost blend and searching for further gains (9 experiments)
XUXIA_SUMMARY.md             condensed writeup of a second teammate's independent verification
                             lane against the pre-merge blend: per-segment blend weighting,
                             rank/calibrated fusion, and GBM-native multi-task stacking
                             (3 experiments, all REJECT, none touching the submitted model)

KuaiRand-Pure/                organizer-provided dataset package (LICENSE, loader, data/);
                              the data/ subdirectory is gitignored and must be downloaded,
                              see above
```

## Limitations and what we'd improve given more time

Every item below was checked before it was written down, not left as a guess. The first one in
particular was verified two independent ways before we trusted it enough to report.

- **A valid/test crossover surfaced in the final merge search and isn't fully explained.** Once a
  4th blend component (this project's own GBM) was added to the 3-model reference and its weight
  tuned on validation, the resulting point beat the 3-model reference on valid by +0.00038 but
  *lost* to it on test by -0.0004 to -0.0009 at every weight checked
  (`experiments/iterMERGE5_four_model_blend`). Two independent checks rule out grid-search
  overfitting as the cause: a follow-up experiment reselected the same weight point under 5-fold
  user-level cross-validation and got the identical answer
  (`experiments/iterMERGE9_cv_regularized_blend`), and a separately-implemented nonlinear
  stacking meta-learner showed the same pattern under honest out-of-fold evaluation
  (`experiments/iterMERGE7_stacked_meta_blend`). Two independent methods agreeing on the same
  result points at a genuine distributional difference between the valid window
  (20220422-20220428) and the test window (20220429-20220508) that we don't yet understand. We
  left this line as a documented, not-promoted finding rather than force it into the submission
  (the promoted model never uses it). It deserves real temporal-robustness analysis (date-shifted
  splits, drift diagnostics on the underlying feature distributions) before trusting any future
  fine-margin gain found this way.
- **The 4-model search space for the current model families is fairly exhausted.** Nine
  merge-track experiments tried linear reweighting, per-component calibration, nonlinear
  stacking, decay-feature transfer across the two tracks, and CV-regularized selection; all but
  one landed as a clean reject, and the one exception is the crossover finding above. Further
  gains from blending these same four components (FM, this project's GBM, yixi's LightGBM,
  yixi's XGBoost) look like a dead end. The more promising next step is a genuinely new signal
  source: sequence modeling over each user's raw interaction history (DIN/SIM-style attention,
  scoped early in the project but never reached before convergence), or multi-task learning
  against the other engagement labels (`is_click`, `is_like`, `is_follow`, `is_comment`,
  `is_forward`) with a different architecture than the two we tried. Both regressed monotonically
  with auxiliary weight (see `experiments/LEDGER.md`'s multitask sections), and the failure mode
  diagnosed there was specific to sharing an embedding table with a BPR objective, which a
  differently-structured multi-task model might avoid.
- **Listwise (grouped-softmax) loss was tried and closed, not left untested.** It was
  gradient-verified against finite differences and swept across a 10x learning-rate range;
  every setting peaked at epoch 1 and degraded after, diagnosed as instability from the
  per-step negative resubsampling. A variant using all of a user's negatives per group (instead
  of the capped subsample that caused the instability) is the natural follow-up but wasn't
  attempted, since it pushes per-group sizes into the hundreds for high-degree users.
  Watch-time-as-censored-regression (the direction the CWM reference implementation takes) was
  deprioritized from the start over its `torch==1.6.0` dependency risk this close to the
  deadline, and not attempted at all.
- **The valid/test gap widens as GBM tree capacity shrinks toward `num_leaves=2`** (about 0.013
  at `num_leaves=2` vs. 0.002-0.003 at `num_leaves=7`, stable across seeds and confirmed under a
  date-shifted split). This is documented rather than hidden; `num_leaves=7` is a tighter-margin,
  lower-variance fallback if that gap becomes a concern in a different deployment setting.

## Team contributions

The human directed the project throughout: set scope, made every promote/reject call on
borderline findings, decided when to reopen post-convergence research and when to stop, and made
the final call to merge and submit the blended model. A teammate, Yixi, independently developed
the LightGBM feature engineering, the lambdarank objective retuning, and the XGBoost ranker that
together make up 90% of the final blend's weight, working from a separate branch of the same
starter kit. A second teammate, Xuxia, independently stress-tested the pre-merge blend from a
different angle: per-segment blend weighting, rank and calibrated fusion, and GBM-native
multi-task stacking. All three closed as REJECT (`XUXIA_SUMMARY.md`); the isotonic-calibration
collapse she found is an independent corroboration of the same failure mode the merge track hit
separately in `iterMERGE6_calibrated_4model`. The bulk of the iterative experimentation itself
(running each hypothesis, checking harness fidelity, computing and cross-checking metrics,
writing up results, and orchestrating which direction to try next) was carried out autonomously
by Claude Code agent sessions under human direction, across the primary research track, the final
merge/verification track that combined the two main lines of work and re-verified every claimed
number from raw CSVs before it was trusted, and Xuxia's independent verification lane.

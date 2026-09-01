# iterMERGE4: multitask auxiliary-label stacking on the 3-model blend

## Hypothesis

`iter67_multitask_gbm` (see `experiments/iter67_multitask_gbm/RESULT.md`)
tested whether predictions of the dataset's other engagement labels
(`is_like`/`is_follow`/`is_comment`/`is_forward`, all <2% prevalent), fed
back as extra GBM input columns, add signal beyond what the main
`long_view` feature set already captures. It was a clean REJECT there
(exact-zero delta, 0 split usage) against the old 2-model `iter63`
`rate_only` single-LightGBM harness. This experiment re-tests the same
idea against yixi's richer current-best representation instead: LightGBM's
`LGB_CANDIDATE_COLUMNS` (includes `decay_tab_rate_3`, `meta_upload_type`,
causal `hist_watch_decay_mean_5`) and XGBoost's `XGB_COLUMNS`, feeding the
3-model 10% FM / 52% LGB / 38% XGB blend that currently reaches valid
primary 0.69943440 / test 0.68432260
(`experiments/iterYIXI10_video_metadata/RESULT.md`,
`experiments/iterMERGE1_verify_yixi10/RESULT.md`). Unlike iter67, this is
the first time the aux-column idea is tested against an XGBoost component
at all.

## Harness-fidelity check (must pass before trusting anything else)

Reproduced from scratch, seed 0, same code path as `iterMERGE1_verify_yixi10/verify.py`
(yixi's `features.py` chain via `load_frames()`, independent LGB/XGB/FM
training, `make_submission.py`'s `train_one_fm`):

| Check | Reference | Reproduced | Delta |
|---|---|---|---|
| LightGBM valid (`LGB_CANDIDATE_COLUMNS`) | 0.68834144 | 0.68834144 | 0.0 |
| XGBoost valid (`XGB_COLUMNS`) | 0.66755420 | 0.66755420 | 0.0 |
| FM valid (5-seed ensemble) | 0.63987792 | 0.63987792 | 0.0 |
| 3-way blend valid (10/52/38, within-user-percentile) | 0.69943440 | 0.69943440 | 0.0 |
| 3-way blend test | 0.68432260 | 0.68432260 | 0.0 |

All five checks passed at exact (`<=1e-6`) tolerance -- see
`results.json`'s `harness_fidelity` block (`all_pass_1e-6: true`). Every
delta reported is exactly `0.0`, not just rounded.

## Aux-label recovery and alignment verification

`aux_labels.py` (iter67's own recovery logic, imported directly, not
copied) re-parses the two raw log CSVs in file-then-file order and
recovers `is_like`/`is_follow`/`is_comment`/`is_forward` per `orig_idx`.
Yixi's feature frames do not expose `orig_idx` themselves, so alignment
had to be independently re-derived: `iterYIXI9_watch_depth_history/features.py`'s
own `load_raw_rows()` builds its `rows` list (with an `ORIG` field defined
identically to iter63's -- a running counter over the same two raw files,
incremented before any date filtering) and this is precisely the row
sequence yixi's chained frames are built from (her own `build_frames()`
asserts user/label alignment against it internally). Splitting that `rows`
list by the same `lengths` dict used to build the frames gives `orig_idx`
per split.

Verified exhaustively (not spot-sampled): reconstructed `long_view` via
`orig_idx` compared against the already-trusted `y[split]` for **every
row in every split** -- 1,141,112 train + 124,909 valid + 170,588 test
rows, **0 mismatches** (see `results.json`'s `alignment_check` block).
The `orig_idx` alignment claim from iter67 holds exactly on yixi's causal
feature frames too.

Train-split prevalence: `is_like` 1.87%, `is_follow` 0.10%, `is_comment`
0.26%, `is_forward` 0.10% -- essentially identical to iter67's numbers
(same underlying data), all rare.

## Implementation

- **Auxiliary classifiers**: 4 independent `LGBMClassifier`
  (`n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=0`),
  mirroring iter67's config exactly, via 5-fold OOF on train
  (`KFold(5, shuffle=True, random_state=0)`, no leakage) plus a separate
  full-train fit applied to valid/test. Trained once on the **union** of
  `LGB_CANDIDATE_COLUMNS` and `XGB_COLUMNS` (a single shared native
  feature set), producing one set of 4 aux prediction columns
  (`aux_is_like`/`aux_is_follow`/`aux_is_comment`/`aux_is_forward`) usable
  by both downstream models -- per the task's framing of "add to both".
  OOF diagnostics show the same low-information signature iter67 found:
  e.g. `is_follow` OOF mean_pred=0.0021 vs actual rate=0.0010 (see
  `results.json`'s `aux_classifier_diagnostics`).
- **Main models**: the 4 aux columns appended to `LGB_CANDIDATE_COLUMNS`
  and to `XGB_COLUMNS` separately, each retrained at seed 0 with yixi's
  exact existing hyperparameters (`LGB_CONFIG` / the yixi5-tuned XGB
  config), then re-blended at the existing 10/52/38 weights, individually
  and combined.

## Standalone ablation (single seed = 0)

| Model | baseline valid | +aux valid | delta | aux columns ever split on? |
|---|---|---|---|---|
| LightGBM (`LGB_CANDIDATE_COLUMNS`) | 0.68834144 | 0.68834144 | **+0.00000000** | No (0/4, importance all 0) |
| XGBoost (`XGB_COLUMNS`) | 0.66755420 | 0.65837336 | **-0.00918084** | Yes (12/2/8/9 splits) |

LightGBM (`num_leaves=2`, `linear_tree=True`) reproduces iter67's exact
finding on this richer feature set too: bit-for-bit identical predictions
with the aux columns present, 0 split usage across all 4 columns. This is
the same mechanism iter67 diagnosed -- these near-base-rate, sub-2%-prevalent
predictions never look attractive next to `decay_rate_5`/`decay_tab_rate_3`/
`hist_watch_decay_mean_5` to a heavily regularized 2-leaf tree learner.

XGBoost (`max_depth=1`, 1000 single-split trees) is different: unlike
iter67's LightGBM-only harness, this is the first time the aux-column idea
has been tested against an XGBoost component, and here it *does* split on
the aux columns (31 total splits across the 4 columns) -- but this
**actively hurts** valid primary by -0.00918, an order of magnitude past
both the preliminary (-0.0003) and promotion (0.001) thresholds in the
wrong direction. With only 1-split trees per round, XGBoost has very few
"good" splits available per tree; when a near-random column happens to
look locally attractive by its split criterion, it consumes a full tree's
capacity on noise that the OOF diagnostics show carries almost no real
signal (`is_follow`/`is_comment`/`is_forward` are all within 2x of their
own base rates in the classifiers' own out-of-fold predictions).

## Blend retest

Performed for completeness even though nothing cleared the standalone
positive gate (XGBoost regressed, LightGBM was a no-op):

| Variant | valid primary | delta vs. baseline | test primary |
|---|---|---|---|
| baseline / baseline (reference) | 0.69943440 | -- | 0.68432260 |
| LGB+aux / baseline XGB | 0.69943440 | **+0.00000000** | 0.68432260 |
| baseline LGB / XGB+aux | 0.69567251 | **-0.00376189** | 0.68175703 |
| LGB+aux / XGB+aux (both) | 0.69567251 | **-0.00376189** | 0.68175703 |

The LGB+aux variant is bit-identical to baseline (LightGBM never used the
columns, so the blend is unaffected). Any variant that includes the
XGB+aux component drags the full 3-model blend down by -0.00376 valid --
well past the -0.001 promotion-delta magnitude in the harmful direction,
confirming the standalone XGBoost regression survives ensembling rather
than being absorbed/cancelled by FM and LightGBM.

## 5-seed confirmation

Not performed. Per protocol, a 5-seed confirmation is only warranted for a
delta clearing +0.001 in a beneficial direction; every ablation here is
either exactly zero or clearly harmful, so there is nothing to confirm.

## Verdict

**REJECT.** `best_variant = baseline_baseline`, `best_valid_delta = +0.00000000`
(see `results.json`'s `verdict` block). Auxiliary engagement-label
prediction columns:
- add nothing to LightGBM on yixi's richer reference representation (exact
  replication of iter67's finding, now on a different feature-interaction
  regime),
- and actively **hurt** XGBoost (-0.00918 valid standalone, -0.00376 valid
  in the full blend) -- a new finding this experiment surfaces that iter67
  could not, since iter67 never had an XGBoost component to test against.

This closes the multitask/aux-label-stacking direction on both harnesses
now tried (2-model iter63 rate-only GBM, and yixi's 3-model FM/LGB/XGB
blend): no configuration of this idea has produced a positive result, and
against XGBoost specifically it is actively counterproductive. No further
work on this feature family is recommended.

## Best blend result

No new blend was produced; the current best remains yixi's 0.10 FM /
0.52 LGB / 0.38 XGB blend at valid primary **0.69943440** / test primary
0.68432260 (`experiments/iterYIXI10_video_metadata/RESULT.md`), unchanged
by this experiment.

## Artifacts

- `run.py` -- single end-to-end script: harness-fidelity check, orig_idx
  alignment re-derivation/verification, 4 aux `LGBMClassifier` OOF
  training, LGB+aux/XGB+aux retraining, and the 4-way blend ablation.
  Writes `results.json` incrementally after each stage (`harness_fidelity`
  -> `alignment_check` -> `aux_classifier_diagnostics` ->
  `standalone_ablation` -> `blend_ablation`/`verdict`).
- `results.json` -- full numeric record of every stage above.
- `run.log` -- full stdout of the run (elapsed 1013.5s / ~16.9 min).
- No files outside this directory were modified; `aux_labels.py` was
  imported directly from `experiments/iter67_multitask_gbm/` (read-only),
  not copied.

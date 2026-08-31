# iter68 — static side-info features, retested on the GBM-native representation

## Provenance

Self-directed research iteration, per instruction: *"i want you to comprehensively
test that field in xuxia's direction, you are a self researching agent. find more
methods, or maybe combine methods that work with our current existing model."*

A census of all 67 prior iterations (`experiments/LEDGER.md`) surfaced a genuine,
previously-untested gap: iter15 (`experiments/iter15_side_info/`) tested static
user/video side-info (`user_features_pure.csv`, `video_features_statistic_pure.csv`)
and rejected it — but only on the **FM-bucketed** representation, which predates
the iter44 discovery that the **GBM-native** (un-bucketed) representation behaves
very differently. Those features were never retested natively. Additionally,
`video_features_basic_pure.csv` (video_type, upload_type, content tag, music_id)
has never been touched by any iteration, bucketed or native.

## Hypothesis

Static side-info that failed as one-hot/bucketed FM inputs (iter15) might carry
real signal once exposed as native LightGBM categorical/numeric splits on top of
iter63's current-best `rate_only` feature set, the same way several other
directions flipped from REJECT (bucketed) to ACCEPT (native) earlier in the
project (per the iter44 precedent).

## Implementation

`experiments/iter68_side_info_native/run.py`, built on `iter63_decay_tab_rate/train.py`'s
`rate_only` feature set and hyperparameters (`num_leaves=2, learning_rate=0.10,
n_estimators=500, min_child_samples=200, reg_lambda=1.0, linear_tree=True`), unchanged.

Three independent joins, each added on top of the unchanged `rate_only` base and
ablated separately and combined:

- **`user`**: 6 demographic fields from `user_features_pure.csv` (active_degree,
  live_streamer, video_author, follow_range, fans_range, register_range) — identical
  fields to iter15, joined natively as categoricals instead of one-hot.
- **`video_stat`**: 5 engagement-count fields from `video_features_statistic_pure.csv`
  (play/like/share/complete/follow counts), log1p-transformed, added as numerics —
  identical fields to iter15.
- **`video_basic`**: 3 categorical fields from `video_features_basic_pure.csv`,
  never touched by any prior iteration — `video_type`, `upload_type`, and
  `primary content tag` (first tag in the video's comma-separated tag list, `UNK`
  if empty). 44 distinct tag values in train.

A harness-fidelity check (`assert`) reproduces iter63's exact published `rate_only`
baseline (valid=0.67168, test=0.65353) before any new number is trusted.

## Result (seed 0)

| variant | valid | test | Δvalid | Δtest |
|---|---|---|---|---|
| baseline | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| +user | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| +video_stat | 0.64011 | 0.62538 | **−0.03157** | −0.02815 |
| +video_basic | 0.64268 | 0.63226 | **−0.02900** | −0.02127 |
| +all | 0.64268 | 0.63226 | −0.02900 | −0.02127 |

`user` is an exact no-op (0 splits used — the model ignores it entirely).
`video_stat` and `video_basic` both cause large regressions, and `+all` is
bit-for-bit identical to `+video_basic` alone, meaning `video_stat`'s damage is
fully subsumed/dominated by `video_basic`'s.

### Isolating `video_basic`'s regression (`debug.py`)

Per-sub-field ablation of `video_basic`'s three categorical columns:

| sub-field alone | valid | test |
|---|---|---|
| `v_type` (2 levels) | 0.67168 | 0.65353 (no-op) |
| `v_upload_type` (few levels) | 0.67142 | 0.65308 (negligible, −0.00026 valid) |
| `v_tag_primary` (44 levels) | **0.64268** | **0.63226** |

The entire `video_basic` regression is attributable to `v_tag_primary` alone — its
solo number matches the combined `video_basic` and `all` results exactly.

### Diagnosis

`v_tag_primary` is a 44-level unordered categorical, far higher cardinality than
any other categorical this project uses (`tab` has ~7 levels, `v_type` has 2-3).
Feature importance (`model.feature_importances_`, split counts) shows it's only
selected ~6 times out of 47 total splits in the augmented model — a small direct
footprint — yet its presence collapses `best_iteration_` from 48→47 and destroys
performance. At `num_leaves=2` (a single split per tree), a high-cardinality
categorical's greedy split search is far noisier than for a low-cardinality one,
and evidently destabilizes the shallow booster's overall tree structure even when
rarely chosen.

(Note: `linear_tree=True` fits a linear regression over all numeric features in
every leaf, so `feature_importances_` split counts undercount a *numeric*
feature's true contribution — e.g. iter63's own `decay_tab_rate_3` shows near-zero
split counts despite being a proven, real signal. This caveat does not change the
diagnosis here since `v_tag_primary` is categorical, not numeric, and categoricals
are not folded into the leaf-level linear model.)

### 5-seed robustness confirmation (`confirm5_tag.py`)

Given the magnitude was unusually large relative to any prior REJECT in this
project (~30x larger), it was checked across 5 seeds before writing up:

| seed | base valid | +tag valid | Δvalid | base test | +tag test | Δtest |
|---|---|---|---|---|---|---|
| 0 | 0.67168 | 0.64268 | −0.02900 | 0.65353 | 0.63226 | −0.02127 |
| 1 | 0.67105 | 0.63567 | −0.03539 | 0.65323 | 0.62460 | −0.02863 |
| 2 | 0.67104 | 0.63567 | −0.03538 | 0.65320 | 0.62460 | −0.02860 |
| 3 | 0.67104 | 0.63567 | −0.03538 | 0.65320 | 0.62460 | −0.02860 |
| 4 | 0.67105 | 0.63567 | −0.03539 | 0.65323 | 0.62460 | −0.02863 |

mean Δvalid = **−0.03411**, wins = 0/5. mean Δtest = −0.02715, wins = 0/5.

Confirmed robust across seeds, not a seed-0 fluke.

## Verdict: REJECT

- `user` demographics: no-op (native categorical, unlike iter15's bucketed REJECT
  which showed an active regression — the native representation neutralizes but
  does not rescue this signal).
- `video_stat` engagement counts: REJECT, large regression.
- `video_basic` / `v_type`, `v_upload_type`: no-op / negligible.
- `video_basic` / `v_tag_primary`: REJECT, large and 5-seed-confirmed regression.
  Raw high-cardinality content-tag as a native categorical split feature is
  actively harmful to this shallow (`num_leaves=2`) booster.

iter63 remains the current best and correctly promoted candidate. The content-tag
signal itself is not necessarily dead, though — see iter69, which reshapes it via
the project's own proven decayed-rate mechanism instead of using it as a raw
categorical split.

# iterMERGE2: decay-axis transfer onto yixi's reference representation

## Hypothesis

This project's own research track (iter71/72/73) found three causal
decayed-engagement-rate feature axes -- author-popularity decay rate,
duration-bucket decay rate, and hour-of-day decay rate, each Laplace-smoothed
`(decayed_pos + 0.5) / (decayed_total + 1.0)` over a half-life-3 exponential
decay window, same construction as yixi's `decay_tab_rate_3` -- but tested
them only against the old `iter63_decay_tab_rate` single-model rate-only
harness, where all three came back as clean no-ops (exact-zero delta,
REJECT). Since then, yixi's harness has grown a much richer reference
representation (`decay_rate_5`, `decay_act_5`, `lastk_rate`, `gap`,
`decay_tab_rate_3` alongside the categoricals) feeding a tuned
LightGBM/XGBoost/FM blend that currently reaches valid primary 0.69943440
(`experiments/iterYIXI10_video_metadata/RESULT.md`). The hypothesis under
test: even though these three axes were null on the old harness, they might
carry information yixi's current feature set doesn't already capture, and
could now clear the +0.0003 preliminary threshold standalone, and if so,
survive being added to her full best blend.

## Harness-fidelity check (must pass before trusting anything else)

Reproduced, at seed 0, using yixi's exact `LGB_CONFIG` /
`selected_xgb_config()` and column lists from
`iterYIXI9_watch_depth_history/common.py` and
`iterYIXI10_video_metadata/common.py` / `RESULT.md`:

| Check | Target | Reproduced | Delta |
|---|---|---|---|
| `LGB_REFERENCE_VALID` (LGB, `decay_tab_rate_3` cols) | 0.6768913269042969 | 0.6768913269042969 | 0.0 |
| `XGB_REFERENCE_VALID` (XGB, rate col) | 0.6697614192962646 | 0.6697614192962646 | 0.0 |
| `CURRENT_XGB_VALID` (XGB, raw-count col, production) | 0.6675541996955872 | 0.6675541996955872 | 0.0 |
| Best 3-model blend (FM 0.10 / LGB 0.52 / XGB 0.38, within-user-percentile) | 0.69943440 | 0.69943440 | 0.0 |

All four checks passed at exact (`<=1e-8`) tolerance -- see
`harness_results.json`. FM component reused from
`iterYIXI8_rank_space_calibration/.frozen_valid_predictions.npz` (hash-verified,
not retrained). Environment: python 3.14.6, lightgbm 4.7.0, xgboost 3.4.1.

## Standalone ablations (single seed = 0)

Each axis's rate-3 column was attached to a fresh copy of yixi's reference
frame (row-count, `user_id`, and label alignment asserted per split before
fitting -- see `common.py:attach_axis_rate`) and added to her reference LGB
columns and, separately, her reference XGB columns.

| Axis | Column | LGB valid | LGB delta | LGB gain frac. | XGB valid | XGB delta | XGB gain frac. | Clears +0.0003? |
|---|---|---|---|---|---|---|---|---|
| author (popularity) | `decay_author_rate_3` | 0.67689133 | +0.000000 | 0.0 | 0.66976142 | +0.000000 | 0.0 | No |
| duration-bucket | `decay_durbucket_rate_3` | 0.67689133 | +0.000000 | 0.0 | 0.66976142 | +0.000000 | 0.0 | No |
| hour-of-day | `decay_hour_rate_3` | 0.67689133 | +0.000000 | 0.0 | 0.66976142 | +0.000000 | 0.0 | No |

All six deltas are exact-zero at full float64 precision (not just rounded --
see `ablation_results.json`), and the new column receives **zero split gain**
in every one of the six fitted models. Both LightGBM (`num_leaves=2`,
`linear_tree=True`) and XGBoost (`max_depth=1`) never once chose to split on
any of the three new columns; predictions are byte-identical to the
reference model without the column. This is the same "clean no-op" signature
iter71/72/73 documented on the old harness, now reproduced on yixi's current,
much richer reference representation.

## Blend retest

Not performed. Per protocol, a full-blend retest is only warranted for a
feature that clears the +0.0003 preliminary threshold standalone. None of
the six ablations did, so there is nothing to carry into the 3-model blend --
attempting it would be noise on a feature both trees refuse to split on. Per
`iterYIXI6_cross_model_feature_transfer/RESULT.md`, standalone-positive
features have failed to survive ensembling before, but the converse -- a
standalone-null feature spontaneously helping the blend -- would require the
blend's tree-model components to use the feature differently than they did
standalone, which is not how either tree learner arrived at these particular
splits (importance is exactly 0 in both models regardless of column order in
`LGB_REFERENCE_COLUMNS` vs `XGB_REFERENCE_COLUMNS`).

## 5-seed confirmation

Not performed (gated behind a blend delta clearing +0.001, which never
triggered).

## Verdict

| Axis | Verdict |
|---|---|
| author-popularity decay rate | REJECT |
| duration-bucket decay rate | REJECT |
| hour-of-day decay rate | REJECT |

All three axes are REJECTed, both standalone (LightGBM and XGBoost
separately) and by extension for the blend (no retest needed since nothing
cleared the standalone gate). Combined with iter71/72/73's original findings
on the old harness, this closes the decayed-rate generalization family at
**4/4 nulls tested against 2/2 harnesses** (old iter63 rate-only harness, and
yixi's current richer reference representation) -- tag (iter69), author,
duration-bucket, and hour-of-day decay rates all fail to add information
beyond what `author_id`/`tab`/`duration_ms`/`decay_tab_rate_3` /
`decay_rate_5` already encode, on both the tree-only and current best-blend
setups. No further work on this feature family is recommended without a
different construction (e.g. a different half-life, a different smoothing
scheme, or a genuinely new grouping key not already implicit in the
categoricals).

## Best blend result

No new blend was produced; the current best remains yixi's
0.10 FM / 0.52 LGB / 0.38 XGB blend at valid primary **0.69943440** / test
primary 0.68432260 (`experiments/iterYIXI10_video_metadata/RESULT.md`),
unchanged by this experiment.

## Artifacts

- `common.py` -- shared constants, fitting helpers, `attach_axis_rate` (loads
  `iter71/72/73` `data_ext.py` modules read-only, computes the Laplace-smoothed
  rate, asserts row/user/label alignment before attaching).
- `run_harness.py` / `harness_results.json` -- fidelity check (all_ok=true).
- `run_ablation.py` / `ablation_results.json` -- the six standalone ablations.
- No files outside this directory were modified.

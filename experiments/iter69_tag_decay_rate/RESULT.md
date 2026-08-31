# iter69 — decayed per-tag rate (combining iter63's rate mechanism with the content-tag signal)

## Provenance

Direct follow-up to [iter68](../iter68_side_info_native/RESULT.md), which found that
the video's primary content tag (44-level categorical) is actively harmful as a raw
native categorical split (5-seed-confirmed mean Δvalid −0.03411), while the low-
cardinality `v_type`/`v_upload_type` fields are a no-op. Per the user's instruction
to *"combine methods that work with our current existing model,"* rather than
abandoning the tag signal, this reapplies the project's own proven "raw count →
Laplace-smoothed decayed rate" transformation — the exact mechanism iter63 used to
turn `decay_tab_3` into `decay_tab_rate_3`, a real, currently-promoted gain — to the
content-tag dimension instead of using the tag as a raw categorical.

## Hypothesis

A per-(user, tag) lazy-decayed engagement *rate* (smoothed, numeric, bounded [0,1])
may carry the same underlying signal as the raw tag categorical without the
high-cardinality instability iter68 found, mirroring exactly why `decay_tab_rate_3`
succeeds where a hypothetical raw-`tab`-interaction categorical might not.

## Implementation

`experiments/iter69_tag_decay_rate/data_ext.py`: generalizes iter63's
`compute_decay_tab_features` (previously hardcoded to the `(user, tab)` key) into
`_compute_decay_key_features(rows, key_fn, halflives)`, a shared implementation now
used by both `compute_decay_tab_features` (key=`(user, tab)`, unchanged) and the new
`compute_decay_tag_features` (key=`(user, primary_tag)`, halflives 3d/7d). The
primary tag is joined from `video_features_basic_pure.csv` (first entry of the
comma-separated tag list, `UNK` if empty), identical to iter68's join.

**Causality verification** (mandatory before trusting any new causal traversal, per
project precedent): 30 random rows across both tag-halflives, `decayed_tag_total`
independently recomputed from raw data — max abs err 3.55e-15, no leakage detected.
Monotonicity sanity check (5000 rows): `decayed_tag_total >= decayed_tag_pos`
everywhere. Both passed cleanly.

`experiments/iter69_tag_decay_rate/train.py`: adds `decay_tag_rate_{3,7} =
(decayed_tag_pos + α) / (decayed_tag_total + 2α)`, α=0.5 (matching iter63's own
constant), on top of the unchanged `rate_only` feature set (which already includes
`decay_tab_rate_3`). A harness-fidelity check reproduces iter63's exact baseline
(valid=0.67168, test=0.65353) before trusting any new number.

Variants: `tag_rate_h3` (+halflife=3d only), `tag_rate_h7` (+halflife=7d only),
`tag_rate_both` (+both).

## Result (seed 0)

| variant | valid | test | Δvalid | Δtest |
|---|---|---|---|---|
| baseline (rate_only) | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| tag_rate_h3 | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| tag_rate_h7 | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| tag_rate_both | 0.67168 | 0.65353 | +0.00000 | +0.00000 |

Exact-zero delta (to 5 decimal places) for all three variants, on both valid and test.

### Diagnosis

Feature importance for `tag_rate_both` shows 0 split-count for both
`decay_tag_rate_3` and `decay_tag_rate_7` — but so does `decay_tab_rate_3` itself
(0 splits, despite being a real, proven contributor via `linear_tree=True`'s
per-leaf linear regression). To rule out the "0 splits ≠ 0 effect" trap that applied
to `decay_tab_rate_3`, checked the feature's distribution directly: `decay_tag_rate_3`
has real variance (mean 0.434, std 0.170, 84,910 distinct values in train) —
comparable to `decay_tab_rate_3`'s own distribution (std 0.204, 126,494 distinct
values) — so this is not a degenerate/constant feature. The exact-zero output
(identical to 5 decimals across all three configurations, not merely a small
positive/negative delta) indicates the leaf-level linear model is assigning it
essentially zero weight, i.e. the signal is fully redundant given the existing
`user_id`/`video_id`/`author_id`/`tab` categoricals already in the feature set —
plausible since each video has exactly one fixed tag, so `video_id` alone already
fingerprints any tag-level effect at the individual-video granularity, leaving no
residual signal for a coarser tag-level aggregate to add.

## Verdict: REJECT (clean no-op, not a regression)

Unlike iter68's raw-categorical version (actively harmful), the decayed-rate
reshaping is harmless — but also provides zero measurable benefit at any of the
three halflife configurations tested. Given the exact-zero result was identical
across three independent configurations (h3, h7, both) rather than a borderline
number, this is treated as a clean, confident null; a 5-seed confirmation was not
run since there is no positive or negative signal to validate (per established
project practice, only >0.001 valid gains or unusually large regressions warrant
multi-seed confirmation before write-up).

iter63 remains the current best and correctly promoted candidate. This closes off
the content-tag signal in both its tested forms (raw categorical: harmful; decayed
rate: redundant/no-op).

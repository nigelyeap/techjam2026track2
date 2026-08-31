# iter71 — decayed per-author rate

## Provenance

Continuation of self-directed "combine methods that work" research, following
[iter69](../iter69_tag_decay_rate/RESULT.md) (content-tag decayed rate: clean
no-op, redundant with `video_id`/`author_id`). `author_id` is already a strong
raw categorical in every feature set since early rounds, but no iteration has
ever tracked a *decayed engagement rate* at author granularity — only the raw
identity. Author sits naturally between per-video (too sparse to generalize)
and per-tab (too coarse) granularity, and needs no new CSV join since
`author_id` is already present in every row.

## Hypothesis

A per-(user, author) decayed engagement rate — same Laplace-smoothed lazy-decay
mechanism as `decay_tab_rate_3` — might capture "this user likes this specific
creator's output" independent of any single video, a genuinely different signal
from the raw `author_id` categorical split.

## Implementation

`experiments/iter71_decay_author_rate/data_ext.py`: generalizes the shared
`_compute_decay_key_features(rows, key_fn, halflives)` mechanism (already used by
`compute_decay_tab_features`) to a new `compute_decay_author_features`, keyed on
`(user_id, author_id)` instead of `(user_id, tab)`. Causality verified: 30
random rows across both halflives, brute-force recompute matches exactly (max
abs err 0.00e+00); monotonicity holds over 5000 rows; zero-activity rows
correctly show 0.0.

`train.py` adds `decay_author_rate_{3,7} = (decayed_author_pos + α) /
(decayed_author_total + 2α)`, α=0.5, on top of the unchanged `rate_only` set.
Harness-fidelity check reproduces iter63's exact baseline before trusting any
new number.

## Result (seed 0)

| variant | valid | test | Δvalid | Δtest |
|---|---|---|---|---|
| baseline (rate_only) | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| author_rate_h3 | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| author_rate_h7 | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| author_rate_both | 0.67168 | 0.65353 | +0.00000 | +0.00000 |

Exact-zero delta across all three configurations, matching iter69's pattern.

### Diagnosis (unifies with iter69's finding)

Both iter69 (tag) and iter71 (author) decayed rates are fully absorbed with zero
measurable effect, while `decay_tab_rate_3` (tab) is a real, proven gain. The
distinguishing factor: `tab` is a genuinely **coarser, cross-cutting** grouping
axis relative to the categoricals already in the feature set (`user_id`,
`video_id`, `author_id`) — a handful of UI surfaces each user revisits
repeatedly, so a per-(user,tab) rate aggregates across many distinct
videos/authors and adds information the model can't otherwise reconstruct. Tag
and author, by contrast, sit at or below the granularity of `author_id` /
`video_id`, which are already present as native categorical splits — so a
decayed rate at that same (or finer) granularity is redundant: the model can
already fit an author-specific (or finer) effect directly via the `author_id`
categorical split, with no need for a separate smoothed-rate summary of it.

This generalizes both nulls into a single, actionable finding: the decayed-rate
transformation adds value specifically when applied to a grouping axis coarser
than what's already in the categorical feature set, not as a generic
"reshape any raw signal into a smoothed rate" trick.

**Correction (post-iter72)**: this "coarseness" theory was directly tested
and falsified by [iter72](../iter72_decay_durbucket_rate/RESULT.md), which
built a duration-bucket axis deliberately as coarse as `tab` (6 levels vs.
~7) and not represented anywhere else as a categorical — under this theory it
should have succeeded like `tab` did. It produced the identical exact-zero
null instead. Coarseness alone is therefore not the operative property; see
iter72's RESULT.md for the revised diagnosis (`tab` is a
recommendation-surface/context signal, not merely a coarse content property).
The redundancy explanation for tag/author specifically (both already
fingerprinted by `video_id`/`author_id`) still stands on its own — only the
generalized "any coarse axis will work" claim above is withdrawn.

## Verdict: REJECT (clean no-op)

No multi-seed confirmation run — exact-zero across three independent
configurations is a confident, unambiguous null, consistent with iter69's
identical pattern.

iter63 remains the current best and correctly promoted candidate. This
closes off further "decayed rate at existing-or-finer categorical
granularity" variants (e.g. `music_id`) as low-expected-value — any such
variant would face the same redundancy with `video_id`/`author_id`, per the
diagnosis above (unaffected by the coarseness-claim correction, which
concerned coarse axes, not fine-grained ones).

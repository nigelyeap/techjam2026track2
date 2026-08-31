# iter72 — decayed per-(user, duration-bucket) rate

## Provenance

Direct test of the "coarseness" diagnosis proposed in
[iter71](../iter71_decay_author_rate/RESULT.md): that the decayed-rate
transformation only adds value on a grouping axis *coarser* than the
categoricals already in the feature set (as `tab` is, at ~7 levels, relative
to `user_id`/`video_id`/`author_id`), while tag/author (iter69/iter71) sit at
or below existing categorical granularity and are therefore redundant. If
that theory is correct, a duration-bucket axis — deliberately built coarse,
comparable to `tab`'s own cardinality, and not already represented as a
categorical anywhere in the feature set — should behave like `tab`, not like
tag/author.

## Implementation

`experiments/iter72_decay_durbucket_rate/data_ext.py`: buckets `duration_ms`
into 6 levels via fixed edges `[15000, 30000, 60000, 120000, 300000]` ms,
chosen from the train-split duration quantiles (10th≈11.5s, 25th≈25.6s,
median≈69s, 75th≈135s, 90th≈235s) — a fixed content property with no label
information, so no leakage risk by construction. Reuses the shared
`_compute_decay_key_features(rows, key_fn, halflives)` mechanism, keyed on
`(user_id, duration_bucket)`. Bucket distribution over train (163680, 163765,
170517, 309590, 287784, 45776 across buckets 0-5) confirms none of the 6
buckets are degenerate/empty.

Causality verified: 30 random rows x 2 halflives, brute-force recompute
matches exactly (max abs err 3.55e-15); monotonicity holds over 5000 rows.

`train.py` adds `decay_dur_rate_{3,7} = (decayed_dur_pos + α) /
(decayed_dur_total + 2α)`, α=0.5, on top of the unchanged `rate_only` set.
Harness-fidelity check reproduces iter63's exact baseline before trusting any
new number.

## Result (seed 0)

| variant | valid | test | Δvalid | Δtest |
|---|---|---|---|---|
| baseline (rate_only) | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| dur_rate_h3 | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| dur_rate_h7 | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| dur_rate_both | 0.67168 | 0.65353 | +0.00000 | +0.00000 |

Exact-zero delta across all three configurations — identical pattern to
iter69 (tag) and iter71 (author), despite duration-bucket being deliberately
built coarse.

## Diagnosis — this refutes iter71's "coarseness" theory, not confirms it

Duration-bucket is comparably coarse to `tab` (6 levels vs. `tab`'s ~7) and,
unlike tag/author, is not already represented anywhere else in the feature
set as a categorical. Under iter71's stated theory this should have produced
a real gain like `decay_tab_rate_3` did. It did not — a clean, unambiguous
null, matching the harness-fidelity-verified baseline to five decimal places
across both halflives and their combination.

This means "coarser than existing categoricals" is **not** the operative
property that makes `tab` special. **Revised diagnosis**: `tab` is not merely
a coarse content/context property — it is a **recommendation-surface /
traffic-source signal** (which UI feed the video was served through:
e.g. main feed vs. slide feed vs. follow feed in KuaiRand-Pure). A user's
engagement rate *conditional on which surface served the content* plausibly
reflects real differences in the recommender's own targeting quality and the
user's surface-specific intent (e.g. "I open the follow feed only for
creators I already like" vs. "I browse the main feed passively") — a
genuinely different causal signal from any property of the content itself
(duration, tag, author). Duration-bucket, tag, and author are all
*intrinsic-to-the-video* properties; `tab` is *extrinsic* (a property of the
serving context, not the content), which is the more likely dividing line
than raw cardinality.

This diagnosis is not itself verified by a further experiment in this round
(no second traffic-source-like column exists in KuaiRand-Pure to cross-check
against), so it should be treated as the best available explanation given
three consistent nulls on content-property axes vs. one real gain on the one
context-property axis tested — not as a proven mechanism.

**Correction (post-iter73)**: this content-vs-context theory was directly
tested and falsified by
[iter73](../iter73_decay_hour_rate/RESULT.md), which tested `hourmin`
(time-of-day served) — an unambiguous context/session property — as a
decayed rate. It produced the identical exact-zero null. Neither coarseness
(iter71's theory) nor content-vs-context (this file's theory) survives
contact with the next experiment; see iter73's RESULT.md for the closing
diagnosis on this family.

## Verdict: REJECT (clean no-op)

No multi-seed confirmation — exact-zero across three independent
configurations is unambiguous, consistent with iter69/iter71's identical
pattern. iter63 remains the current best and correctly promoted candidate.

This closes off the decayed-rate-generalization family for **content**
properties (tag, author, duration-bucket all null) as a productive direction
without a fresh hypothesis specifically targeting a context/serving-surface
signal — of which `tab` is the only one present in this dataset, already
exploited by iter63/iter16.

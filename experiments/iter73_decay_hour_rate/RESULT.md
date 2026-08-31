# iter73 — decayed per-(user, hour-of-day-bucket) rate

## Provenance

Direct test of the revised diagnosis proposed in
[iter72](../iter72_decay_durbucket_rate/RESULT.md), after iter72 falsified
iter71's "coarseness" theory: that `tab`'s uniqueness comes from being a
**context/serving-surface** signal (a property of the visit circumstance),
not an intrinsic **content** property like duration, tag, or author.
`hourmin` (time-of-day the impression was served) is exactly this kind of
signal — a property of the visit, not of the video — and has been carried in
the row tuple since iter18/iter48 but never used as a decayed rate on the
current representation (iter48 only tried a raw sin/cos numeric on the
pre-`linear_tree`, pre-`decay_tab_rate` iter44 baseline).

## Implementation

`experiments/iter73_decay_hour_rate/data_ext.py`: buckets `hourmin` into 6
four-hour windows (0-3, 4-7, 8-11, 12-15, 16-19, 20-23), comparable
cardinality to `tab` and to iter72's duration buckets. Reuses the shared
`_compute_decay_key_features` mechanism, keyed on `(user_id, hour_bucket)`.
Bucket distribution over train (88533, 69167, 182737, 227104, 252055,
321516 across buckets 0-5) confirms no degenerate/empty bucket.

Causality verified: 30 random rows x 2 halflives, brute-force recompute
matches exactly (max abs err 7.11e-15); monotonicity holds over 5000 rows.

`train.py` adds `decay_hour_rate_{3,7} = (decayed_hour_pos + α) /
(decayed_hour_total + 2α)`, α=0.5, on top of the unchanged `rate_only` set.
Harness-fidelity check reproduces iter63's exact baseline before trusting any
new number.

## Result (seed 0)

| variant | valid | test | Δvalid | Δtest |
|---|---|---|---|---|
| baseline (rate_only) | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| hour_rate_h3 | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| hour_rate_h7 | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| hour_rate_both | 0.67168 | 0.65353 | +0.00000 | +0.00000 |

Exact-zero delta across all three configurations — identical pattern to
iter69 (tag), iter71 (author), and iter72 (duration bucket).

## Diagnosis — this falsifies iter72's "context signal" theory too

Time-of-day is unambiguously a context/session property, not a content
property — under iter72's revised diagnosis this should have behaved like
`tab`. It did not: a clean, unambiguous null, bit-identical to the
harness-fidelity baseline.

This is now **4/4 nulls** across every axis tried in the decayed-rate
-generalization family (tag, author, duration-bucket, hour-of-day), spanning
both content and context properties, both fine and coarse cardinalities.
Only the original `tab` axis (iter16/iter63) produces a real gain. Two
theories have now been proposed and falsified in successive iterations
(coarseness in iter71→iter72, content-vs-context in iter72→iter73) — the
pattern no longer supports a clean categorical explanation for *what kind of
axis* generalizes. The more defensible remaining explanation is that `tab`'s
gain is closer to a **dataset-specific empirical fact** discovered by direct
search (iter16 swept several candidate signals and `tab` won), not an
instance of a broader "decayed-rate transformation generalizes to axis
property X" rule. It's also worth noting all four null variants tested
`ALPHA=0.5`/halflives `{3,7}` unchanged from `decay_tab_rate_3`'s own
settings — it remains possible (though untested here, and not pursued
further given 4 consecutive nulls) that a hyperparameter-specific match to
each axis's own natural timescale could unlock a real effect where the
`tab`-tuned defaults do not; this is flagged as a closed-off, low-priority
residual rather than a live hypothesis.

## Verdict: REJECT (clean no-op)

No multi-seed confirmation — exact-zero across three independent
configurations is unambiguous, consistent with the three prior iterations'
identical pattern. iter63 remains the current best and correctly promoted
candidate.

**This closes the decayed-rate-generalization family entirely** — 4
consecutive REJECTs across content and context axes, fine and coarse
cardinalities, is enough to stop generalizing this specific mechanism further
without a fundamentally new hypothesis for why an axis would differ from all
four already tested. Future rounds should pivot to a structurally different
feature/model lever rather than a 5th variant of "decay this key too."

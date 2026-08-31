# iter75 — video orientation + music_type side-info retest

## Provenance

iter68 tested `video_type`/`upload_type`/`tag(primary)` from
`video_features_basic_pure.csv` on the GBM-native representation and found
only the 44-level `v_tag_primary` regressed; `v_type`/`v_upload_type` were
no-ops. Two columns from that same file were never touched by iter68 or any
other iteration: `server_width`/`server_height` (video orientation/aspect
ratio) and `music_type`. `music_id` itself (7202 unique values) is
deliberately excluded — same high-cardinality-destabilization risk already
demonstrated by `v_tag_primary`, and expected to be near-video-granularity
redundant. This is a structurally different lever from Round 22 (decayed-rate
generalization, closed 4/4 REJECT) and Round 23 (num_leaves resweep, REJECT):
a raw side-info categorical retest, in the same family as iter68 but filling
its two untested fields.

## Implementation

`experiments/iter75_side_info_v2/run.py`: `v_orientation` derived from
`server_width` vs `server_height` (3 levels: portrait/landscape/square,
distribution 5650/1821/112 videos) — not the raw pixel dimensions, which have
156/120 unique values and would repeat `v_tag_primary`'s cardinality problem.
`v_music_type` taken directly from the CSV (5 non-null levels + `UNK` for 203
missing rows; distribution 6665/657/203(UNK)/41/14/3). Both are low-cardinality,
genuinely new content properties. Same harness as iter68: single seed=0,
iter55's exact winning LightGBM config, harness-fidelity check against iter63's
baseline before trusting any other number. No causality risk — both are static,
pre-recorded video properties with no label information.

## Result (seed 0)

| variant | valid | test | Δvalid | Δtest |
|---|---|---|---|---|
| baseline (rate_only) | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| +orientation | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| +music_type | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| +both | 0.67168 | 0.65353 | +0.00000 | +0.00000 |

Exact-zero delta across all three configurations.

## Diagnosis

Both fields are genuinely new signal, low cardinality (no v_tag_primary-style
regression risk), and still exact no-ops — not even a small negative like
iter70's interaction term. The recurring exact-zero pattern across most of
Round 22-24's null results (tag, author, duration-bucket, hour-of-day,
v_type/v_upload_type from iter68, now orientation/music_type) is consistent
with a structural property of `num_leaves=2`: a single-split tree per boosting
round only ever picks one categorical split variable, and LightGBM's
`linear_tree` leaf-regression only fits numeric features, not categoricals. A
new categorical with weaker signal than the existing `tab`/`user_id`/
`video_id`/`author_id`/`last1` split candidates simply never gets chosen as
the split feature across any of the 500 boosting rounds — hence bit-identical
output, not a small residual effect. This is a useful structural explanation
for *why* so many categorical additions have been exact zeros throughout this
project, though it doesn't change the practical conclusion for any individual
feature: none of these seven categorical additions add value at this model
capacity.

## Verdict: REJECT (clean no-op)

No 5-seed confirmation needed — exact-zero is unambiguous. iter63 remains the
current best. This closes out the last untested columns in
`video_features_basic_pure.csv`; no further static side-info from this file
remains to test (video_type, upload_type, tag, orientation, music_type all
now tested; music_id deliberately excluded on cardinality-risk grounds).

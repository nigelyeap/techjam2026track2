# iter15 — static side-info features (user_features_pure.csv + video_features_statistic_pure.csv)

## Idea
iter9's causal history features (`activity`/`tab_pos`/`rate`) are derived entirely from
the interaction log itself. Two dataset files ship alongside the log but had never been
touched anywhere in this run: `user_features_pure.csv` (one row per user_id — demographic
/ account-state fields) and `video_features_statistic_pure.csv` (one row per video_id —
aggregate engagement counters). This iteration asks whether stacking either or both of
these static side-info sources on top of iter9's best-known feature set (`activity+tab+rate`)
improves valid primary further.

## Columns selected

**User-side** (`u_active_degree`, `u_live_streamer`, `u_video_author`, `u_follow_range`,
`u_fans_range`, `u_register_range`) — 6 of `user_features_pure.csv`'s 12 non-onehot columns,
used directly as raw categorical strings (they're already discretized in the source CSV,
e.g. `follow_user_num_range` is a bucket label like `"(100,150]"`). Dropped
`is_lowactive_period` (constant `'0'` across all 27,285 users in this file — zero
information) and `friend_user_num_range` (redundant with follow/fans_user_num_range, kept
feature count reasonable). The 18 anonymized `onehot_feat0..17` columns were left unused —
undocumented/opaque, out of scope for "a reasonable subset of useful-looking columns."

**Video-side** (`v_play`, `v_like`, `v_share`, `v_complete`, `v_follow`) — 5 of
`video_features_statistic_pure.csv`'s ~50 raw counters (`play_cnt`, `like_cnt`, `share_cnt`,
`complete_play_cnt`, `follow_cnt`), the ones most directly related to "is this video
good/popular." Continuous and heavily skewed, so bucketed into 10 quantile buckets fit on
TRAIN split only, via `data.py`'s exact `_bucket_edges` pattern (same as `dur_bucket` and
iter9's `activity`/`tab`/`rate`).

`activity`/`tab`/`rate` causal-feature computation and the strict-`<` date-grouped
traversal are a verbatim copy of `iter9_history_dense/data_ext.py::compute_causal_features`
(re-verified in this copy's own `__main__` spot-checks, including the same-date-pair edge
case — see "Causality verification" below).

## Join / UNK coverage

Checked directly against the log before building the pipeline:

| join | log entities | side-info file rows | log entities missing from side-info |
|---|---|---|---|
| user_id → `user_features_pure.csv` | 27,077 unique users | 27,285 rows | **0** |
| video_id → `video_features_statistic_pure.csv` | 7,551 unique videos | 7,583 rows | **0** |

Both files have full coverage of every user/video that appears in the log — **0/1,436,609
rows (0.000%) fell back to the UNK bucket** for either join (confirmed both by direct
set-difference check and by `data_ext.py`'s own instrumented UNK counters at load time).
The UNK-fallback code path (`UNK_USER_SENTINEL` for users, `None`→`'UNK'` bucket string for
videos) exists and was exercised in the encode-time vocab logic, but was never actually
triggered by real data in this dataset — it's defensive/for robustness rather than load-
bearing here.

Loader correctness check: `base_causal` (this file's re-derivation of iter9's exact
`activity+tab+rate` feature set, computed via an independently copied traversal) reproduces
iter9's activity nonzero coverage (92.29%, matching iter9's RESULT.md exactly) and lands
within iter9's own noise band on both splits (see sweep table below) — confirms this
iteration's data pipeline is consistent with iter9's, so any deltas below are attributable
to the side-info features, not a loader discrepancy.

## Leakage caveat — discussion

As instructed, this is flagged explicitly. `user_features_pure.csv` and
`video_features_statistic_pure.csv` are static per-entity tables with **no date/timestamp
column**, unlike iter9's own history features (which are provably causally clean via
strict `<` date-comparison traversal, verified by brute-force spot-check). The values in
these files almost certainly reflect some fixed aggregation window/corpus snapshot that is
**not guaranteed to respect the train (4/8–4/21) / valid (4/22–4/28) / test (4/29–5/8) date
boundary** — they could include information from the eval period itself, which would be a
target leak for a static "popularity" signal like `like_cnt` or `play_cnt`. This is standard
practice in public KuaiRand baselines (these files are typically treated as static side info
describing inherent user/video characteristics rather than time-varying signals), but it is
explicitly **not** the causally-airtight guarantee iter9's features carry.

Given that risk asymmetry, the design deliberately separated:
- **user-side** (`causal_plus_user`) — mostly slow-moving demographic/account-state fields
  (activity tier, streamer/author flags, follower/fan tier, account age) → **lower** leak risk.
- **video-side** (`causal_plus_video`) — aggregate **engagement counts** (play/like/share/
  complete/follow counts) → **higher** leak risk, since these numbers could directly encode
  "this specific video got a lot of long-views" in a way that trivially correlates with the
  label on the very rows that produced those counts.

**Leakage red-flag check (as instructed — no special-casing of test)**: if the video-side
features were leaking future/target information, we'd expect **test performance to look
suspiciously inflated relative to valid** (memorized/leaked signal often transfers
imperfectly and shows as an unusual valid/test gap, or as test outperforming what the
valid-only signal would predict). We observe the **opposite**: `causal_plus_video`'s test
mean (0.60467) is *lower*, not higher, than `base_causal`'s test mean (0.60572), and drops
in exact same direction as valid (0.60997 vs 0.61024). `causal_plus_both` shows the same
pattern (valid 0.60988, test 0.60498 — both down together). There is no evidence of
leakage-driven test inflation here; if anything, the riskier file's addition net *hurts*
both splits in lockstep, consistent with it simply being unhelpful/noisy input rather than
a leaking shortcut the model is exploiting.

## Sweep (3 seeds: 0,1,2)

| config | fields added | n_fields (total) | valid primary mean | valid std | test primary mean | test std |
|---|---|---|---|---|---|---|
| `base_causal` (iter9 re-derived) | activity,tab,rate | 8 | 0.61024 | 0.00013 | 0.60572 | 0.00010 |
| `causal_plus_user` | + 6 user fields | 14 | 0.61013 | 0.00032 | 0.60517 | 0.00033 |
| `causal_plus_video` | + 5 video fields | 13 | 0.60997 | 0.00014 | 0.60467 | 0.00074 |
| `causal_plus_both` | + all 11 side-info fields | 19 | 0.60988 | 0.00036 | 0.60498 | 0.00021 |

Per-seed detail (all in `results.json`):

| config | seed0 valid | seed1 valid | seed2 valid | seed0 test | seed1 test | seed2 test |
|---|---|---|---|---|---|---|
| base_causal | 0.61028 | 0.61038 | 0.61006 | 0.60562 | 0.60567 | 0.60585 |
| causal_plus_user | 0.60968 | 0.61039 | 0.61032 | 0.60478 | 0.60514 | 0.60559 |
| causal_plus_video | 0.61017 | 0.60986 | 0.60987 | 0.60520 | 0.60363 | 0.60518 |
| causal_plus_both | 0.60977 | 0.61036 | 0.60950 | 0.60468 | 0.60515 | 0.60510 |

`base_causal`'s 0.61024 valid mean matches iter9's original 5-seed 0.61013 within noise
(both files' own std is ~0.0001–0.0003), confirming the re-derivation is correct.

**None of the three side-info additions clears the required +0.001–0.002 bar over
`base_causal`/iter9 — all three are flat-to-slightly-worse** on valid: `causal_plus_user`
is statistically tied with `base_causal` (Δ −0.00011, well inside noise), while
`causal_plus_video` and `causal_plus_both` are consistently *below* it by −0.00027 and
−0.00036 respectively. No config qualifies for a 5-seed extension per the task's stated
criterion, so none was run.

## Verdict — **REJECT, no promotion**

`iter9`'s `activity+tab+rate` feature set (current best, valid 0.61013 5-seed) remains the
best model. Neither the safer user-side static features nor the riskier video-side
engagement-count features, alone or combined, produce a real improvement — all land within
or below iter9's own noise band on valid, and the leakage-caveat red-flag check found no
evidence of test-inflation from the riskier video file (if anything it's the reverse:
video-side features net *hurt* both splits together).

**Likely cause**: similar to iter12's finding for item-side causal features — the FM
already has `user_id`/`video_id`/`author_id` as raw learnable fields, so per-entity static
attributes that are highly correlated with (or literally functions of) those raw IDs (e.g.
a user's demographic tier, a video's aggregate popularity) are largely redundant with what
the embedding table can already learn directly from the ID itself, especially since
`author_id` and `video_id` already implicitly carry most of the "is this good content"
signal that `video_features_statistic_pure.csv`'s counts would add. Unlike iter9's `rate`
feature (a genuinely *new*, time-varying, causally-clean signal with no existing raw field
to carry it), these side-info files add information the model can largely already infer,
so they mostly just add noisy/high-cardinality-adjacent dimensions without new signal —
consistent with `causal_plus_video`'s and `causal_plus_both`'s mild *negative* effect
(more distracting input, no offsetting gain).

## Code
`experiments/iter15_side_info/{data_ext.py,train.py,driver.py}`, results in
`experiments/iter15_side_info/results.json`.

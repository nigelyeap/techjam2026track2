# iter41 — LightGBM LambdaRank ranker (standalone + blend with FM+BPR ensemble)

## Hypothesis
Every prior iteration (iter22–iter40) is a variant of the same FM-family
model: bilinear low-rank factorization trained with BPR. Repeated FM
variants have consistently failed to compose additively when ensembled
(iter34, iter39, iter40's sub-experiments). A structurally different model
family — gradient-boosted trees (LightGBM) with `lambdarank`, an objective
that directly optimizes NDCG via pairwise-swap gradients weighted by
|ΔNDCG|, rather than BPR's plain pairwise-concordance proxy — might make
different errors and add value either standalone or in a score-level blend.

## Method
- Same proven feature set as the FM line (iter27/iter38's
  `decay_rate_2.5, decay_act_2.5, decay_tab_3, last1, lastk_rate, gap`),
  loaded via iter27's `load_ext`/`encode_ext` verbatim (no re-derivation).
- Categorical columns declared to LightGBM as-is (global-offset integer
  encoding works fine — LightGBM treats each column's integers as
  independent categories).
- Ranking API: rows grouped contiguously per user via a stable sort by
  `user_id`, with a `group` array of per-user row counts.
- `LGBMRanker(objective='lambdarank', metric='ndcg', eval_at=[5], ...)`,
  early stopping on valid NDCG@5, default hyperparameters
  (`num_leaves=31, learning_rate=0.05, n_estimators=500`).
- Blend: retrained the exact iter38 5-seed FM+BPR ensemble (same code,
  same feature set, same sigmoid-mean formula) to get raw per-row scores
  in `encode_ext`'s original row order, min-max normalized LightGBM's
  scores to [0,1], and swept a linear blend weight α (FM weight) from 0
  to 1 in steps of 0.1, selecting the best α on valid only.

## Harness-fidelity check
Retrained FM ensemble reproduced iter38 exactly: valid=0.63988,
test=0.64187 (bit-for-bit match to published iter38 numbers). Confirms the
data pipeline and label/user-order alignment between the LightGBM and FM
code paths is correct before trusting the blend result.

## Results

**Standalone LightGBM** (best_iteration=292):
| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.70733 | 0.55654 | 0.63193 |
| test | 0.70110 | 0.55231 | 0.62671 |

Underperforms FM+BPR by -0.008 valid / -0.015 test.

**Blend sweep** (α = weight on FM ensemble, 1-α on LightGBM):
| α | valid primary |
|---|---|
| 0.0 (LGB only) | 0.63193 |
| 0.3 | 0.63446 |
| 0.5 | 0.63618 |
| 0.7 | 0.63794 |
| 0.9 | **0.63997** |
| 1.0 (FM only) | 0.63988 |

Monotonically increasing in α across the whole range — LightGBM only ever
drags the blend down; the "best" point at α=0.9 is FM at 90% weight, i.e.
almost pure FM. Gain over pure FM is +0.00009 valid, far below the 0.001
promotion threshold, and **test at that α is 0.63926 — a regression**
versus FM-only's 0.64187 test.

## Diagnosis
LightGBM's errors are not diverse enough from FM's to add ensembling
value — the monotonic-in-α curve is the signature of one model being
strictly weaker across the whole population, not of two models with
complementary blind spots. This is a clean, harness-verified negative
result: a genuinely different model family and a genuinely different
training objective (NDCG-native pairwise vs. BPR) still doesn't clear the
ensembling threshold here, on top of not beating FM standalone.

## Verdict: REJECT (standalone and blended)

Per explicit instruction to keep searching rather than conclude a
plateau, this closes off the "different model family" lever specifically
— it does not imply other levers are exhausted. Next: richer causal input
features from other engagement signals (`is_like`, `is_follow`,
`is_comment`, `is_forward`, `play_time_ms` ratio) as plain FM input
features, and position/time-gap-aware attention scoring, are still
untried and are the next things to test (see iter42+).

## Addendum: hyperparameter sweep (`sweep.py`)

Before concluding standalone LightGBM was capacity-limited rather than
mistuned, ran a 7-point grid over `num_leaves` / `learning_rate` /
`n_estimators` / `min_child_samples` / `reg_lambda`, still on FM's
bucketed feature set:

| valid | test | config |
|---|---|---|
| **0.63423** | 0.62870 | `num_leaves=15, lr=0.05, n_est=500, min_child=200, reg_lambda=1.0` |
| 0.63193 | 0.62671 | `num_leaves=31, lr=0.05, n_est=500, min_child=50, reg_lambda=0.0` (original default) |
| 0.63096 | 0.62622 | `num_leaves=31, lr=0.1, n_est=400, min_child=50, reg_lambda=0.0` |
| 0.62886 | 0.62396 | `num_leaves=63, lr=0.05, n_est=800, min_child=100, reg_lambda=1.0` |
| 0.62872 | 0.62418 | `num_leaves=63, lr=0.05, n_est=800, min_child=50, reg_lambda=0.0` |
| 0.62431 | 0.61955 | `num_leaves=127, lr=0.03, n_est=1200, min_child=30, reg_lambda=0.0` |
| 0.62035 | 0.61483 | `num_leaves=255, lr=0.02, n_est=2000, min_child=20, reg_lambda=0.0` |

Capacity trend is inverted from the naive expectation: **smaller, more
regularized trees win**, and the largest-capacity config is the worst by a
wide margin (-0.014 valid vs. the best). Tuning recovers +0.0023 valid
over the untuned default, but still lands **0.0047 below FM** (0.63423 vs
0.6389) — this is not a mistuning artifact, it's a real ceiling for this
feature representation. Confirmed as the correct diagnosis by iter44,
which fed the same underlying signals to LightGBM as raw un-bucketed
floats instead of FM's pre-quantized categorical buckets and closed most
of this gap (0.63935 valid) with the same small-tree hyperparameter
regime carried over unchanged — see
`experiments/iter44_gbm_native_features/RESULT.md`.

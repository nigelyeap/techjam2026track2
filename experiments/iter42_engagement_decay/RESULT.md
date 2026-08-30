# iter42 — decayed "other engagement" (like/follow/comment/forward) rate feature

## Hypothesis
Every feature used across every prior iteration derives from `long_view`
(the label itself) or from click/activity history. `is_like`, `is_follow`,
`is_comment`, `is_forward` are explicit behavioral signals never used
anywhere in this project. A decayed rate of "any of these fired on a past
row" (same lazy-decay mechanism as the existing `decay_rate`/`decay_act`
features, at the established winning halflife of 2.5d) might carry
independent predictive signal for future `long_view`.

## Method
- Extended iter27's `_load_raw_time` to also parse `is_like`/`is_follow`/
  `is_comment`/`is_forward` and compute a binary `engaged` flag per row.
- Generalized `compute_decay_features` to accept a `label_col` parameter
  (default 6 = `long_view`, unchanged everywhere else) and reused it
  unmodified with `label_col=engaged` to get a fifth independent causal
  traversal (`decayed_engage_pos`/`decayed_engage_total`), same
  causal-isolation guarantee as the other four traversals in this file.
- New `decay_engage_2.5` feature (ratio, Laplace-smoothed, bucketed —
  identical formula/bucketing to `decay_rate_2.5`) added to iter27/iter38's
  proven feature set: `decay_rate_2.5, decay_act_2.5, decay_tab_3, last1,
  lastk_rate, gap` + `decay_engage_2.5`.
- Same FM+BPR training harness, same hyperparameters, as iter38.

## Harness-fidelity check
Seed 0 on the unmodified iter24 feature set reproduced the published
number exactly: valid=0.63894, test=0.63989. Pipeline confirmed correct
before trusting the new-feature result.

## Result (seed 0)
| config | valid | test |
|---|---|---|
| iter24 feature set (baseline) | 0.63894 | 0.63989 |
| + `decay_engage_2.5` | 0.63820 | 0.63917 |

Delta: **-0.00074 valid** — the new feature makes things slightly worse,
not better. Below the (negative-side) noise floor of a single seed, but
clearly not a positive signal, so per the promotion-threshold discipline
(only worth a 5-seed confirm above +0.0003) this was rejected on seed 0
without spending further compute on a 5-seed run.

## Diagnosis
Likely explanation: `is_like`/`is_follow`/`is_comment`/`is_forward` are
much sparser than `long_view` (long_view fires far more often), so the
decayed engagement-rate feature is mostly zero/near-zero for most rows,
adding a near-constant, low-information categorical bucket that the FM
has to spend embedding capacity on without gaining separating power — a
plausible instance of the same "adding a weakly-informative axis dilutes
the model's effective capacity" pattern documented for other rejected
features in this project (see LEDGER.md).

## Verdict: REJECT

Per the explicit instruction to keep searching for new levers rather than
concluding a plateau, this closes off "engagement signals as-is, single
fixed halflife" specifically. Not yet tried: per-signal features (e.g.
`is_like` alone rather than an OR'd bundle), a different halflife, or
using engagement as a training-time sample weight (analogous to iter23's
`decayed_pos` BPR sampling weight) rather than a per-row input feature.
Given the direct-feature version is a clean negative, further variants on
this exact axis are deprioritized in favor of testing the model-family
axis (iter41 LightGBM/iter43 CatBoost) and the time-gap-aware-attention
axis, which are more differentiated hypotheses.

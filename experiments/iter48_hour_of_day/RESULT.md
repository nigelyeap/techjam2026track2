# iter48 — time-of-day (hour-of-day) as a GBM-native feature

## Motivation

`hourmin` (HHMM, e.g. 1900 = 19:00) has been carried in every row tuple since
iter18 but has only ever been used internally for time-ordering (sorting by
`time_ms`) — never once passed to a model as an input feature across 44+
iterations. Time-of-day is a classic recsys signal and is known at inference
time (intrinsic to the current impression, same causal status as `tab` or
`duration_ms` — not derived from any future information), so this is a
genuinely untried, cheap lever distinct from the decay/recency-window feature
family iter18-44 already explored.

Added two features to iter44's exact GBM-native pipeline: `sin`/`cos` of the
fractional hour (`hour + minute/60`, mapped to `[0, 2π)`), to preserve the
24h wraparound (23:00 and 00:00 are adjacent — a raw linear hour feature
would not capture that). Trained with iter44's exact winning hyperparameters
(`num_leaves=2, learning_rate=0.05, n_estimators=500, min_child_samples=200,
reg_lambda=1.0`) unchanged, only the feature set extended.

## Result

| | valid | test |
|---|---|---|
| Baseline (iter44, no hour feature) | 0.66135 | 0.64794 |
| + hour_sin/hour_cos | 0.66054 | 0.64765 |

Slightly **worse** on both splits (-0.00081 valid), well below the 0.0003
look-threshold — no further seeds run per the promotion protocol. Time-of-day
carries no exploitable signal for `long_view` at this feature set/capacity;
plausibly because whatever session-level pattern exists is already implicit
in the recency/decay features (`gap`, `decay_act_2.5`, `lastk_rate`), which
capture *how recently active* a user is far more directly than *what hour it
is*, or because at `num_leaves=2` (one split per tree) there simply isn't
capacity to spend on a feature this weak alongside six already-strong ones.

**Verdict: REJECT.** Closes off a previously-never-tried lever; no promotable
finding. Final model remains iter44's blend (valid 0.66473 / test 0.65197).

Code: `experiments/iter48_hour_of_day/train.py`.

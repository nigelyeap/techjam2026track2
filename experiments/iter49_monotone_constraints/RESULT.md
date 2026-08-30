# iter49 — monotonic constraints on the engagement-rate features

## Motivation

A structural lever, distinct from every hyperparameter/feature change tried
in iter44-48: constrain `decay_rate_2.5`, `decay_act_2.5`, `decay_tab_3`, and
`lastk_rate` (all recency-decayed positive-engagement rates/counts) to a
monotonically non-decreasing relationship with the predicted score, via
LightGBM's `monotone_constraints`. Domain rationale: more/recenter positive
engagement should never *decrease* predicted long-view likelihood — and this
seemed like it could matter specifically at `num_leaves=2` (iter44's winning
capacity), where a single spurious split found by chance has nothing to be
averaged out by. `duration_ms` and `gap` left unconstrained (no clear-cut
expected direction — see iter44's `duration_ms` finding and `gap`'s
plausible U-shape). Otherwise iter44's exact pipeline/hyperparameters,
unchanged.

## Result

| | valid | test |
|---|---|---|
| Baseline (iter44, unconstrained) | 0.66135 | 0.64794 |
| + monotone_constraints | **0.59156** | 0.58511 |

A large, decisive regression (-0.070 valid) — not a marginal effect like
iter45/46/48's small negatives. **Verdict: REJECT, clearly and by a wide
margin — no seeds run.**

Plausible mechanism (not chased further given the size and clarity of the
result): LightGBM's constrained split-finding algorithm restricts valid
split *sequences* down a branch, not just individual per-feature split
directions in isolation — with only one split per tree (`num_leaves=2`)
and five categorical columns sharing the same trees as the four
constrained numeric columns, the constraint solver has very little room to
find a split that satisfies the monotonicity requirement *and* is actually
informative, so most trees likely fall back to weaker splits on the
constrained features or route decision-making almost entirely through the
unconstrained categorical columns instead — a much worse trade at this
capacity than the mild regularization benefit hoped for. Monotonic
constraints may still be worth reconsidering at a *higher* capacity
(`num_leaves=7`, iter44's safer fallback) where trees have more room to
satisfy the constraint without sacrificing split quality, but not chased
here given the clarity of this result and the token-conservation
instruction.

**Verdict: REJECT.** Final model unchanged (iter44's blend, valid 0.66473 /
test 0.65197).

Code: `experiments/iter49_monotone_constraints/train.py`.

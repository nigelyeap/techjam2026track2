# iter70 — explicit interaction terms between the two proven decayed-rate features

## Provenance

Continuation of self-directed research per *"find more methods, or maybe combine
methods that work with our current existing model."* iter63's `rate_only` feature
set contains two independently-proven decayed-rate features — `decay_rate_2.5`
(global per-user recency rate) and `decay_tab_rate_3` (per-(user,tab) recency
rate) — that have never been combined via an explicit cross term.

## Hypothesis

`linear_tree=True`'s leaf regression is linear in its input features, so it cannot
represent a genuine multiplicative interaction between two numerics on its own —
only their independent linear contributions. An explicit product/ratio/diff term
gives the leaf model that capacity directly, the same way an explicit interaction
column would help any linear model.

## Implementation

`experiments/iter70_decay_rate_interaction/train.py`, built directly on iter63's
`rate_only` feature set and hyperparameters (unchanged). No new data extraction
needed — both base columns already exist in iter63's `prepare()` output. Three
interaction terms added independently and combined:
- `x_rate_product = decay_rate_2.5 * decay_tab_rate_3`
- `x_rate_ratio = (decay_rate_2.5 + 0.01) / (decay_tab_rate_3 + 0.01)`
- `x_rate_diff = decay_rate_2.5 - decay_tab_rate_3`

Harness-fidelity check reproduces iter63's exact baseline before trusting any new
number.

## Result (seed 0)

| variant | valid | test | Δvalid | Δtest |
|---|---|---|---|---|
| baseline | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| product | 0.66983 | 0.65297 | −0.00185 | −0.00055 |
| ratio | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| diff | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| all3 | 0.66983 | 0.65297 | −0.00185 | −0.00055 |

`diff` is an exact no-op — expected, since it's a linear combination of two
features already in the set, and the leaf-linear model can already represent
any linear combination without an explicit column for it. `ratio` is also an
exact no-op. `product` (the genuinely non-linear term) causes a small but
real regression, and `all3` matches `product` exactly, confirming `product` is
the sole active ingredient among the three and it's net negative, not neutral.

## Verdict: REJECT

`product` is a small regression (−0.00185 valid), modest relative to iter68's
findings but directionally consistent (adding redundant/noisy signal on top of
an already-well-tuned shallow booster). `ratio` and `diff` are inert by
construction under a linear leaf model. No multi-seed confirmation run — this is
a clean negative, not a borderline or surprising one, per established project
practice.

iter63 remains the current best and correctly promoted candidate. Explicit
interaction terms between the two decayed-rate features do not help.

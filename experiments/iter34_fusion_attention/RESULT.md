# iter34 — fuse iter27 (triple fusion) with iter32 (target-attention feature)

## Idea
iter27 (features + decay-aware sampling α=0.75 + Laplace α=0.5/n_buckets=20)
and iter32 (+`attn_rate_40` target-attention feature) are the two non-
overlapping Round 9 wins over iter24, never combined. Tests whether they
compose additively like iter24+iter23+iter25 did.

## Method
`data_ext.py` merges iter27's feature/sampling pipeline with iter32's
5-traversal attention-feature computation (target attention over the user's
most recent W causal interactions, pooled by softmax similarity to a
pretrained k=8 FM item embedding fit on train only); `train.py` is an
unmodified copy of iter27's (no changes needed, same `load_ext`/`encode_ext`
signatures). Causality: no new mechanism — attention feature already
verified in iter32, sampling weight/formula constants already verified in
iter22/23/25/27; a cross-family check in `data_ext.py`'s self-test confirms
the per-user sampling weight (dict keyed by user_id) and the per-row
attention feature (time_ms-aware column) operate on disjoint data
structures and cannot leak into each other.

## Harness-fidelity check
5 seeds, attention feature excluded from feature list: bit-exact match to
iter27's published `fusion_sampling_alpha0.75` numbers (max |Δ|=0.0 on both
valid and test, all 5 seeds).

## Sweep (3 seeds, valid-only selection)
Added `attn_rate_40` to iter27's winning fused config (sampling_alpha=0.75,
alpha=0.5, n_buckets=20):

| seed | valid | test |
|---|---|---|
| 0 | 0.64002 | 0.64035 |
| 1 | 0.63848 | 0.63827 |
| 2 | 0.63713 | 0.63779 |
| **mean** | **0.63855** | 0.63880 |

vs iter27's matched 3-seed valid mean (0.63816): **Δ=+0.00039** — real
direction but below the 0.001 promotion threshold. Not extended to 5 seeds
per protocol (only candidates clearing the threshold get a 5-seed
confirmation).

## Diagnosis
The two mechanisms do not compose additively the way iter24+iter23+iter25
did. Plausible reason: iter27's decay-aware sampling weight already
upweights users with more/recent train activity, and iter32's target
attention draws its signal from the same recent-interaction history — the
two levers likely have overlapping information content (both are
"recent-activity-aware" mechanisms), unlike iter24/iter23/iter25's fully
disjoint feature/sampling/formula-constant axes. This narrows why iter27's
fusion generalized so well while this second fusion attempt shows sharply
diminishing returns.

## Verdict: REJECT (does not clear the 0.001 valid-margin promotion bar)
iter27 remains current best. Independently verified: harness-fidelity rows
and the 3-seed sweep mean were hand-computed from `results.json` and match
the driver's printed output exactly.

## Code
`experiments/iter34_fusion_attention/{data_ext.py,train.py,driver.py}`,
raw results in `experiments/iter34_fusion_attention/results.json`.

# iter77 — decayed (user, tab, duration-bucket) cross rate

## Provenance

Not another instance of Round 22's closed "decay this new axis alone"
family. Tests whether crossing the one proven decayed-rate axis (`tab`,
iter16/iter63) with the best-explored secondary axis (duration-bucket,
null alone in iter72) reveals interaction signal that neither axis captures
independently — distinct from iter70's already-rejected product term over
two already-computed *rate* columns, since this crosses the raw counts
*before* computing a rate, not the rates themselves. E.g. "this user's
engagement rate on short videos specifically in the follow feed" could
differ from "this user's rate in the follow feed generally" if
duration-preference is itself feed-dependent.

## Implementation

`experiments/iter77_decay_tab_durbucket_cross/data_ext.py`: reuses the
shared `_compute_decay_key_features` mechanism keyed on
`(user_id, tab, duration_bucket(duration_ms))` jointly — 77 distinct
(tab, dur_bucket) cells in train, several very sparse (smallest cells have
counts 1-3), handled by the existing Laplace smoothing (α=0.5). Causality
verified: 30 random rows × 2 halflives, brute-force recompute matches
exactly (max abs err 4.44e-15); monotonicity holds over 5000 rows.
`train.py` adds `decay_cross_rate_{3,7} = (decayed_cross_pos + α) /
(decayed_cross_total + 2α)` on top of the unchanged `rate_only` set (which
already includes `decay_tab_rate_3`). Harness-fidelity check reproduces
iter63's exact baseline before trusting any new number.

## Result (seed 0)

| variant | valid | test | Δvalid | Δtest |
|---|---|---|---|---|
| baseline (rate_only) | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| cross_rate_h3 | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| cross_rate_h7 | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| cross_rate_both | 0.67168 | 0.65353 | +0.00000 | +0.00000 |

Exact-zero delta across all three configurations.

## Diagnosis

Unlike the categorical-addition nulls (iter68/75), this is a *numeric*
feature fed directly into `linear_tree`'s per-leaf linear regression, so the
"never gets picked as a split variable" explanation used for those doesn't
directly apply here — the leaf regression sees this column as a candidate
input regardless of tree structure. The more likely explanation: with 77
sparse cells (many with only 1-3 observations before smoothing), most
`decay_cross_rate` values collapse toward the α=0.5 smoothing prior and are
highly correlated with the already-present `decay_tab_rate_3` (the
per-tab-only rate, which pools far more data per cell) — the leaf-linear
model can already represent any residual linear contribution via the
existing `decay_tab_rate_3` column, so a near-collinear, noisier version
adds nothing (and the L2 term drives its effective coefficient to
~0). This is consistent with iter70's finding that the linear-leaf model
doesn't need dedicated interaction terms when it can already express linear
combinations of its existing inputs.

## Verdict: REJECT (clean no-op)

No 5-seed confirmation needed — exact-zero is unambiguous. iter63 remains
the current best. Combined with iter70 (rate×rate product) and the entire
closed decayed-rate-generalization family (iter68-73), this closes off
"decayed-rate interactions/crosses" as a productive direction for this
model/feature combination: neither crossing keys before rate computation
(this iteration) nor combining rates after computation (iter70) adds value.

# iter66 — Rank-based / calibrated blending — independent re-verification

## Provenance

Reported by teammate "Xuxia" on a separate clone (`XUXIA_SUMMARY.md`
section 6b; see `experiments/LEDGER.md`'s "Parallel track" section).
Independently re-implemented and re-checked on this clone per direct user
instruction.

## Hypothesis

The current blend uses linear combination of a min-max-normalized GBM
score and the FM ensemble's sigmoid-mean score. Two alternative fusion
strategies were tested: (a) rank-based fusion (Borda count, reciprocal-rank
fusion), which discards raw score magnitude and might be more robust to
each model's differing score distribution; (b) isotonic-regression
calibration of the GBM's raw score (fit on train), which might correct for
the GBM's score not being a well-calibrated probability.

## Implementation

Reused `experiments/iter65_segment_blend/scores_5seed.npz` (5 GBM seeds +
unchanged FM ensemble; harness-fidelity already confirmed there). Rank
fusion computed **per user** (pandas `groupby(u).rank()`), matching the
natural "query" grouping for this within-user ranking task: Borda score
`-(rank_gbm + rank_fm)`, RRF score `1/(60+rank_gbm) + 1/(60+rank_fm)`.
Isotonic calibration (`run_isotonic.py`) retrained the seed-0 GBM to get
train-split raw scores (not saved by the shared generator), fit
`sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` on
`(gbm_train_raw, y_train)`, applied to valid/test, then substituted the
calibrated score for the min-max-normalized score in the same alpha=0.14
blend formula.

## Result

### Rank fusion (5 GBM seeds)

| | mean Δvalid | wins | mean Δtest |
|---|---|---|---|
| Borda | **-0.00991** | 0/5 | -0.00229 |
| RRF | **-0.00996** | 0/5 | -0.00215 |

Both regress in every seed. Magnitudes are larger than Xuxia's reported
per-method deltas (Borda -0.00146, RRF -0.00133) — plausibly because this
implementation ranks per-user (their exact grouping wasn't specified) which
throws away more magnitude information than a global rank fusion would in
a task this reliant on within-user score separation — but the qualitative
conclusion (clean, consistent REJECT, ranks lose information the linear
blend uses) matches exactly.

### Isotonic calibration (seed 0)

| | valid | test |
|---|---|---|
| GBM standalone, raw (minmax) | 0.67168 | 0.65353 |
| GBM standalone, isotonic-calibrated | **0.54189** | 0.53201 |
| Current blend (minmax) | 0.67606 | 0.65955 |
| Isotonic-calibrated blend | 0.63877 | 0.64039 |

**Reproduced exactly**: GBM standalone raw valid=0.67168 (matches iter63's
own documented number to 5 decimals), isotonic-calibrated collapse to
**0.54189** valid — identical to Xuxia's reported figure — with the same
root cause independently confirmed: isotonic regression pools the GBM's
122,613 unique valid-split raw scores into only **37** distinct calibrated
levels (Xuxia also reported 37), a monotonic-step-function tie-artifact
that destroys the GBM's fine-grained within-user ranking ability. This is
about as strong a same-mechanism confirmation as is possible without
literally sharing a random seed stream — a structural incompatibility
between lambdarank-style raw scores and isotonic calibration, not a
tuning issue or an artifact of one environment.

## Verdict: **REJECT** (all three: Borda, RRF, isotonic — independently confirmed)

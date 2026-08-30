# iter55 — learning_rate sweep under linear_tree=True

## Motivation

Unlike iter52/53/54 (retests of already-rejected knobs), this is a
genuinely new hypothesis: `linear_tree=True` changes what each boosting
round buys the model (a per-leaf linear fit instead of a flat constant),
so `learning_rate=0.05` — tuned back in iter44 against the old
constant-leaf tree — was never re-validated against this structural
change.

## Method

Single-axis sweep of `learning_rate` over
`{0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20}`, seed=0, all other
hyperparameters unchanged from iter51 (`num_leaves=2, linear_tree=True`).

## Result — sweep (seed=0)

| learning_rate | valid | test |
|---|---|---|
| 0.01 | 0.64402 | 0.63199 |
| 0.02 | 0.64417 | 0.63246 |
| 0.03 | 0.64380 | 0.63233 |
| 0.05 (iter51 baseline) | 0.66932 | 0.65146 |
| 0.07 | 0.63990 | 0.62501 |
| **0.10** | **0.67052** | **0.65277** |
| 0.15 | 0.66671 | 0.65066 |
| 0.20 | 0.66504 | 0.65103 |

`learning_rate=0.10` beats the baseline by +0.00120 valid on the first
run, clearing the 0.0003 look-threshold. Note the sweep is highly
non-monotonic (0.07 collapses to 0.639 between two much better points at
0.05 and 0.10) — early-stopping interacting with the per-leaf linear fit
appears to create a narrow, jagged landscape rather than a smooth one at
this ultra-low-capacity regime.

## Result — 5-seed confirmation at learning_rate=0.10

| seed | valid | test |
|---|---|---|
| 0 | 0.67052 | 0.65277 |
| 1 | 0.67008 | 0.65221 |
| 2 | 0.66993 | 0.65217 |
| 3 | 0.66993 | 0.65217 |
| 4 | 0.67008 | 0.65221 |
| **mean** | **0.67011** (std 0.00021) | **0.65230** (std 0.00023) |

vs. iter51's own 5-seed baseline (mean valid=0.66926, mean test=0.65140):
**+0.00085 valid, +0.00090 test**, 5/5 seeds improving, tight seed
variance (std 0.00021, i.e. the gain is ~4x the seed-to-seed noise). This
falls just under the self-imposed 0.001 "writeup-worthy" round-number
bar, but the consistency across all 5 seeds and the gain/std ratio make
it a credible small real effect rather than noise. **Verdict: PROMOTE
(standalone, marginal)** — worth testing in the submission-level blend.

## Result — blend with the unchanged FM 5-seed ensemble

Reblending this GBM (seed=0, `learning_rate=0.10`) with the unchanged FM
ensemble via the same alpha-sweep pattern as iter44/iter51 found a new
best alpha=0.10:

| | alpha | valid | test |
|---|---|---|---|
| iter51 blend (currently submitted) | 0.08 | 0.67297 | 0.65643 |
| **iter55 blend** | **0.10** | **0.67451** | **0.65832** |

**+0.00154 valid, +0.00189 test** over the currently-submitted iter51
blend — a larger gain at the blend level than the standalone GBM gain
(+0.00085/+0.00090), i.e. the small learning_rate change also produced a
GBM whose errors compose slightly better with the FM ensemble's, not just
a marginally better GBM in isolation. **Verdict: PROMOTE (blend)** —
flagged to the user for an explicit go-ahead before touching
`SUBMISSION.md`/`make_submission.py`/`submission.csv`, per the standing
protocol, not promoted unilaterally.

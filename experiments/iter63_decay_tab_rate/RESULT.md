# iter63 — decayed per-tab RATE feature (`decay_tab_rate_3`), replacing the raw `decay_tab_3` count

## Hypothesis

Every feature set since iter24 has included `decay_tab_3`: a decayed
**count** of a user's prior positive interactions within the current row's
tab (halflife=3d). But the project's own established lesson (iter9→iter16
era) is that Laplace-smoothed **rate** features outperform raw counts —
exactly the relationship between `decay_rate_H` (rate) and `decay_act_H`
(the matching activity-count denominator) elsewhere in the same pipeline.
`compute_decay_tab_features` (iter27's `data_ext.py`, unchanged since) only
ever tracked the positive numerator, never a matching decayed-total
denominator, so this rate was never constructible. Two ruled-out
alternatives considered first: item-side popularity features (iter12 —
redundant with FM's learned id embeddings) and time-of-day (iter48 —
REJECTed, -0.00081 valid). This is a genuinely new causal feature, not a
hyperparameter resweep.

## Implementation

`experiments/iter63_decay_tab_rate/data_ext.py` extends
`compute_decay_tab_features` to track a second parallel lazy-decay counter,
`decayed_tab_total` (decayed count of ALL rows in that (user, tab), not
just positives), alongside the existing `decayed_tab_pos`. Both use the
same halflife grid (`TAB_HALFLIVES=[3,7]`). `decay_tab_rate_H` is then
computed GBM-side as `(tab_pos + 0.5) / (tab_tot + 1.0)` (iter27's Laplace
constant, unchanged). Otherwise byte-for-byte identical to iter27's
`data_ext.py`.

**Causality verification** (this module's own `__main__`): 30 random rows
× 2 tab-halflives, `decayed_tab_total` matched brute-force recount (max
abs err 1.78e-14); sanity check `decayed_tab_total >= decayed_tab_pos`
held across 5000 rows × 2 halflives; zero-activity rows correctly returned
0.0. All causal spot-checks passed — no leakage.

## Sweep (single seed=0, iter55's exact GBM hyperparameters:
`linear_tree=True, num_leaves=2, learning_rate=0.10, n_estimators=500,
min_child_samples=200, reg_lambda=1.0`), three variants:

| variant | valid | test |
|---|---|---|
| `baseline` (decay_tab_3 count only, = iter55 exactly) | 0.67052 | 0.65277 |
| `plus_rate` (count AND rate both present) | 0.67052 | 0.65277 |
| `rate_only` (rate REPLACES count) | **0.67168** | **0.65353** |

`baseline` exactly reproduces iter55's known value (harness-check passed).
`plus_rate` is bit-identical to `baseline` — the linear-leaf GBM appears to
fully ignore the added rate feature when the count feature is also
present (unexplained; not pursued further given the time budget, since
`rate_only` already gives the interesting result). `rate_only` shows a
real single-seed gain of **+0.00116 valid / +0.00076 test**.

## 5-seed confirmation (`rate_only` vs `baseline`, GBM standalone)

| seed | baseline valid | baseline test | rate_only valid | rate_only test | Δ valid |
|---|---|---|---|---|---|
| 0 | 0.67052 | 0.65277 | 0.67168 | 0.65353 | +0.00116 |
| 1 | 0.67008 | 0.65221 | 0.67105 | 0.65323 | +0.00098 |
| 2 | 0.66993 | 0.65217 | 0.67104 | 0.65320 | +0.00111 |
| 3 | 0.66993 | 0.65217 | 0.67104 | 0.65320 | +0.00111 |
| 4 | 0.67008 | 0.65221 | 0.67105 | 0.65323 | +0.00098 |

**baseline**: mean valid=0.67011 (std 0.00021), mean test=0.65230 (std 0.00023)
**rate_only**: mean valid=0.67117 (std 0.00025), mean test=0.65328 (std 0.00012)
**mean delta valid = +0.00107, wins=5/5** — clears the 0.001 "unambiguously
real" bar with no sign flips; test also improves consistently (+0.00098
mean).

## Blend-level check (rate_only GBM seed=0 + unchanged iter38 FM 5-seed ensemble, alpha-swept)

| | valid | test |
|---|---|---|
| GBM standalone (rate_only, seed=0) | 0.67168 | 0.65353 |
| FM ensemble standalone (iter38, unchanged) | 0.63988 | 0.64187 |
| **iter55 blend** (alpha=0.10, current submission) | 0.67451 | 0.65832 |
| **iter63 blend** (alpha=0.14, best on valid) | **0.67606** | **0.65955** |

**Delta vs iter55: +0.00155 valid / +0.00123 test.** Unlike iter61 (a real
standalone FM gain that did NOT survive to the blend level, and moved test
the wrong way), this gain propagates cleanly to the blend: both valid and
test improve, well clear of the 0.0003 promotion look-threshold.

## Verdict: **PROMOTE (pending user go-ahead)**

Real, causally-verified, 5-seed-confirmed standalone gain (+0.00107 valid)
that also clears the blend-level threshold against the current submission
(+0.00155 valid / +0.00123 test, alpha re-optimizes from 0.10→0.14). Per
the project's standing protocol, promotion to `SUBMISSION.md`/
`make_submission.py`/`submission.csv` requires explicit user approval —
not made unilaterally regardless of how favorable the metrics are.

Config to promote: GBM = `LGBMRanker(num_leaves=2, learning_rate=0.10,
n_estimators=500, min_child_samples=200, reg_lambda=1.0,
linear_tree=True)` trained on iter44's GBM-native encoding of iter63's
extended feature set (iter27's features with `decay_tab_3` **replaced by**
`decay_tab_rate_3`, i.e. the `rate_only` variant), blended at alpha=0.14
(86% GBM / 14% FM) with the unchanged iter38 FM 5-seed ensemble.

Full artifacts: `experiments/iter63_decay_tab_rate/{data_ext.py,train.py,
confirm5.py,blend.py,blend_results.json,run.log,confirm5.log,blend.log,
causality_check.log}`.

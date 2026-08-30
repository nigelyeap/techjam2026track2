# iter54 — retest iter48's hour-of-day feature under linear_tree=True

## Motivation

iter48 (hour_sin/hour_cos added to iter44's GBM-native feature set) was a
clean REJECT against the OLD constant-leaf GBM. But `linear_tree=True`
means each leaf fits its own linear regression over all NUM_COLS, not
just the split feature — so a feature that adds nothing to a
piecewise-constant tree could in principle still contribute a nonzero
per-leaf linear coefficient. Re-tests the identical feature addition
under `linear_tree=True` and iter51's winning hyperparameters.

## Method

Identical to iter48 (same `_row_to_dict`, same `CAT_COLS`/`NUM_COLS`
including `hour_sin`/`hour_cos`), with `linear_tree=True` added to the
`LGBMRanker` config, otherwise iter51's exact hyperparameters.

## Result

| | valid | test |
|---|---|---|
| iter51 baseline (no hour feature) | 0.66932 | 0.65146 |
| iter54 (+ hour_sin/hour_cos, linear_tree=True) | 0.66932 | 0.65146 |

**Bit-identical to the baseline** (same `best_iteration=72`, same metrics
to 5 decimal places) — the model's per-leaf linear regression assigned
the hour features exactly zero effective weight, confirming iter48's
original finding rather than reversing it: hour-of-day carries no signal
at this feature set, under either tree type. **Verdict: REJECT.**

# iterXUXIA2 — rank-based / calibrated blending vs. the current linear+minmax blend

## Hypothesis
The current blend does `alpha*FM_score + (1-alpha)*minmax(GBM_score)` — a
linear combination of min-max-normalized raw scores, which assumes the two
models' score distributions are comparably shaped after min-max
normalization. Two alternatives might sidestep that assumption: (1)
rank-based fusion (Borda / reciprocal-rank), which only uses each model's
within-user ranking, not its raw score scale; (2) isotonic-regression
calibration, mapping each model's raw scores to a calibrated
`P(long_view)` before blending.

## Baseline note
Same as iterXUXIA1: this clone's `HEAD` is at iter63 (19 commits past
`XUXIA_INSTRUCTIONS.md`'s iter44 reference), so **iter63's numbers
(valid=0.67606/test=0.65955, alpha=0.14) are the baseline to beat**, per
user instruction.

## Method
- **Rank-average (Borda) fusion**: per-user average rank (ascending,
  ties averaged, scaled to (0,1]) for each model, blended as
  `beta*rank_FM + (1-beta)*rank_GBM`, beta swept 0.00-1.00 (step 0.02).
- **Reciprocal-rank fusion (RRF)**: per-user descending rank (1=best) →
  `1/(K+rank)` per model, swept over `K∈{10,30,60,100}` and the same beta
  grid; best (K, beta) selected on valid.
- **Isotonic calibration**: `IsotonicRegression(y_min=0, y_max=1)` fit on
  **train only** (`gbm_tr_raw`→`ytr`, `fm_tr_ens`→`ytr`, no valid/test
  leakage), applied to valid/test, blended the same way.
- All three compared against the current linear+minmax blend, across the
  same 5 GBM seeds as iterXUXIA1 (FM ensemble fixed).
- (Implementation note: the first version of the rank-fusion code used a
  pure-Python per-user loop and was still running after >6 CPU-minutes on
  a single seed — killed and rewritten with vectorized `pandas.groupby`
  rank computation, which finished the full 5-seed sweep in <6 minutes
  total. No numerical difference in method, just a performance bug in the
  first draft that was caught before trusting any number from it.)

## Harness-fidelity check
```
GBM standalone: valid=0.67168 test=0.65353
FM ensemble standalone: valid=0.63988 test=0.64187
global-alpha blend (alpha=0.14): valid=0.67606 test=0.65955
```

## Result (5 GBM seeds, FM ensemble fixed)
```
--- GBM seed 0 ---
  linear+minmax (current): beta=0.14 valid=0.67606 test=0.65955
  rank-borda:              beta=0.32 valid=0.67476 (delta=-0.00131) test=0.65949
  rank-rrf (k=30):         beta=0.32 valid=0.67491 (delta=-0.00115) test=0.65951
  isotonic-calibrated:     beta=1.00 valid=0.63983 (delta=-0.03623) test=0.64082
[seeds 1-4 show the same pattern; full per-seed numbers in calibrated_blend_results.json]

=== 5-seed summary ===
rank_borda  mean=-0.00146  min=-0.00164  seeds>=+0.001: 0/5
rank_rrf    mean=-0.00133  min=-0.00147  seeds>=+0.001: 0/5
isotonic    mean=-0.03584  min=-0.03623  seeds>=+0.001: 0/5
```

## Diagnosis
**Rank fusion** (Borda and RRF) both land a few thousandths below the
current blend in every seed — a real but small regression, not noise
(consistent sign and magnitude across all 5 seeds). Converting to
within-user ranks throws away magnitude information the linear blend
still exploits (e.g. *how much* more confident the GBM is about the top
candidate vs. the second), so it makes sense this loses a little ground
rather than gaining any — min-max-normalized raw scores were evidently not
a "crude" enough assumption to matter here.

**Isotonic calibration** catastrophically underperforms (beta=1.00 was
selected, i.e. the sweep found *any* nonzero weight on calibrated GBM only
hurts — degenerating to calibrated-FM-alone). Ran a follow-up diagnostic
(`diag_isotonic.py`) to find out why:
```
raw GBM (minmax) standalone valid:         0.67168   (122,612 unique values / 124,909 rows)
isotonic-calibrated GBM standalone valid:  0.54189   (37 unique values -- 80 PAV plateaus)
raw FM ensemble standalone valid:          0.63988   (117,422 unique values / 124,909 rows)
isotonic-calibrated FM standalone valid:   0.63983   (278 unique values -- 350 PAV plateaus)
```
Isotonic regression pools GBM's near-continuous raw score (122k unique
values, from `linear_tree=True`) into only **37** distinct calibrated
levels — a catastrophic tie collapse that alone drags standalone valid
from 0.67168 down to 0.54189, barely above the trivial constant-score
floor (~0.483). FM's calibration is far gentler (350 plateaus, valid
essentially unchanged) because FM's sigmoid output already lives on a
bounded, probability-like scale that isotonic's PAV algorithm doesn't need
to compress much. GBM's raw lambdarank score has no such
probability-scale structure, so PAV — which only guarantees monotonicity,
not resolution — collapses it hard when fit against a single global
long_view base rate on train. This is exactly the tie-artifact-awareness
concern Section 3 flags, just manifesting as an unusually *bad* result
instead of a suspiciously good one: a model that goes from ~123k unique
scores to 37 has lost almost all of its ranking resolution.

## Verdict: REJECT (both alternatives)

Neither rank-based fusion nor isotonic calibration beats the current
linear+minmax blend in any of 5 seeds; both regress. This is a clean,
useful null result, not a failure to find something: it confirms the
current crude-looking linear+minmax approach is already close to what a
fusion strategy can extract from these two models' scores, and identifies
*why* the most theoretically appealing alternative (isotonic calibration)
actively breaks the GBM side specifically (raw-score tie collapse under
PAV), which would otherwise be a tempting thing to retry with "better
tuning" — it isn't a tuning problem, it's a structural mismatch between
lambdarank-style raw scores and isotonic calibration. No further work on
this axis is recommended; the existing linear+minmax blend (alpha=0.14)
stands.

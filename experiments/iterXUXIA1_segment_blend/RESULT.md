# iterXUXIA1 — per-segment blend alpha (by `tab` and by activity tertile)

## Hypothesis
The current blend uses one global `ALPHA_BLEND` for every row (0.14 as of
iter63, 86% GBM / 14% FM). It's plausible the optimal GBM/FM mix differs by
segment — e.g. by `tab`, or by user activity tier (light vs. heavy users,
already distinguished by the decayed-activity feature `decay_act_2.5`).

## Note on baseline version
`XUXIA_INSTRUCTIONS.md` (Section 5) states the harness-fidelity reference as
`iter44 blend: valid=0.66473 test=0.65197`. Reproducing `make_submission.py`
on this clone's actual `HEAD` (`a7d6e3b` + one further commit) instead gives
`iter63 blend: valid=0.67606 test=0.65955` — `main` had advanced 19 commits
past iter44 by the time this work started (iter45-iter63 promotions:
`linear_tree=True`, `learning_rate=0.10`, decayed per-tab rate feature,
alpha retuned 0.1→0.14). The FM ensemble standalone number
(valid=0.63988/test=0.64187) reproduced exactly, confirming the environment
itself was not the problem. Per user instruction, **iter63's numbers
(valid=0.67606 / test=0.65955, alpha=0.14) are used as the baseline to beat
throughout this file**, not iter44's.

## Method
- Reused iter63's exact winning GBM (`rate_only` variant, `linear_tree=True`,
  `lr=0.10`, `num_leaves=2`) and iter38's unchanged 5-seed FM ensemble.
- **Tab segmentation**: 15 distinct `tab` values present in valid. Swept
  alpha independently per tab on a 41-point grid (0.00-0.40, step 0.01),
  selecting the alpha maximizing `evaluate()` restricted to that tab's own
  valid rows. Applied each tab's chosen alpha to the matching test rows
  (same tab value), then scored the *entire* valid/test set with the
  project's standard full-set `evaluate()` (segmentation only changes which
  alpha a row receives — not how the metric itself is computed).
- **Activity-tier segmentation**: tertile-split `decay_act_2.5` using edges
  fit on valid only (`[3.384, 7.043]`, no leakage into test), giving 3
  balanced tiers (~41.6k valid rows each). Same per-segment sweep/apply/
  evaluate procedure as tabs.
- Repeated the entire procedure across 5 GBM seeds (0-4), FM ensemble held
  fixed (project convention for GBM-only resweeps, e.g. iter55-63's
  `blend.py`), to guard against the extra free parameters (15 tab alphas or
  3 tier alphas vs. 1 global alpha) fitting valid-set noise.

## Harness-fidelity check
`harness_check.py`, using this experiment's own cached GBM+FM code path,
reproduced iter63's exact numbers before any segmentation logic was trusted:
```
GBM standalone: valid=0.67168 test=0.65353
FM ensemble standalone: valid=0.63988 test=0.64187
global-alpha blend (alpha=0.14): valid=0.67606 test=0.65955
```

## Segment-size sanity check (seed 0; sizes are seed-independent)
Tab sizes are highly imbalanced — one tab dominates, several are tiny:
```
tab=1:  92672 valid rows / 20119 users  -> alpha=0.13 (stable, ~matches global)
tab=0:  13726 valid rows / 5579 users   -> alpha=0.00
tab=4:   7877 valid rows / 4209 users   -> alpha=0.02
tab=6:   5170 valid rows / 2212 users   -> alpha=0.15
tab=2:   3834 valid rows / 1126 users   -> alpha=0.15
tab=8:    547 valid rows / 340 users    -> alpha=0.00
tab=5:    291 valid rows / 129 users    -> alpha=0.40 (grid ceiling — noise)
tab=12:   226 valid rows / 112 users    -> alpha=0.01
tab=3:    177 valid rows / 162 users    -> alpha=0.00
tab=11:   121 valid rows / 63 users     -> alpha=0.01
tab=9-14:  <100 rows each               -> alpha=0.00 (degenerate)
```
Tier sizes are large and balanced (~41.6k rows / 5.4k-15k users each) —
these are not a small-sample problem.

## Result (5 GBM seeds, FM ensemble fixed)
| GBM seed | global alpha | global valid | tab-seg valid (Δ) | tier-seg valid (Δ) |
|---|---|---|---|---|
| 0 | 0.13 | 0.67612 | 0.66733 (-0.00879) | 0.67093 (-0.00519) |
| 1 | 0.14 | 0.67583 | 0.66743 (-0.00840) | 0.67102 (-0.00480) |
| 2 | 0.13 | 0.67540 | 0.67572 (+0.00031) | 0.66859 (-0.00681) |
| 3 | 0.13 | 0.67540 | 0.67572 (+0.00031) | 0.66859 (-0.00681) |
| 4 | 0.14 | 0.67583 | 0.66743 (-0.00840) | 0.67102 (-0.00480) |

Tab-segment: mean Δ = **-0.00499 valid**, 0/5 seeds ≥ +0.001 (best case only
+0.00031, still below the promotion floor). Tier-segment: mean Δ =
**-0.00568 valid**, 0/5 seeds ≥ +0.001. Full data in
`segment_sweep_results.json`.

## Diagnosis
Both segmentations **regress** relative to the single global alpha in every
seed:
- **Tab**: the regression traces directly to the small-segment overfitting
  the instructions flagged in advance — tabs 5, 9-14 hold under 300 rows and
  land on degenerate grid-boundary alphas (0.00 or 0.40) that are clearly
  fit to per-tab noise, not real GBM/FM heterogeneity. The one large,
  stable tab (tab=1, 92.7k rows) lands on alpha≈0.13, essentially the same
  as the global optimum — consistent with there being no real per-tab
  signal to exploit even where sample size isn't the limiting factor.
- **Tier**: this is the more informative result, since all three tiers are
  large and balanced (~41.6k rows), ruling out a small-sample explanation.
  It still regresses in every seed. The per-tier alpha chosen by sweeping
  the segment-restricted metric doesn't transfer well to the full-set
  metric — segment-local `evaluate()` changes each user's exposure set
  (only that user's rows in the given tier are considered), so the
  "locally optimal" alpha is optimizing a subtly different objective than
  the one actually being scored. Combined with the tab result, this
  suggests the single global alpha=0.14 is already close to a genuine
  optimum with no exploitable segment-level heterogeneity between the GBM
  and FM's relative strengths.

## Verdict: REJECT

Neither segmentation clears the +0.001 valid bar in any of 5 seeds; both
regress relative to the current global-alpha blend in nearly every seed.
This confirms — rather than refutes — the overfitting risk the instructions
called out for this method: more free parameters (per-segment alphas)
without a genuine underlying heterogeneity to fit just adds noise. The
existing single global alpha=0.14 (iter63's fine sweep) stands as the
best-known blend weight; no further work on this axis is recommended.

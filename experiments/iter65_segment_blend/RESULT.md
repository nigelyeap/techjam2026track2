# iter65 — Per-segment blend alpha (by `tab`, by activity tertile) — independent re-verification

## Provenance

This finding was first reported by a teammate ("Xuxia") running an
independent Claude Code instance on a separate clone of this repo (see
`XUXIA_SUMMARY.md`, section 6a, and `experiments/LEDGER.md`'s "Parallel
track" section). This iteration independently re-implements and re-checks
the same hypothesis from scratch on this clone, per direct user instruction
("try the methods listed under xuxia's yourself").

## Hypothesis

The current best submission (iter63) blends the GBM and FM ensemble scores
with a single global `alpha=0.14`. If the two models' relative strength
varies across segments of the data (e.g. some tabs or activity levels favor
the GBM more, others the FM more), a per-segment alpha could beat the
global one.

## Implementation

Reused iter63's exact `rate_only` GBM feature pipeline
(`experiments/iter63_decay_tab_rate/train.py`), trained fresh at all 5
seeds (0-4), plus the unchanged iter38 FM 5-seed ensemble
(`experiments/iter65_segment_blend/gen_scores.py`). **Harness-fidelity
check**: seed-0 global-alpha blend reproduced iter63 exactly
(valid=0.67606, test=0.65955) before trusting any new result.

Two segmentations, both computed on `tab`/activity-decay values already
present in the GBM-native feature set (no new features):
- **By `tab`** (15 distinct values, sizes 5 to 92,672 valid rows — highly
  imbalanced).
- **By activity tertile** (3 balanced groups, ~41.6k valid rows each, cut
  points from valid quantiles only).

For each segment, alpha was swept on a grid (0.00-0.40, step 0.02)
restricted to that segment's own valid rows, picking the value maximizing
`evaluate()` on that subset; the winning alpha was then applied to both
the valid and test rows in that segment. The final valid/test primary was
computed over the **full** set (not per-segment), for a fair comparison to
the global-alpha baseline.

## Result (5 GBM seeds)

| | by-tab mean Δvalid | by-tab wins | by-tier mean Δvalid | by-tier wins |
|---|---|---|---|---|
| vs. global alpha=0.14 | **-0.00878** | 0/5 | **-0.00510** | 0/5 |

Representative seed-0 row: global valid=0.67606/test=0.65955; by-tab
valid=0.66696 (-0.00910)/test=0.65101 (-0.00854); by-tier valid=0.67281
(-0.00325)/test=0.65791 (-0.00165).

Both segmentations regress in **every one of 5 seeds**, with test moving
the same direction as valid throughout. This independently confirms
Xuxia's reported direction and conclusion (their numbers: by-tab mean
-0.00499 valid, by-tier mean -0.00568 valid, also 0/5 wins) — the exact
magnitudes differ (different alpha grid, and per-segment-restricted
`evaluate()` semantics can differ slightly by implementation detail), but
the qualitative finding is identical and equally clean: the global alpha
is already close to a real optimum, and per-segment fitting overfits small
or imbalanced groups rather than capturing genuine GBM/FM heterogeneity.

## Verdict: **REJECT** (independently confirmed)

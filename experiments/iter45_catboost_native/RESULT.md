# iter45 — CatBoost on iter44's GBM-native (un-bucketed) encoding

## Motivation

iter43 rejected CatBoost, but only on FM's bucketed encoding — the same bottleneck iter44
diagnosed and fixed for LightGBM (native encoding beat FM outright, valid 0.66135 vs 0.63988).
iter44's own writeup flagged CatBoost-on-native-encoding as an untried, cheap follow-up. Reused
iter44's `prepare()`/`CAT_COLS`/`NUM_COLS` unchanged so the feature set is bit-for-bit identical
to the LightGBM run — only the model changes.

## A real bug caught before trusting the first result

The first run produced `valid primary=0.4835` (essentially the trivial random floor, ~0.483 —
see iter44's tie-artifact baseline) alongside `test primary=0.6222` (real signal) — a huge,
implausible valid/test gap for the same model on the same feature set. Root cause: CatBoost's
`Pool` requires rows grouped/sorted by `user_id` for ranking, so `train.py` predicted on a
**user-sorted** copy of the valid frame, but evaluated those scores against the **original,
unsorted** `y['valid']`/`u['valid']` arrays — a row-order misalignment, not a real ranking
failure. Fixed by evaluating against the same sorted `uva`/`yva` arrays the sorted predictions
came from (`experiments/iter45_catboost_native/train.py`, one-line fix). Rerun after the fix
produced a self-consistent valid/test pair (0.62127/0.62221) — the kind of harness bug the
project's verification discipline exists to catch before anything gets promoted.

## Standalone result

Default config (`depth=6, iterations=500, learning_rate=0.05, l2_leaf_reg=3.0, loss=YetiRank`):
valid 0.62127 / test 0.62221 — already below FM's ensemble (0.63988/0.64187) and below
LightGBM's *first-pass* native result (0.63935/0.63698), let alone LightGBM's tuned optimum
(0.66135/0.64794).

A 12-point sweep (`sweep.py`) tested depth 1-6, three loss functions (`YetiRank`,
`PairLogitPairwise`, `PairLogit`), and learning-rate/l2 variants at the best depth. Findings:

- **Shrinking capacity helps, mirroring LightGBM's pattern, but the effect is much smaller and
  does not extend to the literal floor.** depth=6→0.62127, depth=5→0.62440, depth=4→0.62746,
  depth=3→0.62783, depth=2→0.62717–0.62964 (best across lr/l2 variants), but **depth=1 is worse**
  (0.60331) — unlike LightGBM's num_leaves=2 floor, CatBoost's optimum is an interior point
  (depth=2), not the boundary.
- `YetiRank` clearly beats both pairwise losses at depth=2 (`PairLogitPairwise` 0.59005,
  `PairLogit` 0.58390 vs `YetiRank` 0.62717) — listwise loss matters more for CatBoost here than
  the encoding did.
- Best found: `depth=2, learning_rate=0.1, l2_leaf_reg=3.0, YetiRank` → **valid 0.62964 / test
  0.62999**.

**Verdict: REJECT (standalone).** Even at its best-found config, CatBoost-native (0.62964 valid)
falls well short of FM's ensemble (0.63988) and far short of LightGBM-native (0.66135) — a
gap of ~0.031 valid that capacity/loss tuning alone doesn't close. Unlike iter41→iter44's
LightGBM story, giving CatBoost the native encoding closed only part of its gap to FM, not all
of it — the native encoding was necessary but not sufficient for CatBoost specifically. Possible
explanation (not verified further here, flagged as future work): CatBoost's ordered-boosting
default and symmetric-tree structure may fit un-bucketed continuous features differently than
LightGBM's leaf-wise growth; not chased further given the size of the remaining gap.

Not chased as a blend candidate on its own strength, but tested anyway as a third diversity
source in iter47's stacking experiment — see that writeup for whether its errors are different
enough from FM/LightGBM's to add value despite being individually weaker.

Code: `experiments/iter45_catboost_native/{train.py,sweep.py}`.

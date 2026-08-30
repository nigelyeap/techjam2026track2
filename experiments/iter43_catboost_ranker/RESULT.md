# iter43 — CatBoostRanker (new library, second GBM test)

## Hypothesis
iter41 showed LightGBM (one GBM implementation) underperforms FM on the
FM-bucketed feature set. Before concluding "GBMs don't fit this problem,"
test a second, independently-implemented GBM library with a different
default ranking loss (CatBoost's `YetiRank`, a listwise NDCG-approximation
loss, vs. LightGBM's pairwise `lambdarank`) and its native `Pool`
categorical handling, to rule out a LightGBM-specific implementation
quirk before attributing the gap to the feature representation itself.

## Method
- Same feature set and `load_ext`/`encode_ext` pipeline as iter41 (via
  importlib, no re-derivation).
- All feature columns (bucketed categoricals + the global-offset integer
  encoding) passed to `catboost.Pool` as `cat_features` (every column
  treated as categorical, matching how FM/LightGBM both see this data).
- `CatBoostRanker(loss_function='YetiRank', eval_metric='NDCG:top=5',
  early_stopping_rounds=50, use_best_model=True)`.
- Default run: `depth=6, iterations=1000, learning_rate=0.05,
  l2_leaf_reg=3.0`.
- Sweep: depth (3, 4), iterations/learning_rate/l2_leaf_reg combinations,
  and the loss-function axis (`YetiRank`, `QueryRMSE`, `PairLogitPairwise`).

## Results

**Default config:**
| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.67772 | 0.54109 | 0.60941 |
| test | 0.67558 | 0.53880 | 0.60719 |

Badly underperforms both FM (0.6389 valid) and even LightGBM's untuned
default (0.6319 valid) — CatBoost is not simply matching LightGBM's
shortfall, it's noticeably worse out of the box.

**Sweep** (ranked by valid primary):
| valid | test | config |
|---|---|---|
| **0.61241** | 0.61056 | `depth=4, iterations=300, lr=0.05, l2_leaf_reg=3.0, YetiRank` |
| 0.61217 | 0.60973 | `depth=4, iterations=500, lr=0.03, l2_leaf_reg=10.0, YetiRank` |
| 0.61045 | 0.60737 | `depth=3, iterations=300, lr=0.05, l2_leaf_reg=3.0, YetiRank` |
| 0.59910 | 0.59607 | `depth=3, iterations=300, lr=0.05, l2_leaf_reg=3.0, QueryRMSE` |
| (aborted) | | `depth=4, iterations=300, lr=0.05, l2_leaf_reg=3.0, PairLogitPairwise` — ran ~57min without finishing (CatBoost's pairwise loss appears far slower on this data than YetiRank/QueryRMSE); killed to free CPU for the higher-priority iter44 blend once the verdict was already unambiguous from the other 4 configs |

Shallower depth (3 vs. the default 6) barely moves the needle (0.610 vs
0.609), and depth=4 only adds another +0.002 — an order of magnitude
smaller effect than iter41's LightGBM sweep saw from its own
capacity/regularization axis. `QueryRMSE` (a regression-style loss on raw
relevance) is markedly worse than `YetiRank`, confirming the ranking-native
loss is the right choice for this library; it just doesn't close the gap.

## Diagnosis
The depth/capacity insensitivity (unlike LightGBM's sweep, where tree size
had a large, consistent effect) points away from a tuning problem and
toward the same root cause diagnosed for LightGBM in iter41/iter44: the
FM-bucketed categorical representation throws away the continuous
ordering/magnitude information both tree libraries are built to exploit.
CatBoost is simply more sensitive to this starvation than LightGBM was —
it never gets close enough to FM for hyperparameters to matter much.
This was not re-tested with iter44's native-feature representation (out
of scope for this iteration; LightGBM was the library carried forward
into iter44 since it already showed the stronger response). If GBM
score-blending becomes a priority again, CatBoost-on-native-features is a
cheap follow-up, but LightGBM-on-native-features (iter44) is already the
much stronger candidate and should be exhausted first.

## Verdict: REJECT (standalone, both feature representations tested: default bucketed only)

Confirms iter41's finding generalizes across GBM libraries, not just
LightGBM's specific implementation — the FM-bucketed feature
representation is the common bottleneck, not a library quirk. Per the
"don't give up on open-source models" instruction, this does not close
the model-family lever: iter44's LightGBM-on-native-features direction
turned out to be the correct one to keep pursuing — after a follow-up
sweep to very low tree capacity (num_leaves=2), it now stands at
**valid 0.66135, test 0.64794**, beating the FM ensemble outright
(valid 0.63988, test 0.64187) and becoming the new project-best single
model (see `experiments/iter44_gbm_native_features/RESULT.md`). CatBoost
was not re-tested on iter44's native-feature representation or its
low-capacity regime; a cheap follow-up if score-blending diversity across
GBM libraries is wanted later, but not the priority now.

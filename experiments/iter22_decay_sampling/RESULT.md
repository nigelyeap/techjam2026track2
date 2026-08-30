# iter22 — decay-aware BPR sampling weight

## Idea
iter16 (current best) feeds the model recency-decayed causal history features
(`decay_rate_3`, `decay_act_3`, halflife=3 days) plus flat `tab_pos`, but its
BPR training loop still picks *which users* get sampled as anchors using the
same flat `pos_len[user] ** alpha` weight iter3/iter9 used (`pos_len` = raw,
undecayed count of a user's positive rows in train) — a quantity now stale
relative to iter16's own recency-decay thesis. This iteration asks: does
re-deriving the sampling weight itself from a recency-decayed activity count
(matching the halflife already used for the input features) change training
dynamics for the better?

**Feature set fed to the model is held fixed at iter16's exact winner
throughout: `(decay_rate_3, decay_act_3, tab)`.** Only the BPR user-sampling
WEIGHT is changed — never the model's input features.

## Mechanism
`data_ext.compute_final_decayed_pos(train_rows, halflife=3)` computes, for
each user, a single scalar: the recency-decayed count of that user's TRAIN
positive rows, decayed to one fixed reference date = the max date present in
train (i.e. "as of the end of the train period"). It reuses the exact same
exponential-decay formula (`0.5 ** (gap_days / halflife)`) as
`compute_decay_features`'s `decayed_pos` output (same halflife=3d mechanism
that produces `decay_rate_3`/`decay_act_3`), but evaluated ONCE per user
rather than causally per training row — this is a training-time sampling
frequency choice, not a per-row feature, so there is no leakage concern: the
value never enters any row's feature vector, and `pos_len` itself (which it
replaces) was already a non-causal aggregate over all of train with no
per-row restriction. `train.py`'s `run_bpr_ext` gained a `sampling_mode`
(`flat`|`decay`) + `alpha` + `decay_halflife` parameterization; `flat` mode
with `alpha=1.0` reproduces iter16 exactly bit-for-bit (verified: seed 0
valid 0.6194308996200562 / test 0.6176092624664307, identical to iter16's
seed-0 numbers in its `results.json`).

## Phase 1 — decayed-sampling-weight alpha sweep (3 seeds: 0-2, halflife=3d)
| alpha | valid mean | valid std | test mean | test std |
|---|---|---|---|---|
| **0.5** | **0.62254** | 0.00069 | **0.62104** | 0.00050 |
| 1.0 | 0.62079 | 0.00049 | 0.61770 | 0.00222 |
| 1.5 | 0.61854 | 0.00027 | 0.61309 | 0.00189 |
| 2.0 | 0.61640 | 0.00023 | 0.61099 | 0.00156 |

Monotonically decreasing in alpha over the whole swept range — flatter
sampling (alpha=0.5, closer to uniform-over-active-users) is clearly best,
sharper sampling (alpha=2.0, over-concentrating on the highest-decayed-
activity users) is clearly worst. alpha=0.5 already beats iter16's 5-seed
baseline (valid 0.62030 / test 0.61698) by a wide margin even at only 3 seeds.

**Parity/baseline reference**: `sampling_mode=flat, alpha=1.0` (iter16's
exact original scheme, re-run here for the harness-fidelity check above) is
also iter16's own already-published 5-seed result — no need to re-run 5 seeds
of the baseline since bit-exact reproduction was confirmed at seed 0.

## 5-seed confirmation — decayed sampling, alpha=0.5, halflife=3d
| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.62177 | 0.62174 |
| 1 | 0.62239 | 0.62067 |
| 2 | 0.62345 | 0.62072 |
| 3 | 0.62211 | 0.62048 |
| 4 | 0.62398 | 0.62147 |
| **mean** | **0.62274** | **0.62101** |
| **std** | 0.00084 | 0.00050 |

vs. iter16 (flat `pos_len^1.0` sampling, current best going in):
valid mean 0.62030 (std 0.00048) / test mean 0.61698 (std 0.00187)

**Δ = +0.00244 valid / +0.00403 test.** Both deltas clear the ~0.001-0.002
"real signal" bar the run has used throughout, and — unlike iter11's
cautionary case (a valid-only win that was exactly canceled by a same-size
test regression) — here the win holds in the **same direction** on both
splits, and matched seed-by-seed:

| seed | Δ valid (decay − iter16) | Δ test (decay − iter16) |
|---|---|---|
| 0 | +0.00234 | +0.00413 |
| 1 | +0.00175 | +0.00588 |
| 2 | +0.00267 | +0.00131 |
| 3 | +0.00198 | +0.00566 |
| 4 | +0.00348 | +0.00321 |

5/5 seeds improve on both splits, no sign flips — this is not winner's-curse
noise from a 4-point alpha grid, it's a consistent, generalizing effect.

## Why it likely helps
iter16's flat `pos_len^alpha` sampling weight still emphasizes users who were
highly active AT ANY POINT during the ~3-week log window, including users
whose activity has since faded (large flat `pos_len` but low recent/decayed
activity). Those users' rows now carry LOW-magnitude `decay_rate_3`/
`decay_act_3` feature values (since those features are themselves causally
decayed as of each row's own date), so over-sampling them as BPR anchors
spends training steps disproportionately on rows where the recency-decayed
input features are least informative/most attenuated. Switching the sampling
weight to the same decayed-activity notion realigns *which* rows get
oversampled with *which* rows the decayed features actually describe well —
and softening the exponent to alpha=0.5 (flatter than iter16's implicit
alpha=1.0) further reduces over-concentration on a shrinking set of
still-decay-heavy users, consistent with the monotonic-decreasing-in-alpha
sweep result above.

## Status: **PROMOTE**
New best: FM + activity-weighted BPR with **decayed-positive-count sampling
weight** (`decayed_pos_total^0.5`, halflife=3 days, evaluated once per user
as of the end of the train period) replacing iter16's flat `pos_len^1.0`
sampling weight, feeding the model iter16's exact unchanged feature set
(`decay_rate_3`, `decay_act_3`, `tab`).

- valid primary mean **0.62274** (std 0.00084), 5-seed — vs iter16 0.62030
  (std 0.00048): **+0.00244**
- test primary mean **0.62101** (std 0.00050), 5-seed — vs iter16 0.61698
  (std 0.00187): **+0.00403**
- Consistent 5/5 seeds, both splits, no sign flips.

## Residual finding for future iterations
The alpha sweep was monotonically decreasing in the swept range {0.5, 1.0,
1.5, 2.0} — alpha=0.5 (the flattest of the four, closest to uniform-over-
BPR-eligible-users) was best, suggesting the true optimum may lie below 0.5
(untested here; the grid was fixed to mirror iter7's original range per this
iteration's brief). Worth sweeping alpha in {0.0, 0.1, 0.25, 0.5} in a future
iteration to see whether flatter-still (or fully uniform-over-eligible-users)
sampling combined with a decayed weight does even better, and whether that
finding also transfers back to reconsidering iter7's flat-`pos_len` alpha
sweep (which found flat sampling insensitive to alpha in this same range,
under the OLD iter9-era features — this iteration shows the picture changes
once decayed features are in play, at least for the decayed sampling scheme).

## Code
`experiments/iter22_decay_sampling/{data_ext.py,train.py,driver.py}`, full
sweep + 5-seed data in `experiments/iter22_decay_sampling/results.json`.

# iter33 — iter30's `init_scale_mult=0.5` stabilization lever applied to iter28's setup (DeepFM on iter24's refined feature set)

## Idea
iter28 stacked iter26's DeepFM deep part on top of iter24's refined feature
set (`decay_rate_2.5, decay_act_2.5, decay_tab_3, last1, lastk_rate, gap`)
and found a flat-to-slightly-negative 5-seed valid result (0.63244 vs
iter24's own 0.63251, **−0.00007**) despite a modest, consistent test-side
gain (+0.00153) — rejected per this project's valid-only selection protocol.
Separately, iter30 swept four variance-reduction levers on plain iter26
`deep_h32` (DeepFM on iter19's *older* feature set) and found
`init_scale_mult=0.5` (halving the MLP's He/linear-readout weight-init
scale) was the only lever that reduced seed-to-seed variance on **both**
valid and test simultaneously without hurting the mean (valid std −24%, test
std −18%). The two findings were never combined. This iteration asks: does
applying iter30's stabilization lever to iter28's exact setup nudge iter28's
noisy, near-zero valid delta into a real positive one?

## Setup
- `data_ext.py`, `train.py` copied from `iter28_deepfm_refined_features/`
  (iter24's refined feature set, unchanged; BPR sampling/optimizer
  hyperparameters unchanged).
- `model.py` copied from `iter28_deepfm_refined_features/model.py` (itself a
  verbatim copy of `iter26_deepfm/model.py`) with **one** addition: an
  `init_scale_mult` constructor kwarg (default `1.0`, exact no-op) that
  multiplies the MLP's init scale, using iter30's exact formula
  (`scale = base_scale * init_scale_mult`), verified bit-for-bit equivalent
  to iter30's own `DeepFMVR` mechanism before running anything (see
  Verification below).
- Feature cache (`.cache_v1_2-2.5-3-3.5__tab_3-7.pkl`, 381MB) reused via
  symlink to `iter28_deepfm_refined_features/`'s cache file rather than
  recomputed — loaded in 3.9s, avoiding a full re-run of the causal-feature
  traversal.
- `results.json` written incrementally (append+save after every single run,
  not buffered) per this round's process instructions, since this exact
  pair of source experiments (iter28, iter30) were both killed mid-run by a
  platform session-limit event last round.

## Verification (init_scale_mult mechanism)
Before running anything, confirmed in isolation:
```
DeepFM(..., seed=0, init_scale_mult=1.0).mlp_W == DeepFM(..., seed=0).mlp_W   # bit-exact
DeepFM(..., seed=0, init_scale_mult=0.5).mlp_W / DeepFM(..., seed=0).mlp_W == 0.5   # exact ratio
```
Both checks passed — `init_scale_mult=1.0` is a true no-op and `0.5` scales
init as intended.

## Phase 0 — harness-fidelity check (`ref_deep_h32`, init_scale_mult=1.0, 3 seeds)

| seed | valid | test |
|---|---|---|
| 0 | 0.63196 | 0.62798 |
| 1 | 0.63484 | 0.63132 |
| 2 | 0.63125 | 0.62979 |

**Bit-exact match to iter28's own published per-seed `deep_h32` numbers**
(Δvalid = Δtest = 0.0000000 for all 3 seeds) — confirms this dir's copy of
iter28's harness (data_ext + model + train) introduces zero drift before the
stabilization lever is applied.

## Phase 1 — stabilized config (`stab_0.5`, init_scale_mult=0.5, 3 seeds)

| seed | valid | test |
|---|---|---|
| 0 | 0.63341 | 0.62902 |
| 1 | 0.63470 | 0.63089 |
| 2 | 0.63148 | 0.63007 |

3-seed valid mean: `stab_0.5` 0.63320 vs `ref_deep_h32` 0.63268 —
**+0.00052**, a positive 3-seed direction, which per this iteration's plan
triggered the 5-seed extension (the bar for extending was intentionally
loose here, since the goal was checking for even a modest gain, not a large
one).

## 5-seed confirmation (both configs, seeds 0–4)

| config | valid mean | valid std | test mean | test std |
|---|---|---|---|---|
| `ref_deep_h32` (init_scale_mult=1.0) | 0.63244 | 0.00125 | 0.62996 | 0.00111 |
| `stab_0.5` (init_scale_mult=0.5) | 0.63252 | 0.00133 | 0.62988 | 0.00108 |

`ref_deep_h32`'s 5-seed numbers are **bit-exact to iter28's own published
5-seed `deep_h32`** (0.63244/0.00125 valid, 0.62996/0.00111 test) — the
reference reproduction is faithful end to end, not just at 3 seeds.

Per-seed deltas (`stab_0.5` − `ref_deep_h32`, matched seed indices):

| seed | Δ valid | Δ test |
|---|---|---|
| 0 | +0.00145 | +0.00104 |
| 1 | −0.00013 | −0.00044 |
| 2 | +0.00023 | +0.00028 |
| 3 | −0.00061 | −0.00220 |
| 4 | −0.00054 | +0.00092 |

**Only 2/5 seeds improve on valid, 3/5 regress** — the encouraging 3-seed
direction (seeds 0–2, all positive) does not hold once seeds 3–4 are added;
this is the classic small-sample-noise pattern the 5-seed confirmation step
exists to catch, not a real, consistent-direction effect.

**The variance-reduction effect itself does not reproduce either**: valid
std goes from 0.00125 (`ref`) to 0.00133 (`stab_0.5`), a **+6% increase**,
opposite in sign to iter30's own −24% finding on iter19's older feature set.
Test std moves from 0.00111 to 0.00108, a negligible −3% (vs iter30's
−18%). A plausible explanation: iter24's richer 6-field feature set changes
the deep part's training dynamics enough (different embedding-magnitude
distribution feeding the MLP, from decayed/momentum features vs iter19's
flatter ones) that halving the MLP's init scale no longer dominates
seed-to-seed variance the way it did on the older feature set — the
stabilization lever is not a universal property of the architecture, it is
somewhat feature-set-dependent.

## Comparison against iter24 (standing valid-best, 5-seed: valid 0.63251/std 0.00050)

| | iter24 (5-seed) | iter28 `deep_h32` (5-seed, = this run's `ref_deep_h32`) | iter33 `stab_0.5` (5-seed) |
|---|---|---|---|
| valid primary | 0.63251 | 0.63244 (**−0.00007**) | 0.63252 (**+0.00001**) |
| test primary | 0.62843 | 0.62996 (+0.00153) | 0.62988 (+0.00145) |

The stabilization lever moves the valid delta from iter28's −0.00007 to
+0.00001 — technically flips the sign, but by an amount roughly **1/130th**
of `stab_0.5`'s own valid std (0.00133) and with no consistent per-seed
direction (see table above). This is indistinguishable from noise, not a
real gain. Test moves negligibly in the other direction (+0.00153 →
+0.00145), also within noise.

## Verdict: **REJECT**

`init_scale_mult=0.5` does not nudge iter28's setup (DeepFM on iter24's
refined feature set) into a real valid-side win. The combination:
- does not clear iter24's valid bar (+0.00001, statistically zero, no
  sign-consistent per-seed pattern — 2/5 seeds up, 3/5 down);
- does not even reliably reduce variance in this setting, unlike its
  clean effect on iter26/iter19's older feature set in iter30 (valid std
  actually rose slightly here instead of falling).

This is a legitimate "we checked a well-motivated combination and it
doesn't work" result. Per this project's valid-only promotion protocol,
iter24 remains the standing best; iter33 is not promoted. Residual finding
for a future round: iter30's `init_scale_mult=0.5` stabilization effect
appears to be specific to the feature set/model configuration it was
originally tuned on (iter19's), not a general property of the DeepFM deep
part — worth keeping in mind before assuming any single variance-reduction
lever transfers across feature-set changes without re-verification.

## Code
`experiments/iter33_deepfm_stabilized/{data_ext.py,model.py,train.py,driver.py}`,
raw results in `experiments/iter33_deepfm_stabilized/results.json` (10 rows:
5-seed `ref_deep_h32` harness-fidelity/reference + 5-seed `stab_0.5`), full
run trace in `driver_log.txt`. No interruptions this run — driver completed
all phases in one uninterrupted background run (~10 minutes wall clock),
polled via a real pid-bound blocking wait (`while kill -0 <pid>; do sleep
15; done`) on the agent's own verified pid (confirmed via `lsof -p <pid> |
grep cwd` pointing at this directory before trusting it).

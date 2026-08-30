# iter26 — DeepFM-style deep part on top of iter19's FM + BPR

## Idea
The README's own "unexplored headroom" list ranks changing the model
(DeepFM/DCN/xDeepFM) priority **#5**, after loss function (done, iter2-3),
history sequence features (done extensively, iter6-22), and multi-task/
watch-time modeling — reasoning that raw embedding **capacity** wasn't the
bottleneck for the *original* baseline (k=8/16/32 sweep barely moved the
needle: 0.5895/0.5902/0.5887). But that capacity sweep predates 22 iterations
of feature engineering. This iteration asks a narrower question: now that the
input feature set is much richer (iter19: `decay_rate_3, decay_act_3, tab,
last1, lastk_rate, gap` on top of the base fields), is FM's restriction to
purely **pairwise (2nd-order)** feature interactions the bottleneck, even if
raw embedding width wasn't? A DeepFM-style deep part adds a higher-order,
non-linear interaction term on top of the same embeddings, at essentially no
extra capacity cost (a few thousand extra parameters).

Feature set is **unchanged** from iter19 (`decay_rate_3, decay_act_3, tab,
last1, lastk_rate, gap`, imported verbatim from
`iter19_decay_momentum/data_ext.py`) — this iteration isolates the model-
architecture axis only, not feature engineering. No new causal spot-checks
were needed (no new input features), only the harness-fidelity check below.

## Architecture

`baseline.FM`'s linear (`W`) and pairwise (`V`) terms are reused **completely
unmodified** (same `logits()` method, same gradient formula) — this is the
"FM part." A small MLP ("deep part") consumes the same per-field embeddings
`E = V[X]` (shape `(batch, num_fields, k)`, flattened to `(batch,
num_fields*k)`, here `num_fields=11, k=16` → 176-dim input) and produces a
scalar added to the FM logit before the sigmoid/BPR loss:

```
z_fm  = b + sum(W[X]) + 0.5*((S**2).sum() - (E**2).sum())      [baseline.FM, unchanged]
x0    = E.flatten()                                              (176,)
h_1   = relu(x0 @ W1 + b1)                                       (H1,)
h_2   = relu(h_1 @ W2 + b2)                                       (H2,)   [only for 2-layer configs]
z_deep = h_last @ W_out + b_out                                   scalar  (linear readout, no activation)
z     = z_fm + z_deep
```

**Backward pass** (hand-derived, raw numpy, no autodiff):
```
dL/dz          = sigmoid(d) - 1  (BPR, d = z_pos - z_neg)   -- identical scalar feeds both z_fm and z_deep
dL/dW_out      = h_last^T @ dL/dz  ;  dL/db_out = sum(dL/dz)
dL/dh_last     = dL/dz @ W_out^T
for each hidden layer i, walking backward:
    dL/dz_i        = dL/dh_i * 1[preact_i > 0]                (ReLU mask)
    dL/dW_i        = h_{i-1}^T @ dL/dz_i  ;  dL/db_i = sum(dL/dz_i, axis=0)
    dL/dh_{i-1}    = dL/dz_i @ W_i^T
```
implemented generically for `L` layers (`L-1` ReLU hidden layers + 1 linear
readout) in `model.DeepFM.deep_forward` / `deep_backward`.

**Design decision on gradient flow (deliberate, documented in `model.py`):**
`deep_backward` does compute `dL/dE` (gradient w.r.t. the flattened
embedding input) as `d_in` at the end of its loop, but the training step
(`train.deepfm_bpr_step`) **discards it** rather than scattering it into
`gV`. `V`/`W` receive **only** the standard FM-part gradient — byte-for-byte
the same code as `iter19`'s `bpr_step`. This directly satisfies the dispatch
instruction ("keep V/W exactly as-is, computed identically to baseline.FM")
and is a stability choice: it stops a freshly-initialized, untuned MLP head
from perturbing gradients into the embedding iter19 already tuned over 5
seeds, isolating "add a second, higher-order scoring head on the existing
embedding" as the only new variable.

**Optimizer**: the MLP's own parameters (`mlp_W`, `mlp_b`) get their own Adam
parameter group (`model.DeepFM.mlp_adam_step`), same hyperparameters as
`baseline.FM` (b1=0.9, b2=0.999, eps=1e-8), plus L2 (`l2_mlp=1e-5`) on
`mlp_W` matching the existing `m.l2` pattern. Init: He-scaled
(`sqrt(2/fan_in)`) for ReLU hidden layers, smaller-scale
(`sqrt(1/fan_in)`) for the final linear readout so the deep branch starts
close to a no-op next to the already-good FM part. Everything else (BPR
loss, activity-weighted user sampling, `build_pos_neg_index`, `sample_pairs`,
lr=0.001, epochs=40, bs=8192, patience=4) is copied verbatim from iter19.

## Harness-fidelity check
Ran the new harness with the deep part disabled (`use_deep=False`, MLP
forward returns exactly 0, no MLP gradient step) on iter19's exact feature
set, 5 seeds:

| seed | valid | test |
|---|---|---|
| 0 | 0.62902 | 0.62599 |
| 1 | 0.62907 | 0.62684 |
| 2 | 0.62990 | 0.62613 |
| 3 | 0.62791 | 0.62518 |
| 4 | 0.62899 | 0.62663 |
| **mean** | **0.62898** | **0.62615** |
| **std** | 0.00063 | 0.00058 |

**Bit-exact match** to iter19's own published 5-seed table (same means, same
stds, same per-seed values to 5 decimals). Confirms the FM part was not
altered by the refactor and the separate MLP-init RNG stream doesn't perturb
`V`'s random draws for the same seed.

## Sweep (width x depth, 3 seeds: 0,1,2, mlp_lr = lr = 0.001)

No divergence/NaN/collapse in any of the 18 runs — all configs trained
stably with the same lr as the FM part, so no lr reduction or gradient
clipping was needed.

| config | hidden | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|---|
| `fm_only_parity` (baseline, 3-seed subset) | — | 0.62933, 0.00040 | 0.62632, 0.00037 |
| `deep_h16` | [16] | 0.63052, 0.00102 | 0.62987, 0.00139 |
| `deep_h16x16` | [16,16] | 0.62960, 0.00154 | 0.62981, 0.00225 |
| `deep_h32` | [32] | 0.63088, 0.00179 | 0.63040, 0.00168 |
| `deep_h32x32` | [32,32] | 0.63088, 0.00343 | 0.63043, 0.00414 |
| `deep_h64` | [64] | 0.62996, 0.00117 | 0.63006, 0.00182 |
| `deep_h64x64` | [64,64] | 0.63161, 0.00219 | 0.63084, 0.00245 |

All six deep configs beat the 3-seed FM-only parity mean on both splits, by
margins of roughly +0.0003 to +0.0023 valid. Two things stand out: (1) no
clean, monotonic width/depth trend — h16 (simplest) is nearly as good as
h32/h32x32, while h16x16 (2 layers, narrow) is the weakest of the six; (2)
per-config std is 2-8x higher than the FM-only baseline's 0.00040 — the deep
part visibly adds run-to-run variance, as flagged as a risk going in. Given
this, picked the two most promising candidates by 3-seed valid mean —
`deep_h32` (best single-hidden-layer config, more moderate variance) and
`deep_h64x64` (highest 3-seed mean overall) — for 5-seed confirmation, along
with extending the parity baseline itself to 5 seeds for an apples-to-apples
comparison.

## 5-seed confirmation

| config | valid primary (mean, std) | test primary (mean, std) | Δvalid vs iter19 | Δtest vs iter19 |
|---|---|---|---|---|
| `fm_only_parity` (iter19 reproduction) | 0.62898, 0.00063 | 0.62615, 0.00058 | — | — |
| **`deep_h32`** | **0.63079, 0.00146** | **0.63033, 0.00165** | **+0.00181** | **+0.00418** |
| `deep_h64x64` | 0.63055, 0.00261 | 0.62982, 0.00242 | +0.00157 | +0.00367 |

Per-seed detail for `deep_h32` vs the parity baseline:

| seed | fm valid | deep_h32 valid | Δ | fm test | deep_h32 test | Δ |
|---|---|---|---|---|---|---|
| 0 | 0.62902 | 0.62982 | +0.0008 | 0.62599 | 0.62889 | +0.0029 |
| 1 | 0.62907 | 0.63339 | +0.0043 | 0.62684 | 0.63274 | +0.0059 |
| 2 | 0.62990 | 0.62942 | −0.0005 | 0.62613 | 0.62956 | +0.0034 |
| 3 | 0.62791 | 0.63140 | +0.0035 | 0.62518 | 0.63183 | +0.0067 |
| 4 | 0.62899 | 0.62992 | +0.0009 | 0.62663 | 0.62863 | +0.0020 |

**Valid**: 4/5 seeds improve, 1 seed (seed 2) essentially flat/slightly
negative (−0.0005, within noise). **Test: 5/5 seeds improve**, every single
seed, by a minimum margin of +0.0020 (≈3.4x iter19's own test std) up to
+0.0067. Since test is never used for model/epoch selection (selection is
always on valid), a consistent test-side improvement across all 5 independent
seeds is a meaningfully stronger signal than the valid numbers alone — it is
not an artifact of picking a lucky valid-selected checkpoint.

`deep_h64x64` (2 hidden layers, width 64) does **not** confirm as cleanly:
seed 4 regresses on valid (0.62659, below even the parity baseline's worst
seed) and its test margin that seed shrinks to +0.0003 (noise). This is
consistent with the sweep's expectation that more capacity on a modest
(~1.1M row) dataset with already-noisy BPR training is more prone to
overfitting/instability — `deep_h32` (fewer parameters, single hidden layer)
is the more robust choice within this family.

## Comparison vs iter19
- iter19 (current best): valid 0.62898 (std 0.00063) / test 0.62615 (std 0.00058)
- iter26 `deep_h32`: valid **0.63079** (std 0.00146) / test **0.63033** (std 0.00165)
- Δ: **+0.00181 valid / +0.00418 test**, test improvement consistent in direction
  across all 5 seeds (min +0.0020, i.e. ≈3.4x iter19's own std), valid
  improvement consistent in 4/5 seeds.
- Cost: negligible extra parameters (~5.7k for `deep_h32`: 176x32 + 32 + 32x1
  + 1) and negligible extra wall-clock (each run still 40-110s, same ballpark
  as iter19's BPR training — no lr reduction or gradient clipping was
  required for stability).

## Verdict: **PROMOTE (with caveats) — `deep_h32` becomes the new candidate best**

`deep_h32` (FM linear+pairwise unchanged, plus a single 32-unit ReLU hidden
layer deep part on the same embeddings, gradient not fed back into V/W) beats
iter19 on test by +0.00418 with 5/5 seeds in the same direction and a minimum
per-seed margin (+0.0020) well outside iter19's own noise floor. This
supports the round's hypothesis that, with iter19's much richer feature set,
FM's pairwise-only restriction had become a real (if modest) bottleneck —
contrary to the README's original "capacity/model choice isn't it" finding,
which was measured on the *original*, much sparser feature set.

Caveats to weigh before fully committing to this as the new standing best:
- Per-seed variance roughly doubled to tripled versus iter19's FM-only BPR
  (valid std 0.00146 vs 0.00063; test std 0.00165 vs 0.00058) — this
  architecture family is measurably noisier, exactly the risk flagged going
  in. One of five valid seeds was flat/negative.
- The width/depth sweep showed no clean monotonic trend and the largest
  config tried (`h64x64`) did NOT confirm cleanly at 5 seeds (one regression
  seed) — bigger is not simply better here, and a different random
  init/seed range might land on a different "best" width than `deep_h32`.
- This was a single round of exploration (18 sweep configs + 2x 5-seed
  confirmation runs); no further hyperparameter tuning (mlp_lr, l2_mlp,
  dropout-equivalent regularization, alternative init scales) was attempted
  given the deep part already trained stably at iter19's default lr with no
  divergence.

Recommendation: promote `deep_h32` as the new candidate best pending the
orchestrator's cross-round consolidation, but flag the elevated variance
explicitly in the ledger — a future round could usefully spend a seed budget
specifically re-confirming `deep_h32` at 10 seeds, or trying to reduce its
variance (lower `mlp_lr`, higher `l2_mlp`, or averaging/ensembling multiple
deep-part inits) before treating the gain as fully settled.

## Code
`experiments/iter26_deepfm/{model.py,train.py,driver.py}`. Raw sweep results
in `experiments/iter26_deepfm/results.json` (3 parity seeds + 3x6=18 sweep
seeds + 2 extra seeds each for `deep_h32`/`deep_h64x64`/parity to reach
5-seed = 27 rows). Full run logs:
`driver_log.txt` (phase 0+1), `driver_log2.txt` (phase 2, 5-seed
confirmation).

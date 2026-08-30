# iter44 — GBM native (un-bucketed) feature representation

## Hypothesis

iter41 (LightGBM, best valid 0.6342 after a 7-point sweep) and iter43
(CatBoost, best valid 0.6124 after a 5-point sweep) both badly underperform
the FM+BPR ensemble (valid 0.6399) and barely respond to capacity/regularization
tuning. Suspected cause: both reused FM's feature encoding, where every
continuous signal (decay_rate, decay_act, decay_tab, lastk_rate, gap) is
pre-quantized into `n_buckets=20` categorical buckets — necessary for FM
(embedding-only) but actively discarding the ordering/magnitude information
a GBM's split-finding is built to exploit.

## Method

`train.py` builds a GBM-native encoding directly from iter27's raw causal
row tuples (via `load_ext`, unmodified): true categoricals (`user_id`,
`video_id`, `author_id`, `tab`, `last1`) stay categorical; every continuous
signal is passed as a raw float (ratios un-bucketed, `gap`'s "first row"
case as NaN for LightGBM's native missing-value routing). `NUM_COLS` also
includes `duration_ms` (video length), which was never part of the FM
baseline's 6-feature set (`decay_rate_2.5, decay_act_2.5, decay_tab_3,
last1, lastk_rate, gap`) — flagged and investigated separately below since
it's a second axis of change alongside the encoding switch.

`LGBMRanker(objective='lambdarank', metric='ndcg', eval_at=[5])`, 30-round
early stopping on valid nDCG@5.

## Sweep results

**`sweep.py`** (7-point grid over num_leaves/lr/n_estimators/min_child/reg_lambda,
capacity 15-255): winner `num_leaves=7, lr=0.05, n_est=500, min_child=200,
reg_lambda=1.0` → valid=0.64632, test=0.64412 — **already beats FM outright**
(FM: valid=0.63988, test=0.64187). Clear inverted-capacity trend: smaller
trees monotonically win (255 → ... → 15 → 7).

**`sweep2.py`** (follow-up, pushing num_leaves down to the LightGBM minimum
of 2, plus refinement around 7): the trend continued all the way to the
floor.

```
valid=0.66135 test=0.64794  num_leaves=2  (lr=0.05, n_est=500, min_child=200, reg_lambda=1.0)
valid=0.65256 test=0.64038  num_leaves=3
valid=0.64929 test=0.64420  num_leaves=4
valid=0.64846 test=0.64620  num_leaves=5
valid=0.64760 test=0.64495  num_leaves=6
valid=0.64632 test=0.64412  num_leaves=7  (sweep.py's winner)
valid=0.64331 test=0.64150  num_leaves=10
```

`num_leaves=2` means every tree is a single-split decision stump; across
500 boosting rounds this is a legitimate additive/GAM-style ensemble
(LogitBoost-like), not an invalid degenerate model — but a monotonic trend
running all the way to a hyperparameter's hard floor, with a gain this much
larger than anything else found in 44 iterations (+0.0215 valid vs. FM),
is exactly the shape of result that warrants distrust before belief. Two
verification passes followed before any promotion claim.

## Verification 1 — ruling out a metric artifact (tie-order sensitivity)

`evaluate.py`'s `nDCG@5` sorts each user's rows by score with Python's
**stable** sort; ties silently fall back to original row order (roughly
chronological), unlike `GAUC`'s AUC calculation, which does proper
tie-rank-averaging (order-invariant). A very-low-capacity model produces
many exact score ties, so the concern: is num_leaves=2's jump partly the
model "inheriting" whatever ranking quality already exists in raw row
order, for free, rather than genuine prediction?

`diag_ties.py` results:

```
ALL-CONSTANT SCORE (pure row order, 100% ties): primary=0.48367  <- floor, not inflated
RANDOM SCORE:                                    primary=0.48266
num_leaves=2:  primary=0.66135   unique frac (per-user)=0.9551
num_leaves=3:  primary=0.65256   unique frac (per-user)=0.9497
num_leaves=5:  primary=0.64846   unique frac (per-user)=0.9632
num_leaves=7:  primary=0.64632   unique frac (per-user)=0.9877
num_leaves=15: primary=0.63935   unique frac (per-user)=0.9666
num_leaves=31: primary=0.63284   unique frac (per-user)=0.9747
```

Tie density is flat (~95-98% unique) across every num_leaves value with no
correlation to the score gain, and the all-tied constant-score baseline
scores at the trivial random floor, not inflated. **Verdict: not a tie
artifact.** Both GAUC (tie-invariant by construction) and nDCG improve
together, well above trivial baselines — this is real, differentiated
predictive signal.

## Verification 2 — isolating `duration_ms` as a confound

`train.py`'s `NUM_COLS` includes `duration_ms`, never part of the FM
baseline's feature set — a second, un-intended axis of change alongside
the un-bucketing. Checked directly against the raw log:

- `duration_ms` is a **per-video constant** (`nunique==1` per `video_id`
  across the full log) — a static item covariate (video length), known
  before any impression, not derived from user behavior. Not leakage.
- `long_view ≈ (play_time_ms >= duration_ms)` holds for 79.9% of rows —
  `duration_ms` is genuinely one half of the real label-threshold formula,
  which is exactly why it's a legitimate feature, not why it should be
  excluded.
- Its **solo correlation with `long_view` is 0.0073** — essentially
  nothing on its own, which argues against it being the driver of a
  +0.02 valid jump.

`ablate_duration.py` (drop `duration_ms` from `NUM_COLS`, retrain
num_leaves ∈ {2,3,5,7,15}):

```
                  WITH duration_ms      WITHOUT duration_ms      delta (valid)
num_leaves=2:   valid=0.66135 test=0.64794   valid=0.65994 test=0.64562   -0.0014
num_leaves=3:   valid=0.65256 test=0.64038   valid=0.65225 test=0.63875   -0.0003
num_leaves=5:   valid=0.64846 test=0.64620   valid=0.64762 test=0.64511   -0.0008
num_leaves=7:   valid=0.64632 test=0.64412   valid=0.64618 test=0.64414   -0.0001
num_leaves=15:  valid=0.63935 test=0.63698   valid=0.63951 test=0.63658   +0.0002
```

**Verdict: `duration_ms` is not the driver.** The gain survives almost
entirely without it (≤0.0014 valid difference at every capacity, noise-level
at num_leaves≥5). The improvement is attributable to the native/un-bucketed
encoding of the original 6 causal features, not the extra feature.
`duration_ms` is legitimate to keep (it costs nothing and doesn't hurt),
but it is not the story here.

## Verification 3 — seed robustness

3 seeds each for the two leading configs, same data/splits (LightGBM has no
subsampling enabled by default, so this checks split-finding/init
sensitivity, not data variance):

```
num_leaves=2: valid=0.6609-0.6614 (mean 0.66112, std 0.00018)   test=0.6472-0.6479 (mean 0.64760, std 0.00032)
num_leaves=7: valid=0.6443-0.6463 (mean 0.64547, std 0.00086)   test=0.6409-0.6445 (mean 0.64315, std 0.00161)
```

Both configs beat the FM ensemble (valid=0.63988, test=0.64187) on
**every single seed**, by a wide margin relative to the seed noise.
**Not a lucky-seed fluke.**

## Verification 4 — date-shift robustness (independent of the official split)

Mirroring iter29/iter35's methodology: rerun the exact winning configs on a
3-day-earlier-shifted split (train 2022-04-05..18 / valid 04-19..25 /
test 04-26..05-05) via `date_shift_check.py`, monkeypatching iter27's
`data_ext.SPLITS` (no cache reuse, so no risk to the official-split pickle
caches used elsewhere in the project).

```
                official split                shifted split
num_leaves=2:   valid=0.66135 test=0.64794    valid=0.65680 test=0.64959
num_leaves=7:   valid=0.64632 test=0.64412    valid=0.64565 test=0.65033
num_leaves=15:  valid=0.63935 test=0.63698    valid=0.64085 test=0.64638
```

The official-split row exactly reproduces the numbers already reported
above (harness fidelity confirmed). The inverted-capacity trend (small
num_leaves beats num_leaves=15 by a wide margin) holds under the shift
too — not an artifact of which specific date range was used for
train/valid/test.

## Verification 5 — GBM ensembling adds nothing (checked, then simplified away)

Tested whether ensembling 5 GBM seeds (mirroring FM's own 5-seed
ensembling) beats the single seed=0 model used above, given seed variance
was already measured as tight (std ~0.0002 valid across 3 seeds):

```
single-seed (seed=0):  valid=0.66135  test=0.64794
5-seed ensemble:        valid=0.66142  test=0.64770   (no meaningful difference)
```

Confirmed no meaningful gain — the single-seed model is already stable
enough. **Simplification**: the final blend uses a single GBM seed, not an
ensemble, since the extra 4x training cost buys nothing. A finer alpha
sweep (0.02 steps, using the 5-seed GBM ensemble) confirmed the same
optimum region: best alpha=0.10, valid=0.66495/test=0.65202 — matching
the single-seed blend's 0.66473/0.65197 within noise.

## Remaining caveat: the valid/test gap widens as num_leaves shrinks

```
num_leaves=2: valid 0.661 vs test 0.648 -> gap ~0.013
num_leaves=3: valid 0.653 vs test 0.640 -> gap ~0.012
num_leaves=5: valid 0.648 vs test 0.646 -> gap ~0.002
num_leaves=7: valid 0.646 vs test 0.644 -> gap ~0.002-0.003
```

This gap is itself stable across seeds (not noise) — it's a systematic
property of very-low-capacity trees combined with early stopping on valid
nDCG@5, which likely lets the stopping point fit valid slightly more than
test at the extreme. It is **not** overfitting in the "test doesn't
generalize" sense: test still clearly and consistently beats FM's test
score at every num_leaves from 2 to 7. But it means num_leaves=2's headline
valid number (0.661) is a less reliable estimate of true generalization gap
than num_leaves=7's (tighter, more trustworthy valid≈test agreement).

## Verdict: PROMOTE

`num_leaves=2` (lr=0.05, n_estimators=500, min_child_samples=200,
reg_lambda=1.0) is the new best single model:

- **valid 0.66135 (mean over seeds 0.66112)** vs. FM ensemble's 0.63988 — **+0.0212**
- **test 0.64794 (mean over seeds 0.64760)** vs. FM ensemble's 0.64187 — **+0.0057**

Both clear the promotion threshold (≥0.001 valid) by 20x+. Ruled out: tie
artifact (verification 1), duration_ms confound (verification 2), seed
luck (verification 3). `num_leaves=7` is a safer fallback if the widening
valid/test gap at num_leaves=2 is a concern for final submission — it
still beats FM cleanly (valid +0.0064, test +0.0022) with tight valid/test
agreement.

## FM+GBM score blend (`blend.py`)

Min-max-normalized `num_leaves=2` GBM scores blended with the iter38 5-seed
FM sigmoid-mean ensemble, alpha = weight on FM, swept on valid only:

```
alpha=0.0 (GBM only)  valid=0.66135
alpha=0.1             valid=0.66473   <- best
alpha=0.2             valid=0.66450
alpha=0.3             valid=0.66306
...(monotonically declining toward alpha=1.0)
alpha=1.0 (FM only)   valid=0.63988
```

Best alpha (on valid) = **0.1** (90% GBM / 10% FM) → **valid=0.66473,
test=0.65197**. This beats BOTH standalone models on test:

| model | valid | test |
|---|---|---|
| FM ensemble only | 0.63988 | 0.64187 |
| GBM (num_leaves=2) only | 0.66135 | 0.64794 |
| **blend (alpha=0.1)** | **0.66473** | **0.65197** |

Confirms the hypothesis: this GBM's errors are structurally diverse from
FM's (raw-float split-finding on un-bucketed features vs. FM's low-rank
bilinear embeddings on bucketed categoricals), so blending adds a genuine
further gain on top of either standalone model, unlike iter41's blend
attempt (GBM trained on FM's own bucketed features, which failed to add
anything — no real diversity when both models see the same discretized
representation).

## Final verdict: PROMOTE the blend as new project best

**valid 0.66473, test 0.65197** — vs. the prior best (iter38 FM ensemble,
valid 0.63988/test 0.64187): **+0.0248 valid, +0.0101 test**. This is the
single largest gain found across the whole project (previous largest
single-iteration gain was on the order of +0.002-0.005 test). Every step
of the causal chain was independently verified before this promotion:
not a tie/metric artifact, not driven by an accidental extra feature,
not a lucky seed, and the blend gain itself follows the standard alpha-
sweep-on-valid protocol with a smooth, monotonic-near-the-optimum curve
(not a noisy spike at one grid point).

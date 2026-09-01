# TikTok TechJam 2026: Track 2 Submission

Within-user ranking on KuaiRand-Pure. Label: `long_view`. Metric: mean(GAUC, nDCG@5) ("primary").

## Result

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| FM baseline (organizer-provided, test) | 0.6610 | 0.5282 | **0.5946** |
| iter27 single-model config (5-seed mean, test) | n/a | n/a | 0.63889 |
| iter38 FM ensemble (test, prior final result) | 0.7156 | 0.5681 | 0.64187 |
| iter44 blend (test, prior final result) | n/a | n/a | 0.65197 |
| iter51 blend (test, prior final result) | n/a | n/a | 0.65643 |
| iter55 blend (test, prior final result) | n/a | n/a | 0.65832 |
| iter63 blend (valid, prior final result) | n/a | n/a | 0.67606 |
| iter63 blend (test, prior final result) | n/a | n/a | 0.65955 |
| **Final model (yixi10 3-model blend, valid)** | n/a | n/a | **0.69943** |
| **Final model (yixi10 3-model blend, test)** | n/a | n/a | **0.68432** |

- **Test primary: 0.68432.** A within-user-percentile-normalized blend of three components:
  10% weight on the unchanged iter38 FM+BPR ensemble, 52% on a LightGBM ranker (`num_leaves=2`,
  `linear_tree=True`, `learning_rate=0.10`, lambdarank with `truncation_level=50`/`sigmoid=2.0`)
  trained on a chained causal feature set that adds a 5-day historical watch-depth decay feature
  and a native `upload_type` categorical on top of iter63's features, and 38% on an `XGBRanker`
  tuned independently by teammate yixi on the same native (un-bucketed) causal features. Found by
  yixi (`experiments/iterYIXI1` through `iterYIXI10`) working from a separate branch of this
  project's own research, merged in and independently re-verified from raw CSVs (all three
  components retrained from scratch, no cached/frozen artifacts reused) in
  [`experiments/iterMERGE1_verify_yixi10/RESULT.md`](experiments/iterMERGE1_verify_yixi10/RESULT.md):
  exact match to her claim on every component and the final blend.
- **Valid primary: 0.69943.** Model selection was valid-only throughout.
- **Improvement over baseline: +0.08972 absolute, +15.09% relative** (test primary vs. the FM
  baseline's test primary; +0.0234 valid / +0.0248 test over the prior iter63 blend, from adding
  an XGBoost component and yixi's LightGBM feature/objective refinements)
- Reached across 12 iteration rounds / 37 iterations to convergence (3 consecutive
  non-improving rounds under a pre-declared ε=0.002, N=3 rule), then Round 13 (FM ensembling,
  +0.0020 valid / +0.0030 test), Round 14 (iter44's GBM-native representation + blend,
  a further +0.0248 valid / +0.0101 test), Round 16 (iter51's `linear_tree=True` GBM + blend,
  a further +0.0082 valid / +0.0045 test), Round 17 (iter55's `learning_rate=0.10` resweep
  under `linear_tree=True` + blend, a further +0.0015 valid / +0.0019 test), and Round 19
  (iter63's decayed per-tab rate feature + blend, a further +0.0016 valid / +0.0012 test), all
  reopened on request to keep maximizing score post-convergence, up to the submission deadline.

## What the final model is

The base model (iter27) is an FM (factorization machine) trained with a pairwise BPR ranking
loss instead of the baseline's pointwise logloss, plus three independently-validated additions
layered on top:

1. **Recency-decay / momentum features** (from iter24): exponentially-decayed interaction rate
   and per-tab decayed rate/count features, plus `last1` (most recent watch outcome), `lastk_rate`
   (short-window rate), and `gap` (time since last interaction), all computed causally (train-only
   history, no leakage into valid/test).
2. **Activity-weighted BPR sampling** (from iter23): users are sampled for BPR pairs proportionally
   to `decayed_positive_count ** 0.75` (decay half-life 3 days) rather than uniformly, so training
   attention follows recent activity rather than raw historical volume.
3. **Retuned formula constants** (from iter25): Laplace smoothing `alpha=0.5` and `n_buckets=20`
   for the rate-bucketing features, retuned from the defaults inherited from earlier iterations.

Each addition was found and validated independently (separate iteration, separate 5-seed
confirmation), then fused together in iter27. **iter38** trains this exact config at 5 seeds and
averages the sigmoid-transformed scores of the 5 resulting models, a prediction-time ensemble
that reduces the pure random-init variance already visible in iter27's own 5-seed std
(~0.0007-0.0008) without changing any feature, loss, or hyperparameter. This was the final model
through Round 13.

**Round 14 (iter44) found a further, independent gain and became the final submitted model.**
The FM/BPR line above discretizes every continuous feature into `n_buckets=20` categorical
buckets before training, necessary for FM's embedding lookups, but wasteful for a GBM, which is
built to exploit continuous ordering/magnitude directly. Building a from-scratch, un-bucketed
("GBM-native") encoding of the same causal features and training a `LightGBM LGBMRanker` on it
closed most of two earlier GBM attempts' gap to FM (iter41, iter43, both underperformed FM when
forced through FM's bucketed encoding); a follow-up sweep then found the metric responds
strongly and monotonically to *shrinking* tree capacity, down to LightGBM's hard floor
(`num_leaves=2`): valid 0.66135/test 0.64794 standalone, beating the FM ensemble outright. This
unusually large, monotonic-to-a-floor result was verified rather than promoted outright: ruled
out as a tie-sorting artifact of `evaluate.py`'s stable-sort `nDCG@5`, ruled out as driven by the
one new feature (`duration_ms`) this encoding silently added, confirmed stable across seeds, and
confirmed to hold under a 3-day-shifted date split (see
[`experiments/iter44_gbm_native_features/RESULT.md`](experiments/iter44_gbm_native_features/RESULT.md)
for the full verification chain). Blending its score with the (unchanged) iter38 FM ensemble
(10% weight on FM, 90% on the GBM, the optimum of a validation-only alpha sweep confirmed at two
grid resolutions) gave a further, genuine gain from real model-family diversity (unlike iter41's
earlier attempt to blend a GBM trained on FM's *own* features, which added nothing): valid
0.66473, test 0.65197. This was the final submitted model through Round 15.

**Round 16 (iter51) found one further, independent structural gain and became the new final
submitted model.** At `num_leaves=2`, every tree in iter44's GBM makes exactly one split and
predicts a flat constant on each side, a piecewise-*constant* step function per tree. LightGBM's
`linear_tree=True` option instead fits a (regularized) linear regression per leaf, so with only 2
leaves the tree becomes a genuine piecewise-*linear* function: one split, but each side gets its
own linear model over the continuous features instead of a flat constant. This single-axis change
on top of iter44's exact pipeline and hyperparameters gained +0.0079 valid on the first run,
confirmed tight across 5 seeds (mean valid 0.66926, range 0.66915–0.66943, tighter than iter44's
own seed variance). Re-blending this GBM with the unchanged iter38 FM ensemble at a re-swept
optimum (8% weight on FM, 92% on the GBM) gave a further gain from the same model-family
diversity as iter44's blend: valid 0.67297, test 0.65643, the final submitted model through
Round 16. See
[`experiments/iter51_linear_tree/RESULT.md`](experiments/iter51_linear_tree/RESULT.md) for the
full standalone and blend results.

**Round 17 (iter55) found a further, genuinely new gain.** `linear_tree=True` changes what each
boosting round buys the model (a per-leaf linear fit instead of a flat constant), so
`learning_rate=0.05` (tuned back in iter44 against the old constant-leaf tree) had never been
re-validated against this structural change. A single-axis sweep found `learning_rate=0.10` beats
the baseline by +0.0012 valid on the first run, confirmed tight across 5 seeds (mean valid
0.67011, +0.00085 over iter51's own 5-seed mean, 5/5 seeds improving). Re-blending this GBM with
the unchanged iter38 FM ensemble at a re-swept optimum (10% weight on FM, 90% on the GBM) gave a
further, larger gain at the blend level: **valid 0.67451, test 0.65832**, the final submitted
model through Round 18. Three other Round 17 hypotheses (retesting capacity, the tree's own
`linear_lambda` regularization, and the previously-rejected hour-of-day feature, all under
`linear_tree=True`) were tested first and confirmed iter51's configuration as a robust local
optimum along those axes before this new lever was found. Round 18 then resweept the two FM-side
hyperparameters analogous to `learning_rate` (embedding dimension `k`, iter60; FM's own
`learning_rate`, iter61) plus BPR's negative-sampling weight (iter62), all three REJECTed
(iter61 found a real standalone FM gain that did not survive re-blending with the GBM). See
[`experiments/iter55_learning_rate_sweep/RESULT.md`](experiments/iter55_learning_rate_sweep/RESULT.md)
for the full standalone and blend results.

**Round 19 (iter63) found a genuinely new feature and became the final submitted model through
Round 19** (superseded by Round 20 below). Every feature set since iter24 has carried
`decay_tab_3`, a decayed per-tab positive **count**, but never the matching decayed-total
denominator needed to turn it into a Laplace-smoothed
**rate**, the same count→rate upgrade that already won earlier in the project (the iter16-era
`decay_rate` feature). Extending the pipeline to track that denominator and replacing the count
with the rate (`decay_tab_rate_3`) gave a real, causally-verified, 5-seed-confirmed standalone GBM
gain (mean valid +0.00107, 5/5 seeds, test +0.00098 mean) that (unlike iter61's FM-side gain)
propagated cleanly through to the blend level: re-blending with the unchanged iter38 FM ensemble
at a re-swept optimum (14% weight on FM, 86% on the GBM) gave **valid 0.67606, test 0.65955**, the
final submitted model through Round 19. See
[`experiments/iter63_decay_tab_rate/RESULT.md`](experiments/iter63_decay_tab_rate/RESULT.md) for
the full causality-verification, sweep, 5-seed-confirmation, and blend detail.

**Round 20 (merge with teammate yixi's independent branch) found a larger, genuinely new gain
and became the current final submitted model.** Yixi worked a separate research track on the
same problem from `origin/main`, adding XGBoost as a third model family alongside this project's
FM and LightGBM, plus several of her own feature/objective refinements (chained through
`iterYIXI1`-`iterYIXI10`): the 5-day user-decay pair transferred to XGBoost, a lower learning
rate for XGBoost, ranking-objective-aligned LightGBM retuning, a causal 5-day historical
watch-depth decay feature, and a native `upload_type` categorical. Her final within-user-percentile
blend (10% FM / 52% LightGBM / 38% XGBoost) reuses this project's own FM code unchanged
(confirmed by reading her `blend.py`: it calls this file's `train_one_fm`) and already includes
iter63's `decay_tab_rate_3` feature in her LightGBM columns, a genuine merge, not two disjoint
models. Rather than trust her `RESULT.md` claim directly, it was independently re-verified from
raw CSVs first: every component (LightGBM, XGBoost, FM) and the final blend matched her claimed
numbers to 8 decimal places (see
[`experiments/iterMERGE1_verify_yixi10/RESULT.md`](experiments/iterMERGE1_verify_yixi10/RESULT.md))
before being promoted: **valid 0.69943, test 0.68432**, a +0.0234 valid / +0.0248 test gain over
iter63. A follow-up merge-track experiment
([`experiments/iterMERGE2_decay_axis_transfer/RESULT.md`](experiments/iterMERGE2_decay_axis_transfer/RESULT.md))
tested transferring this project's own untried decay-rate feature axes (author-popularity,
duration-bucket, hour-of-day) onto yixi's richer LightGBM/XGBoost representation; all three
were clean nulls (exact-zero delta, no split ever used), extending an existing 2/2-null pattern
(tag/`num_leaves=2` harness) to 4/4 nulls across two independent harnesses. The joint
research-merge effort continues past this promotion in
[`experiments/MERGE_LEDGER.md`](experiments/MERGE_LEDGER.md), tracked separately from this
project's own [`experiments/LEDGER.md`](experiments/LEDGER.md) and yixi's stale
[`YIXI_SUMMARY.md`](YIXI_SUMMARY.md) (which only covers her first four experiments,
`iterYIXI1`-`iterYIXI4`) to avoid concurrent-write conflicts.

**Round 21 (merge-track continuation, plus a third teammate's independent verification lane)
did not change the submitted model, but closed the remaining search space and corroborated a
key finding twice over.** Nine merge-track experiments in total
([`experiments/MERGE_LEDGER.md`](experiments/MERGE_LEDGER.md)) tried linear reweighting of a
4th blend component, per-component calibration, nonlinear meta-learner stacking, decay-feature
transfer across the two research lines, and CV-regularized weight selection; all closed REJECT
except one genuine finding, a valid/test crossover from adding this project's own GBM as a 4th
blend component (+0.00038 valid, -0.0004 to -0.0009 test at every weight checked), confirmed by
two independent methods (5-fold user-level CV re-selection, and a separately-implemented
stacking meta-learner under honest out-of-fold evaluation) and left as a documented, not-promoted
finding, detailed in the README's "Limitations" section. Separately, before a third teammate
(Xuxia)'s own code arrived, her three hypotheses were independently reimplemented from her
self-report alone as `iter65_segment_blend`, `iter66_calibrated_blend`, and
`iter67_multitask_gbm` (`experiments/LEDGER.md` Round 21). Her isotonic-calibration-collapse
result matched this independent reimplementation to the digit: GBM standalone valid
0.67168 to 0.54189, exactly 37 calibrated levels, on two separately-written codebases. Her actual
code then merged in as ground truth (`XUXIA_SUMMARY.md`, `experiments/iterXUXIA1`-`iterXUXIA3`):
per-segment blend alpha, rank/calibrated fusion, and GBM-native multi-task stacking via OOF
auxiliary features, all three REJECT, none touching the submitted model. Combined with the 80
main-track and 11 Yixi-track iterations and the 9 merge-track experiments above, this brings the
project to **103 logged iterations across four tracks**, all summarized in
[`RUN_LOG_SUMMARY.md`](RUN_LOG_SUMMARY.md). The submitted model remains the Round 20 yixi10
3-model blend: **valid 0.69943440, test 0.68432260**.

Code: [`experiments/iter27_triple_fusion/data_ext.py`](experiments/iter27_triple_fusion/data_ext.py),
[`experiments/iter27_triple_fusion/train.py`](experiments/iter27_triple_fusion/train.py) (FM base
config, per seed), [`experiments/iter38_seed_ensemble/driver.py`](experiments/iter38_seed_ensemble/driver.py)
(FM ensembling), [`experiments/iter44_gbm_native_features/train.py`](experiments/iter44_gbm_native_features/train.py)
(GBM-native model), [`experiments/iter51_linear_tree/train.py`](experiments/iter51_linear_tree/train.py)
(`linear_tree=True` variant), [`experiments/iter55_learning_rate_sweep/train.py`](experiments/iter55_learning_rate_sweep/train.py)
(`learning_rate=0.10` variant, reuses iter51's `run()`),
[`experiments/iter63_decay_tab_rate/data_ext.py`](experiments/iter63_decay_tab_rate/data_ext.py)
(`decay_tab_rate_3` feature), [`experiments/iter63_decay_tab_rate/train.py`](experiments/iter63_decay_tab_rate/train.py)
(GBM trained on the extended feature set), [`experiments/iter63_decay_tab_rate/blend.py`](experiments/iter63_decay_tab_rate/blend.py)
(the FM+GBM blend, superseded as the final model by the Round 20 blend below,
kept for provenance), [`experiments/iterYIXI10_video_metadata/features.py`](experiments/iterYIXI10_video_metadata/features.py)
(yixi's chained causal feature frames: 5-day user-decay, causal watch-depth
history, `upload_type`), [`experiments/iterYIXI5_xgboost_optimization/results.json`](experiments/iterYIXI5_xgboost_optimization/results.json)
(tuned XGBoost config), [`experiments/iterMERGE1_verify_yixi10/verify.py`](experiments/iterMERGE1_verify_yixi10/verify.py)
(independent from-scratch reproduction of the yixi10 blend), [`make_submission.py`](make_submission.py)
(current final end-to-end reproduction: the yixi10 3-model blend).

## Reproducing the result

```bash
# from the repo root, with KuaiRand-Pure/data/ already present (see README.md for download)
pip install lightgbm xgboost pandas   # new dependencies for the GBMs; the FM/BPR line stays numpy-only
python3 make_submission.py submission.csv
```

This rebuilds yixi's causal feature frames (5-day user-decay, causal watch-depth history,
`upload_type`), trains her tuned XGBoost ranker and refined LightGBM ranker (a few seconds each,
CPU) and iter27's exact FM configuration at 5 seeds (~25s each on CPU, one core, ~2 minutes
total, still well within the official baseline's resource profile), blends all three via
within-user-percentile normalization at the confirmed weights (10% FM / 52% LightGBM / 38%
XGBoost), evaluates on valid/test, writes `submission.csv` in the format `submit.py` requires,
then self-validates that file with `submit.py`'s own `read_submission` alignment check.

To reproduce the full 5-seed result reported above directly against the experiment harness:

```bash
cd experiments/iter27_triple_fusion
python3 driver.py   # or see results.json for the already-recorded 5-seed run
```

To validate/score any submission CSV independently:

```bash
python3 submit.py --check submission.csv
python3 submit.py --score --split valid submission.csv   # if scoring against valid locally
```

## How we got here

Every iteration (hypothesis, method, harness-fidelity check, results, verdict) is logged in
[`experiments/LEDGER.md`](experiments/LEDGER.md). Highlights:

- **Directions that worked**: switching the loss from pointwise to pairwise BPR (biggest single
  jump), recency-decay features over raw historical aggregates, activity-weighted user sampling,
  and constant retuning on top of those.
- **Directions we tried and closed with a documented reason** (so they aren't retried):
  - *Multi-task learning* (auxiliary heads for `is_click`/`is_like`/`is_follow`/`is_comment`/
    `is_forward`): tried twice, with two different architectures (shared raw score, then
    shared-embedding-only per-task heads). Both regressed monotonically with auxiliary weight.
    The second attempt included a from-scratch gradient derivation that was verified against
    finite-difference numerical gradients before any real training run (caught and fixed a real
    bug in that process). Diagnosis: pushing gradients from base-rate-calibrated pointwise
    auxiliary losses into the shared embedding table conflicts with the rank-invariant BPR
    objective, a structural conflict, not an implementation bug.
  - *Model capacity* (FM embedding dimension `k` ∈ {8, 16, 24, 32}): flat; `k=16` (the default
    since iteration 1) is already at the optimum for this feature/sampling configuration.
  - *DeepFM* (in place of FM): explored across several iterations; higher seed-to-seed variance
    with no reliable valid-set gain over the FM+BPR line.
- **Resource usage**: CPU-only (numpy), no GPU at any point. Per-round agent-dispatch counts
  (used as a token-cost proxy) are logged in LEDGER.md's "Resource usage" table.
- **Round 13 (post-convergence)**: iteration had already converged (3 consecutive non-improving
  rounds under a pre-declared ε=0.002, N=3 rule) after Round 12, but was explicitly reopened on
  request to keep pushing score. Two further directions were tried:
  - *Score-level ensembling* (iter38, **PROMOTE**): averaging sigmoid-transformed predictions
    across the 5 already-trained, already-validated seeds of iter27's config, rather than just
    averaging their metrics. This is a distinct axis from any feature/loss/hyperparameter change:
    the 5-seed std on iter27 (~0.0007-0.0008) is pure random-init noise, and ensembling reduces it
    directly. Bought +0.0020 valid / +0.0030 test for ~5x train/inference compute (still under
    2 minutes total, CPU-only) and no new model risk, since every seed it combines was already
    individually confirmed. This became the final submitted model.
  - *Listwise (grouped-softmax) loss* (iter39, **REJECT**): the one loss-function alternative to
    pairwise BPR that the starter kit's own README names as untried. Implemented with a
    from-scratch gradient (verified against finite-difference numerical gradients before any real
    training run), then swept across a 10x learning-rate range. All settings peaked at epoch 1 and
    monotonically worsened every epoch after, diagnosed as the per-step random resubsampling of a
    capped negative set per group producing a moving, high-variance objective, unlike BPR's stable
    pairwise draw. Best achievable point was roughly at parity with (not better than) the BPR
    baseline, and only by stopping almost immediately, a fragile, non-robust result, not a real
    gain. Documented and closed; iter27+iter38 remained the selected model at the time.
- **Round 14 (further post-convergence iteration, on request)**: two earlier attempts to fold a
  GBM into the pipeline (iter41: LightGBM, iter43: CatBoost) had both underperformed FM when
  forced through FM's own bucketed feature encoding, and were closed as REJECT rather than as
  "GBMs don't work here": the diagnosis pointed at the encoding, not the model family, and per
  the standing instruction not to give up on non-neural/open-source-model directions, this was
  revisited. Giving the GBM its own un-bucketed encoding of the same causal features (iter44)
  closed the gap and then reversed it, with an unusually large gain that triggered a longer
  verification chain than usual (tie-artifact check, feature-confound ablation, seed robustness,
  date-shift robustness) before being trusted: see
  [`experiments/iter44_gbm_native_features/RESULT.md`](experiments/iter44_gbm_native_features/RESULT.md).
  Blending it with the unchanged iter38 FM ensemble gave a further, genuine gain from real
  model-family diversity, becoming the final submitted model through Round 15 (**PROMOTE**, valid
  0.66473/test 0.65197).
- **Round 15 (post-convergence, on request)**: six further directions were tried against iter44's
  exact pipeline: CatBoost as a second GBM library on the native encoding, an extreme-low-capacity
  hyperparameter depth sweep, a logistic stacking meta-learner over the FM/GBM/CatBoost scores, a
  time-of-day feature, monotonic constraints on the engagement-rate features, and GOSS boosting;
  all six landed as clean, well-diagnosed **REJECT**s, closing several previously-open questions
  (CatBoost's native-encoding ceiling, the GBM hyperparameter/boosting-algorithm search space at
  `num_leaves=2`, whether stacking beats a fixed blend weight) without finding a new gain. See
  [`experiments/LEDGER.md`](experiments/LEDGER.md)'s Round 15 section for detail on each.
- **Round 16 (post-convergence, on request)**: `linear_tree=True` (**PROMOTE**), see the
  `linear_tree` paragraph above. The first genuine gain found across seven consecutive Round
  15/16 methods, and the final submitted model through Round 16.
- **Round 17 (post-promotion, on request to keep testing)**: three follow-up retests of whether
  `linear_tree=True`'s structural change reopens previously-closed directions (capacity,
  `linear_lambda` regularization, hour-of-day feature) all landed **REJECT**, confirming iter51's
  configuration as a robust local optimum; a fourth, genuinely new hypothesis, a `learning_rate`
  resweep under `linear_tree=True` (iter55), found a real further gain (**PROMOTE**). See the
  iter55 paragraph above and [`experiments/LEDGER.md`](experiments/LEDGER.md)'s Round 17 section
  for detail on each.
- **Round 18 (FM-side search, on request after the GBM hyperparameter space was confirmed
  exhausted)**: resweeping the two FM hyperparameters most analogous to the levers that had just
  paid off on the GBM side: embedding dimension `k` (iter60) and `learning_rate` (iter61), plus
  BPR's negative-sampling weight (iter62), all landed **REJECT**; iter61 found a real standalone FM
  gain (+0.00108 valid, 5-seed confirmed) that notably did *not* survive re-blending with the GBM
  (blend valid gain below the promotion threshold, test moved the wrong way), an instructive
  counterexample to "any real standalone gain promotes." See
  [`experiments/LEDGER.md`](experiments/LEDGER.md)'s Round 18 section for detail on each.
- **Round 19 (new feature-engineering angle, on request after Round 18's FM-side search was
  exhausted)**: iter63's decayed per-tab rate feature (**PROMOTE**, see the iter63 paragraph
  above) became the current final submitted model. Two other angles were ruled out first without
  a full experiment: item-side popularity features (already closed in iter12, redundant with the
  FM's learned id embeddings) and time-of-day (already closed in iter48). See the iter63 paragraph
  above and [`experiments/LEDGER.md`](experiments/LEDGER.md)'s Round 19 section for detail.

## Limitations / future work

- **User-history sequence modeling** (DIN/SIM-style attention over a user's raw interaction
  sequence) was scoped in the original plan but not reached before convergence; the
  recency-decay features are a lightweight proxy for recency signal, not a learned sequence
  model, and remain the most likely next lever.
- **Watch-time-as-censored-regression** (the CWM line of work referenced in the starter kit) was
  deprioritized from the start due to its `torch==1.6.0` dependency risk this close to a hard
  deadline; not attempted.
- Multi-task learning and additional model capacity are both closed directions for this feature
  set specifically; a materially different feature set could reopen either.
- **Listwise (grouped-softmax) loss** (iter39) was implemented, gradient-verified, and swept, but
  closed as a REJECT. See Round 13 above. A variant using ALL of a user's negatives per group
  (instead of the capped `M_max=16` subsample that iter39 diagnosed as the source of its
  instability) was identified as the natural follow-up but not attempted, since it would push
  per-group sizes into the hundreds for high-degree users and materially change the compute
  profile, and the existing diagnosis already explained the observed failure mode well enough not
  to warrant it ahead of the already-confirmed iter38 ensemble win.
- **Further ensemble variants**, more than 5 seeds, or weighted (rather than uniform) averaging
  across seeds, were not explored for the FM line. Diminishing returns are likely (the gain from
  1→5 seeds was already a reduction of pure random-init variance, not new signal), and this was
  since confirmed on the GBM side too: a 5-seed GBM ensemble (iter44) scored valid 0.66142/test
  0.64770, no meaningful gain over the single seed (0.66135/0.64794), the final model uses a
  single GBM seed for that reason.
- **A third GBM library on the native-feature representation** (CatBoost) was since tried
  (iter45, Round 15): it underperformed LightGBM on the same native encoding and added nothing
  in a 3-way stack (iter47), closed, not a further lever. See
  [`experiments/iter45_catboost_native/RESULT.md`](experiments/iter45_catboost_native/RESULT.md).
- **The valid/test gap widens as GBM tree capacity shrinks toward `num_leaves=2`** (~0.013 at
  num_leaves=2 vs. ~0.002-0.003 at num_leaves=7, stable across seeds and confirmed present under
  a date-shifted split too), documented in
  [`experiments/iter44_gbm_native_features/RESULT.md`](experiments/iter44_gbm_native_features/RESULT.md)
  rather than hidden. `num_leaves=7` (valid 0.64632/test 0.64412 standalone) is a tighter-margin
  fallback if this gap is a concern.

## Setup

See [`README.md`](README.md) for environment/data-download instructions. The FM/BPR line
(iter1-iter39) is Python 3.9+ and numpy only, no other dependencies. **The final submitted model
(Round 20, the yixi10 3-model blend) additionally requires `pandas`, `lightgbm`, and `xgboost`**
(all pip-installable) for its two GBM components: see `pip install lightgbm xgboost pandas` in
"Reproducing the result" above.
`make_submission.py` and everything under `experiments/` assume `KuaiRand-Pure/data/` is present
at the repo root, per that README.

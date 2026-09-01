# Merge Ledger — reconciling yixi's branch with local research

Started 2026-09-01 ~01:45 SGT, after merging `origin/main` (yixi's two commits,
`e3d97ac` + `98ee890`, iterYIXI1-11) into local `main` on top of local
iter64-81 (merge commit `9e3b618`). Tracks the autonomous multi-agent effort
to combine both research tracks. Kept separate from `LEDGER.md` (this
project's own iter1-81 history) and `YIXI_SUMMARY.md` (stale, only covers
yixi's first commit) to avoid concurrent-write conflicts with either.

## State at merge time

**Our best (iter63, currently submitted):** GBM (`rate_only`, num_leaves=2,
linear_tree) + iter38 5-seed FM/BPR sigmoid-mean ensemble, alpha=0.14 blend.
valid=0.67606449, test=0.65955.

**Yixi's best (iterYIXI10, NOT yet independently reproduced by us):**
10% FM / 52% LightGBM / 38% XGBoost, within-user-percentile-normalized blend.
valid=0.69943440, test=0.68432260. Built by chaining, on top of her
iterYIXI1 XGBoost-native baseline: 5-day user-decay transfer (YIXI4),
lower-rate XGBoost retrain (YIXI5), ranking-objective-tuned LightGBM (YIXI7),
causal 5-day watch-depth decay feature (YIXI9), native `upload_type`
categorical (YIXI10). Her FM component is confirmed (read `blend.py`) to
call this repo's own `submission.train_one_fm` / sigmoid-mean ensemble —
**same shared FM code we use**, not a divergent implementation. Her
LightGBM reference columns already include our `decay_tab_rate_3`
(iter63's own feature), so some merging already happened organically.
Her discipline matches ours: valid-only selection, PRELIMINARY_DELTA=0.0003,
PROMOTION_DELTA=0.001, 5-seed confirmation, harness-fidelity checks before
trusting any new number.

**Not yet tried by either side** (candidate merge directions):
- Our iter69/71/72/73 decay-rate transfers (tag, author, duration-bucket,
  hour axes) never tested against her LightGBM/XGBoost reference columns —
  she only has `decay_rate_5`/`decay_act_5`/`decay_tab_rate_3`/`lastk_rate`/
  `gap`/`duration_ms`.
- Our iter66 calibrated/isotonic blend approach never tried on her 3-way
  ensemble (she only swept weights + monotonic transforms in YIXI8, all
  REJECT against within-user-percentile).
- Our iter65 segment blend / iter67 multitask heads never tested with her
  XGBoost or improved LightGBM in the loop.

## Round 1 (dispatched 2026-09-01 ~01:50 SGT)

- **iterMERGE1_verify_yixi10**: independently reproduce iterYIXI10's full
  pipeline end-to-end (not just trust her RESULT.md) — required before this
  number can be treated as a real candidate for anything, per this project's
  own causality-verification discipline.
- **iterMERGE2_decay_axis_transfer**: transfer our iter71 (author-rate),
  iter72 (duration-bucket-rate), iter73 (hour-rate) causal decay features
  into yixi's current-best LightGBM and XGBoost reference feature sets,
  individually and combined, re-running her exact blend/weight-search
  machinery. First genuinely untried merge direction.

## Round 1 results

**iterMERGE1_verify_yixi10 — VERIFIED.** Independently retrained all three
components (LightGBM, XGBoost, FM) from raw CSVs, no reuse of yixi's cached
artifacts, seed 0. Exact match to her claim on every number (delta
+0.00000000 throughout): LightGBM 0.68834144, XGBoost 0.66755420, FM
0.63987792, blend valid=0.69943440, test=0.68432260. This is now the
project's confirmed best-known number, +0.02337 valid / +0.02477 test over
our submitted iter63. See `experiments/iterMERGE1_verify_yixi10/RESULT.md`.

**iterMERGE2_decay_axis_transfer — REJECT (clean nulls, all 3 axes).**
Harness-fidelity confirmed exactly first (LGB=0.6768913269,
XGB=0.6697614193/0.6675541997, full blend=0.69943440). Then added
author-popularity, duration-bucket, and hour-of-day decay-rate features
(same Laplace-smoothed half-life-3 construction as `decay_tab_rate_3`,
sourced from `iter71`/`iter72`/`iter73`) individually to yixi's LightGBM and
XGBoost reference columns — 6 single-seed ablations, all exact-zero delta;
neither model ever split on any of the 3 new columns. Extends this
project's decayed-rate-axis-generalization finding to 4/4 nulls (tag,
author, duration-bucket, hour) confirmed on two independent harnesses (our
own `num_leaves=2` GBM, and now yixi's richer LightGBM+XGBoost reference).
See `experiments/iterMERGE2_decay_axis_transfer/RESULT.md`.

**Round 1 conclusion:** best-known number stands at valid 0.69943440 / test
0.68432260 (yixi's YIXI10 blend, independently verified). User approved
promoting this as the new submission candidate (2026-09-01 ~02:05 SGT).
Merge loop continues past this promotion — Round 2 to follow, targeting
directions not yet tried by either side (iter65 segment blend / iter66
calibrated blend applied to the 3-model ensemble; iter67 multitask signal;
any interaction between our iter64-81 findings and her XGBoost component).

## Round 2 (dispatched 2026-09-01 ~02:25 SGT, ~9h35m to deadline)

Note: `iter65_segment_blend`/`iter66_calibrated_blend`/`iter67_multitask_gbm`
were already clean REJECTs against the 2-model iter63 blend (independently
confirmed, see their RESULT.md). Re-running them verbatim on the 3-model
blend is low expected value, so Round 2 targets genuinely new-to-both-sides
directions instead:

- **iterMERGE3_seed_ensemble_gbm**: currently only FM is 5-seed ensembled;
  LightGBM and XGBoost (yixi's reference configs) are single-seed (seed 0).
  Seed-ensemble both (seeds 0-4, sigmoid-mean or plain mean, whichever the
  existing FM ensembling convention uses) the same way, re-sweep the 3-way
  blend weights on valid. Cheap (no new features), mechanical, plausible
  variance-reduction gain, untried by either side.
- **iterMERGE4_multitask_aux_3model**: iter67's multitask aux-label columns
  (`is_like`/`is_follow`/`is_comment`/`is_forward` predictions as extra GBM
  input columns) were REJECT on the 2-model iter63 blend. Re-test on yixi's
  richer LightGBM/XGBoost reference columns (already includes
  `decay_tab_rate_3`, `upload_type`, causal watch-depth history) — a
  different feature-interaction regime than iter67's. One more shot given
  it's cheap (reuses iter67's `aux_labels.py` recovery code) and combined
  with XGBoost specifically has never been tried.

Both agents instructed: use `Bash run_in_background: true` for any training
launch (not bare background shell), append results to `results.json`
incrementally, write partial `RESULT.md` after harness-fidelity + after
sweep (not only at the end) — per this session's two prior background-
tracking mixups (iterMERGE1/iterMERGE2).

## Round 2 results

**iterMERGE3_seed_ensemble_gbm — REJECT.** Harness-fidelity confirmed
exactly first. Seed-ensembling LightGBM+XGBoost (seeds 0-4, mirroring FM's
existing 5-seed convention) made the blend worse: production weights gave
valid 0.69868320 (-0.00075 vs. reference), and a 49-combo weight re-sweep's
best still only reached 0.69878483 (-0.00065). Root cause: XGBoost is fully
seed-invariant under the current tuned config (all 5 seeds gave identical
0.66755420, confirmed directly — likely `subsample`/`colsample_bytree`
near 1.0), and LightGBM's seed-0 draw already outperforms its own 5-seed
average (seed 0 was a favorable draw; averaging traded it away). Current
production blend (single-seed LGB/XGB, 5-seed FM, 10/52/38, valid
0.69943440/test 0.68432260) remains best-known. See
`experiments/iterMERGE3_seed_ensemble_gbm/RESULT.md`.

**iterMERGE4_multitask_aux_3model — REJECT.** Harness-fidelity confirmed
exactly first. Re-verified iter67's `orig_idx` alignment (0 mismatches on
all splits, yixi's frames). Adding 4 aux-label prediction columns
(is_like/is_follow/is_comment/is_forward) to LightGBM: 0 feature
importance, exact-zero delta (same null as iter67). Adding them to
XGBoost: actively harmful, standalone valid 0.66755420→0.65837336
(-0.00918), blend 0.69943440→0.69567251 (-0.00376) — XGBoost's capacity
let it fit noise in the low-prevalence (0.1-1.9%) aux predictions rather
than signal. Worse outcome than iter67's original null. See
`experiments/iterMERGE4_multitask_aux_3model/RESULT.md`.

**Round 2 conclusion:** both directions REJECT. Best-known number remains
unchanged at valid 0.69943440 / test 0.68432260 (yixi10 blend, already
submitted). Merge loop continues into Round 3.

## Round 3 (dispatched 2026-09-01 ~02:50 SGT, ~9h10m to deadline)

Context: our own local track (`iter64`-`iter81`, 18 experiments since iter63)
is now fully exhausted — every one REJECT, no PROMOTE since iter63 itself.
Round 2's two merge-specific ideas were also both REJECT. Simple
feature-column-transfer and multitask/seed-ensemble ideas appear to be a
dry well against yixi's current feature set. One structurally different
direction remains untried: combining iter63's own distinct GBM (different
feature set: `decay_rate_2.5`/`decay_act_2.5`/`decay_tab_rate_3`/`last1`/
`lastk_rate`/`gap`, `num_leaves=2` linear-tree, `rate_only` variant — vs.
yixi's richer LightGBM columns) as a 4th ensemble member alongside FM/LGB/
XGB, rather than transferring individual columns between them.

- **iterMERGE5_four_model_blend**: add iter63's `rate_only` GBM (5-seed,
  matching FM's ensembling convention) as a 4th component to the blend,
  weight-search the 4-way combination on valid only. Single agent this
  round (conservative token use given the thinning prior after 4
  consecutive REJECTs).

## Round 3 results

**iterMERGE5_four_model_blend — PRELIMINARY (not promoted).** Harness-fidelity
confirmed exactly for the 3-model reference (all 5 checks delta 0.0) and for
iter63's own `rate_only` standalone (0.6716787219047546 — note: the dispatch
note's stated reference, 0.6768913269042969, turned out to be a *different*
yixi-chain constant; corrected and documented in `RESULT.md`). The initial
run hit an apparent row-misalignment `AssertionError` comparing yixi's
frames against iter63's own DataFrame — investigated rather than
worked around, and found to be a **false alarm**: iter63's own `user_id`
column is cast to a train-only-categories categorical dtype
(`train.py:88-91`), so any valid/test user unseen in train legitimately
casts to NaN — a cold-start encoding choice, not a row-order bug. Re-checked
against the trusted uncast identity arrays: **0 mismatches** on both splits
(124,909 valid + 170,588 test rows). Position-based combination confirmed
safe; no `orig_idx` join was actually needed once the comparison used the
correct arrays (the `orig_idx` technique from iterMERGE4 was prepared as a
fallback but the direct fix made it unnecessary).

Adding iter63's `rate_only` GBM (5-seed sigmoid-mean ensemble, valid
0.6714168/test 0.6533625 standalone) as a 4th blend member: best 4-way
weight point (`fm=0.072, lgb=0.5184, xgb=0.3296, i63=0.08`, found via
coarse-then-fine valid-only grid search) reached valid **0.69981837**
vs. the 3-model reference **0.69943440** — delta **+0.00038397**. This
clears `PRELIMINARY_DELTA=0.0003` but is well short of
`PROMOTION_DELTA=0.001` (~38% of the way). Given the borderline size, a
5-seed confirmation pass was run (per protocol, reserved for
surprising/borderline results): the gain holds up **seed-by-seed**, not
just as a sigmoid-mean-ensemble artifact — all 5 individual iter63 seeds
independently clear +0.0003 at the same weight point (range +0.00033 to
+0.00045) — and holds as a genuine **plateau**, not a grid spike — the
top-3 neighboring fine-grid points all independently clear the preliminary
bar too. However, test primary is consistently **below** the 3-model
reference at every point checked (-0.0004 to -0.0009, reported for the
record only, never used for selection) — a stable valid-only gain with no
visible test transfer. Left as PRELIMINARY rather than escalated: real and
robust by both checks performed, but too small relative to the promotion
bar and directionally inconsistent with test to justify disturbing the
current submission. See `experiments/iterMERGE5_four_model_blend/RESULT.md`.

**Round 3 conclusion:** best-known *promotable* number remains unchanged at
valid 0.69943440 / test 0.68432260 (yixi10 blend, already submitted).
iterMERGE5 is the first Round-2/3 direction to clear the preliminary bar at
all (Round 2's two directions were flat/negative), but confirmed too small
and test-inconsistent to promote. Genuinely new structural finding this
round: iter63's `train.py` train-only-categories cast on `user_id` (a
legitimate cold-start encoding choice, now documented) can look like a
cross-pipeline row-misalignment bug if compared naively — worth remembering
for any future cross-pipeline comparison work in this codebase.

Time check: ~03:21 SGT, ~8h39m to deadline (2026-09-01 12:00 SGT hard
cutoff). Four consecutive merge-round results now (MERGE2 REJECT, MERGE3
REJECT, MERGE4 REJECT, MERGE5 PRELIMINARY-not-actionable) — the "structural
merge" prior (new feature/model combinations between the two tracks) is
thinning fast, though MERGE5 shows the well hasn't fully run dry. Per "run
until told to stop," continuing to Round 4 with the remaining untried
structurally-different directions from the original candidate list (iter65
segment blend / iter66 calibrated blend applied to the 4-model or 3-model
ensemble — never tried on either yixi's XGBoost component or the now-
confirmed 4-model iterMERGE5 combination).

## Round 4 (dispatched 2026-09-01 ~03:25 SGT, ~8h35m to deadline)

Context: with ~8.5h left and a thinning prior, staying conservative on
token/agent count. iterMERGE5's PRELIMINARY result is itself now a valid
base to try calibration on (its valid/test crossover is exactly the kind
of miscalibration-shaped pattern `iter66_calibrated_blend`'s isotonic
approach was built for) — more promising than blindly repeating iter65's
segment-blend idea, which had no particular reason to interact with the
new 4th component. Single agent, single hypothesis:

- **iterMERGE6_calibrated_4model**: apply `iter66_calibrated_blend`'s
  isotonic-regression calibration approach (fit per-component isotonic
  maps on a held-out slice of valid, calibrate before blending) to the
  iterMERGE5 4-model (FM/LGB/XGB/i63) combination — untried by either
  side, and specifically targets the valid/test-crossover pattern
  iterMERGE5 just surfaced (a plausible sign of a blend-weight overfit to
  valid that calibration is designed to reduce). Reuses iterMERGE5's
  already-verified row-alignment/harness code directly rather than
  rederiving it.

## Round 4 results

**iterMERGE6_calibrated_4model — REJECT (large regression, not close).**
Harness-fidelity confirmed exactly (7 checks including the raw 4-model
blend at iterMERGE5's best weights, all delta 0.0). Applying isotonic
regression (`iter66_calibrated_blend`'s technique) per-component to
LGB/XGB/i63 before blending collapsed each model's score distribution to
35-68 distinct output levels (vs. 15,776-122,612 unique raw valid-split
scores) — a coarse step function that destroys the fine-grained
within-user rank ordering GAUC/nDCG@5 depends on. Standalone valid dropped
-0.130 to -0.149 per component (FM, which has much lower native
cardinality, was a near-no-op at +0.00003). Best calibrated 4-way blend
across two blending variants (percentile-of-calibrated vs. direct) and a
425-combo combined weight search: valid 0.63862032 vs. 3-model reference
0.69943440 — delta **-0.06081408**, two orders of magnitude past the
promotion threshold in the harmful direction. No 5-seed confirmation
warranted (not borderline). See
`experiments/iterMERGE6_calibrated_4model/RESULT.md`. Note: `iter66`'s own
calibration success was on a much lower-cardinality *blend* score, not
high-cardinality raw tree-model scores directly — this experiment shows
the technique does not transfer to that setting.

**Round 4 conclusion:** best-known promotable number remains unchanged at
valid 0.69943440 / test 0.68432260. Best PRELIMINARY (not promoted) finding
remains iterMERGE5's raw 4-model blend (valid 0.69981837/test 0.68367088,
delta +0.00038397, well short of the 0.001 promotion bar). Five merge
rounds in (MERGE1 verify, MERGE2-4 REJECT, MERGE5 PRELIMINARY, MERGE6
REJECT) — the calibration and column-transfer directions are now closed;
remaining unexplored territory is thin (iter65 segment-blend applied to
the 4-model combination is the last item from the original candidate
list). Time check: 06:50 SGT, ~5h10m to deadline (2026-09-01 12:00 SGT
hard cutoff). Given the thinning prior and shrinking time budget, continuing
the loop per "run until told to stop," but scoping remaining rounds even
more conservatively (single narrow hypothesis per dispatch, reusing
verified harness code rather than re-deriving).

## Round 5 (dispatched 2026-09-01 ~06:55 SGT, ~5h05m to deadline)

`iter65_segment_blend` (per-tab / per-activity-tertile alpha) was already
independently confirmed REJECT *twice* against the 2-model iter63 blend
(0/5 seed-wins on both segmentations, by this project and by teammate
Xuxia separately) — re-running the same mechanism on the 4-model blend is
low expected value for the same reason Round 2 skipped iter65/66/67
verbatim, so it is skipped again here rather than dispatched. All five
rounds so far have only tried **linear** combinations of the 4 component
scores (fixed weighted-sum, whether raw or isotonic-calibrated). One
mechanically distinct, genuinely untried direction remains cheap enough
for the remaining time budget:

- **iterMERGE7_stacked_meta_blend**: replace the fixed-weight linear blend
  with a small logistic-regression (or shallow GBM, whichever the agent's
  own held-out check finds less prone to overfitting) meta-learner over
  the 4 components' within-user-percentile scores, fit via k-fold
  out-of-fold stacking *within valid itself* (never touching train/test
  for the meta-fit) to avoid trivially overfitting a 124,909-row single
  split. Tests whether a nonlinear/interaction combination of FM/LGB/XGB/
  i63 beats the best linear weight point found in iterMERGE5. Single
  agent, given time budget (~5h to deadline) and the thinning prior.

## Round 5 results

**iterMERGE7_stacked_meta_blend — REJECT.** Harness-fidelity confirmed
exactly (7 checks incl. row alignment and the raw 4-model blend sanity
check, all delta 0.0). Replaced the fixed-weight linear blend with a
5-fold, user-level out-of-fold (OOF) stacked meta-learner over the 4
within-user-percentile component scores — logistic regression (swept
C=0.001-3.0, best 0.69856703) and a shallow GBM (swept num_leaves=2-4,
best 0.69904566 at num_leaves=2). Best OOF valid (0.69904566, the only
honest non-leaking selection number) is **worse** than both the 3-model
reference (0.69943440, delta -0.00038874) and iterMERGE5's linear optimum
(0.69981837, delta -0.00077271) — a nonlinear combiner finds no exploitable
interaction structure among just 4 already-percentile-normalized,
largely-agreeing component scores. Notable side finding, reported for the
record only (not used for selection): the refit-on-all-valid model's test
score (0.68481672) is actually *higher* than both baselines' test scores
(reference 0.68432260, MERGE5 0.68367088) — the opposite of MERGE5's
valid-up/test-down crossover — but this is a single in-sample-refit
observation with no seed/fold robustness check, and the governing OOF valid
number is already a regression, so the verdict stands as REJECT regardless.
See `experiments/iterMERGE7_stacked_meta_blend/RESULT.md`.

**Round 5 conclusion:** best-known promotable number remains unchanged at
valid 0.69943440 / test 0.68432260. Best PRELIMINARY (not promoted) finding
remains iterMERGE5's linear 4-model blend. Six merge rounds in (MERGE1
verify, MERGE2/3/4/6/7 REJECT, MERGE5 PRELIMINARY) — linear weight-search,
per-component calibration, and nonlinear stacking have all now been tried
on the 4-model combination; none finds anything promotable, and the
underlying signal appears close to saturated at 4 components with this
feature set. Time check: 07:01 SGT, ~4h59m to deadline (2026-09-01 12:00
SGT hard cutoff). Given the now heavily thinned prior (5 of 6 merge-round
results REJECT, the 6th only PRELIMINARY-not-actionable) and shrinking time
budget, continuing per "run until told to stop" but the search space of
genuinely novel, cheap, low-risk merge directions is nearly exhausted —
further rounds should stay narrowly scoped and will likely need to look
beyond simple recombination of the same 4 base components (e.g. genuinely
new causal features, which both tracks' independent searches have already
found hard to come by after their own respective iteration counts).

## Round 6 (dispatched 2026-09-01 ~07:05 SGT, ~4h55m to deadline)

`iterMERGE2_decay_axis_transfer` tested the tag/author/duration-bucket/
hour-of-day decay-rate features against yixi's **rich** `LGB_CANDIDATE_
COLUMNS`/`XGB_COLUMNS` reference sets (30+ columns) and found 4/4 clean
nulls — neither model ever split on the new columns. That test was never
run against iter63's own **minimal** `rate_only` feature set (only 6
columns, `num_leaves=2` linear-tree) — the 4th blend member since
`iterMERGE5`. A `num_leaves=2` tree with only 6 competing features has far
less competition for splits than a 30+-column model; a weak feature that
never wins a split among 30 candidates might still win among 6. Cheap
(reuses `iter63_decay_tab_rate/train.py` and the existing decay feature
construction from `iter71`/`iter72`/`iter73`), mechanically distinct from
every prior merge-round hypothesis (none tested a decay-axis transfer
specifically against i63's own model), single agent given the thinning
prior and time budget.

- **iterMERGE8_decay_axis_on_i63**: add author-popularity, duration-bucket,
  and hour-of-day decay-rate features (same construction as `iterMERGE2`)
  individually to iter63's own `rate_only` LightGBM feature set, retrain
  at 5 seeds each, re-run the 4-model blend weight search (reusing
  `iterMERGE5_four_model_blend/run.py`'s verified harness) if any variant
  shows standalone signal.

## Round 6 results

- **iterMERGE8_decay_axis_on_i63**: REJECT, clean and unambiguous. All 7
  harness-fidelity checks (iter63 standalone, iter63 5-seed ensemble,
  3-model reference, 4-model raw blend at iterMERGE5's best weights, row
  alignment) reproduced at exact `0.0` delta before any new feature was
  introduced. Standalone ablation (seed 0) attaching each of the 3
  decay-rate axes individually to iter63's own minimal 6-column `rate_only`
  set: **all three showed exact-zero valid delta (+0.0000000000) and zero
  new-column split usage** in `feature_importance()` — predictions
  byte-identical to the unmodified baseline for all three variants. Per the
  pre-registered gate (skip 5-seed retrain/blend retest if all axes show
  zero standalone signal), the run correctly stopped after the standalone
  stage. This falsifies the round's motivating hypothesis (fewer competing
  features, ~6-7 here vs. yixi's 30+, might let a weak feature win a split)
  — these three engineered rate features carry no exploitable signal for
  `long_view` beyond what `decay_tab_rate_3`/`decay_rate_2.5`/
  `decay_act_2.5`/`last1`/`lastk_rate`/`gap`/categoricals already encode,
  independent of split-competition level. Combined with `iterMERGE2`'s
  earlier 4/4 null against yixi's rich feature sets, the decay-axis-transfer
  direction (author/duration-bucket/hour-of-day) is now closed across both
  harnesses in this codebase — no further work recommended on this feature
  family without a genuinely different construction (different half-life,
  smoothing scheme, or grouping key). See
  `experiments/iterMERGE8_decay_axis_on_i63/RESULT.md`.

**Round 6 conclusion:** best-known promotable number remains unchanged at
valid 0.69943440 / test 0.68432260 (yixi10 3-model blend, currently
submitted). Best PRELIMINARY (not promoted) finding remains iterMERGE5's
linear 4-model blend (valid 0.69981837 / test 0.68367088). Seven merge
rounds in now (MERGE1 verify, MERGE2/3/4/6/7/8 REJECT, MERGE5
PRELIMINARY-not-actionable) — 6 of 7 rounds since the initial verify have
been clean rejects, and the one non-reject finding is well under the
promotion bar with a valid/test inconsistency. Time check: 2026-09-01
07:18 SGT, ~4h42m to the 2026-09-01 12:00 SGT hard deadline (no late
submissions).

Every cheap, mechanically-distinct recombination of the 4 known components
(raw linear weight search, per-component isotonic calibration, k-fold OOF
nonlinear stacking, decay-axis feature transfer onto the weakest component)
has now been tried and closed. The remaining search space for a *merge*-side
gain that's both novel and cheap enough to fit the shrinking time budget is
very thin. Per "conservative token count" discipline and the explicit
instruction to continue until told to stop, the loop continues, but future
rounds should either (a) target a specific, narrow, still-untested
recombination detail (e.g., a weighted/robust variant of the linear search
that explicitly regularizes against iterMERGE5's valid/test crossover,
which is the one loose thread no round has directly addressed) or (b) stay
dispatched only when a concrete, falsifiable hypothesis exists — not dispatch
for its own sake against an increasingly exhausted 4-component search space.

## Round 7 (dispatched 2026-09-01 ~07:19 SGT, ~4h41m to deadline)

The one loose thread no round has directly addressed: iterMERGE5's linear
4-model optimum and iterMERGE7's stacked meta-blend both show a valid-up/
test-down (or, for MERGE7's non-selection in-sample number, the reverse)
inconsistency between the fine-grid-searched weight point and test — a
classic sign of overfitting a 135-point grid to a single 124,909-row valid
split. Neither prior round asked whether a **coarser, regularized** weight
point (e.g., the grid's best point re-selected via k-fold user-level
cross-validated valid score, not single-split valid) closes this gap and
still clears the promotion bar. This directly probes the specific
overfitting mechanism rather than trying another new combination axis —
mechanically distinct from MERGE5 (single-split grid search) and MERGE7
(nonlinear meta-learner), reuses the exact same verified harness
(`iterMERGE5_four_model_blend/run.py`, `iterMERGE7_stacked_meta_blend`'s
k-fold user-level splitting code), single agent given the time budget.

- **iterMERGE9_cv_regularized_blend**: re-run the linear 4-model weight
  search (same coarse+fine grid as `iterMERGE5`), but score each candidate
  weight point via 5-fold user-level cross-validated mean valid score
  (reusing `iterMERGE7`'s fold-splitting code) instead of single-split
  valid, to find a weight point that is robust rather than overfit to one
  split; report both the CV-selected point's single-split valid (for
  comparability to prior rounds) and its test score, to check whether
  regularizing the selection closes or narrows the valid/test crossover
  gap.

## Round 7 results

- **iterMERGE9_cv_regularized_blend**: PRELIMINARY, not actionable — same
  substantive finding as iterMERGE5, with an important negative
  confirmation added. All 7 harness-fidelity checks (3-model reference,
  iter63 standalone, iter63 5-seed ensemble, 4-model raw blend at
  iterMERGE5's weights, row alignment) reproduced at exact `0.0` delta.
  5-fold user-level CV mean valid AT iterMERGE5's original weight point
  (fm=0.072/lgb=0.5184/xgb=0.3296/i63=0.08) was **0.69979823**, within
  0.00002 of its single-split valid (0.69981837) — this point was already
  CV-robust before any CV-based reselection was attempted. Re-running the
  coarse+fine grid search scored by CV mean valid (instead of single-split
  valid) converged to **the identical weight point** (to 4 decimal places):
  single-split valid 0.69981754 (delta +0.00038314 vs. 3-model reference,
  clears PRELIMINARY_DELTA), CV mean valid 0.69979733, test 0.68366760
  (delta -0.00065500, still regresses vs. reference). **Crossover NOT
  resolved.** This directly answers the round's motivating question: the
  valid/test crossover iterMERGE5 first surfaced is not an artifact of the
  fine grid search overfitting its weight choice to a single 124,909-row
  valid split — there was no overfitting gap to regularize away, since CV
  selection lands on the same point iterMERGE5's naive single-split search
  already found. This points toward a genuine distributional difference
  between the valid and test splits themselves as the real explanation,
  consistent with `iterMERGE7`'s independent finding that a completely
  different nonlinear combiner produces the same crossover pattern under
  honest out-of-fold evaluation. Closes "try a more robust selection
  procedure" as a promising direction for resolving the crossover
  specifically. See
  `experiments/iterMERGE9_cv_regularized_blend/RESULT.md`.

**Round 7 conclusion:** best-known promotable number remains unchanged at
valid 0.69943440 / test 0.68432260 (yixi10 3-model blend, currently
submitted). Best PRELIMINARY (not promoted) finding remains iterMERGE5's
linear 4-model blend point — now doubly-confirmed as CV-robust, but still
well short of the 0.001 promotion bar due to its persistent test-side
regression. Eight merge-track experiments in now (MERGE1 verify,
MERGE2/3/4/6/7/8 REJECT, MERGE5/9 PRELIMINARY-not-actionable, MERGE9
additionally closing the "is it just overfitting?" question about MERGE5's
own finding). Time check: 2026-09-01 07:31 SGT, ~4h29m to the 2026-09-01
12:00 SGT hard deadline (no late submissions).

The merge-track search space is now very thoroughly covered for the
"recombine the same 4 known components" family: raw linear weight search
(MERGE5), per-component calibration (MERGE6), nonlinear stacking (MERGE7),
decay-axis feature transfer (MERGE8), and CV-regularized weight selection
(MERGE9) have all been tried and closed, with two independent lines of
evidence (MERGE7's OOF stacking, MERGE9's CV selection) now agreeing that
the valid/test crossover is intrinsic to this 4-component combination
rather than a fixable selection artifact. Given the shrinking time budget
(~4.5h to deadline) and the now very high cost/expected-value ratio of a
9th recombination attempt against an increasingly closed search space, the
orchestrator is pausing new round dispatch here rather than manufacturing a
further hypothesis for its own sake — consistent with "conservative token
count" discipline, which counsels against dispatching an ~8-10 minute agent
run against a search space this thin purely to keep the loop occupied. Per
the user's explicit "run until told to stop" instruction, the loop remains
open and will resume dispatching on any of: (a) yixi pushing new work to
the shared repo (would need to be checked for), (b) a genuinely new
causal-feature idea distinct from every axis tried so far, or (c) an
explicit user instruction to continue more aggressively. The orchestrator
will periodically re-check for new pushes from yixi and re-evaluate, while
treating the remaining time budget as primarily needed for submission
integrity (the current SUBMISSION.md/submission.csv/make_submission.py
state is already locked in at the verified yixi10 blend and requires no
further action) rather than further speculative merge rounds.

(Further results appended below as rounds continue.)

# Devpost submission, Track 2: Autonomous ML Research Agent for Within-User Ranking on KuaiRand-Pure

## Inspiration

Track 2 hands you an unusually clean research problem: rank each user's own feed exposures by
`long_view` likelihood, scored by mean(GAUC, nDCG@5). The organizer starter kit ships a working FM
baseline (hidden test primary 0.5946) and a "from here" section naming directions nobody had tried
yet. That combination, a fixed metric, a fixed harness, and a documented set of open hypotheses, is
exactly the setup an autonomous coding agent should be good at: propose a change, implement it,
evaluate it against a metric that doesn't lie, and decide honestly whether to keep it. We wanted to
find out how far that loop could go if we mostly got out of its way, and how it would behave when
it hit a wall.

## What it does

The final submitted model is a within-user-percentile blend of three independently-trained
components:

- an **FM ranking model** trained with pairwise BPR loss (iter38, a 5-seed sigmoid-mean ensemble),
- a **LightGBM ranker** (`num_leaves=2`, `linear_tree=True`, `learning_rate=0.10`, lambdarank
  objective) trained on a feature set that adds a 5-day historical watch-depth decay feature and a
  native `upload_type` categorical, weighted 52%,
- an **XGBoost ranker** tuned independently on native, un-bucketed causal features, weighted 38%,

with the FM component taking the remaining 10%. It scores **validation primary 0.69943440,
hidden test primary 0.68432260**, against the organizer FM baseline's hidden test primary of
0.5946 (GAUC 0.6610, nDCG@5 0.5282). That's **+0.08972 absolute, +15.09% relative**, fully
reproducible end to end with `python3 make_submission.py`.

## How we built it

We ran an autonomous, orchestrator-driven iteration loop using Claude Code as the coding agent,
split across four tracks: three that eventually merged into the submitted model, and a fourth
that independently stress-tested it afterward.

**Track 1: the main FM/GBM line (`experiments/LEDGER.md`, ~80 logged iterations).** The loop
followed a fixed protocol on every iteration: hypothesize from the starter kit's own named-untried
directions, implement, gradient-check any hand-derived loss (verified against finite-difference
numerical gradients on a toy example before trusting a real run, which caught a real bug in a
multi-task gradient derivation), reproduce a known-good reference result bit-exact before trusting
a new sweep, select on validation only, and confirm any gain above 0.001 across 5 seeds before
promoting it. We pre-declared a convergence rule up front (3 consecutive rounds with no validation
improvement ≥ 0.002, ε=0.002/N=3) rather than deciding informally when to stop, and it triggered
after Round 12: switching the loss from pointwise to pairwise BPR was the single biggest jump,
followed by recency-decay features, activity-weighted BPR sampling, and constant retuning.

We kept going past that point on explicit instruction not to treat "converged" as "done." Two GBM
attempts (LightGBM, CatBoost) had been closed earlier as REJECT because they underperformed FM, but
the actual diagnosis pointed at the feature encoding, not the model family: both had been forced
through FM's own bucketed representation, which throws away exactly the continuous magnitude signal
a GBM needs. Giving the GBM its own un-bucketed, "GBM-native" encoding of the same causal features
(iter44) closed the gap and then reversed it by a wide margin, and a follow-up sweep found the
metric responded strongly and monotonically to *shrinking* tree capacity all the way to LightGBM's
floor, `num_leaves=2`. A result that large and that monotonic is exactly the shape of a bug, not a
discovery, so before promoting it we specifically hunted for a stable-sort tie-artifact in the
evaluation harness and a silently-added confound feature, and only trusted it once both came back
clean and it held across 5 seeds and a date-shifted split. Blending the GBM with the FM ensemble
gave a further gain from genuine model-family diversity. Three more rounds each found one further,
genuinely new lever on top of that GBM (`linear_tree=True` turning each one-split tree into a
piecewise-linear function instead of piecewise-constant; a `learning_rate` resweep that had never
been re-validated under `linear_tree`; a decayed per-tab engagement *rate* feature replacing a
count that had been sitting unused in the feature set since early in the project), each one
reblended with the FM ensemble and confirmed at 5 seeds before promotion.

**Track 2: Yixi's independent branch (`iterYIXI1`-`iterYIXI11`, 11 iterations).** A teammate worked
the same problem from a separate branch, reusing this project's own FM training code unchanged but
adding XGBoost as a third model family and pushing her own line of feature and objective
refinements: transferring the 5-day user-decay features to XGBoost, retuning its learning rate,
retuning LightGBM's objective toward ranking, and adding a causal watch-depth history feature and a
native `upload_type` categorical. Her chain converged on a 10% FM / 52% LightGBM / 38% XGBoost
blend at valid 0.69943440 / test 0.68432260, a large jump over anything the main track had found on
its own.

**Track 3: merge and verify (`experiments/MERGE_LEDGER.md`, 9 iterations).** Rather than trust
Yixi's number and stop, the orchestrator independently retrained all three of her components from
raw CSVs, seed 0, with no reuse of her cached artifacts. Every component and the final blend
matched her claim to 8 decimal places, so the orchestrator promoted it as the new best-known
result, which became the submitted model. The merge loop then kept searching for a further gain
by combining both tracks' findings, which is where most of the interesting negative results in
this project live, and where the loop's most convincing self-check happened (see below).

**Track 4: Xuxia's independent verification lane (`XUXIA_SUMMARY.md`, `experiments/iterXUXIA1`-`3`,
3 iterations).** A second teammate ran a separate stress-test of the blend from a different angle
than either main track: per-segment blend weighting (by `tab` and by activity tertile, both
regressed against the single global alpha in all 5 seeds), rank-based and calibrated fusion
(Borda/RRF and isotonic regression, all regressed), and a GBM-native multi-task stacking variant
(leakage-free OOF auxiliary features from `is_like`/`is_follow`/`is_comment`/`is_forward`, gain
below the promotion threshold with 3 of 4 auxiliary columns unused by any split). All three closed
clean REJECT, none touching the submitted model, and this lane's harness-fidelity check caught the
same stale-baseline trap the orchestrator has hit before: the handoff instructions pointed at an
old iter44 reference, and the lane correctly reproduced the current `main` instead of silently
scoring against a stale number.

## Challenges we ran into, and what we rejected

A documented REJECT is a result in its own right here. The merge track produced several worth
naming because they show the loop actually checking its own work rather than chasing a number:

- **A 4th component that looked promising and wasn't reliable.** Adding the main track's own
  minimal-feature GBM as a 4th blend member alongside FM/LightGBM/XGBoost raised validation by
  +0.00038 at the best weight point (confirmed tight across 5 seeds and across neighboring grid
  points, so it wasn't a spike). But test score moved the wrong way, down 0.00065 relative to the
  3-model blend. That's a validation/test inconsistency, not noise, and it's exactly the kind of
  number a less disciplined loop promotes because the validation metric went up.
- **We didn't trust one check. We ran two independent ones, and both said the same thing.** The
  obvious suspect for the 4-component blend's valid-up/test-down pattern is that a fine-grained
  weight grid search overfit its choice to a single 124,909-row validation split. First check:
  reselect the blend weight using 5-fold user-level cross-validation instead of a single split. It
  converged to the exact same weight point, which rules out grid-search overfitting as the cause.
  Second check, from a structurally different angle: fit a k-fold, user-level out-of-fold stacked
  meta-learner (logistic regression and a shallow GBM) over the four components' percentile
  scores, a model that can't overfit a single split the way a grid search can because its score is
  computed only on held-out folds. Its best honest out-of-fold score still came in below both the
  3-model blend and the 4-model linear optimum, finding no exploitable nonlinear structure among
  four already-agreeing, already percentile-normalized scores. A cross-validated linear reselection
  and a nonlinear out-of-fold stacker are different enough in mechanism that they're unlikely to
  share a blind spot, and they agreed: the +0.00038 was a genuine distributional difference between
  the validation dates (20220422-20220428) and the later test dates (20220429-20220508), not a
  selection artifact. We rejected the higher-validation model anyway.
- **Calibration made things much worse.** We tested whether isotonic regression could fix that
  inconsistency by recalibrating each component's scores before blending. It collapsed the tree
  models' score distributions from hundreds of thousands of unique values down to 35-70 discrete
  levels. That's catastrophic for a ranking metric that depends on fine-grained within-user
  ordering, and validation dropped by 0.061, two orders of magnitude past anything close to
  useful. Rejected outright, no seed confirmation needed, the regression wasn't borderline.
  Xuxia's independent verification lane hit the identical failure mode from a different starting
  point (isotonic calibration on the pre-merge blend, not the 4-component one): it collapsed the
  GBM's roughly 123,000 unique raw scores down to 37 discrete levels and dropped validation by
  0.036. Two separately-implemented experiments against two different blend configurations finding
  the same catastrophic isotonic collapse is stronger evidence than either alone that this is a
  structural mismatch between tree-based ranking scores and monotonic-step calibration, not a
  one-off tuning mistake.
- **Three decay-rate features, tested twice, stayed null.** Author-popularity, duration-bucket, and
  hour-of-day decay-rate features (the same construction pattern that had won earlier for a
  different feature) were already a clean null against Yixi's rich 30+-column feature set. We
  retested them in a minimal 6-column feature set with far less split competition, on the theory
  that a weak feature might win a split when it has fewer rivals. It didn't: zero LightGBM splits
  used, byte-identical predictions to the unmodified baseline. Four for four nulls across two
  independent harnesses closes that feature family for this problem generally, across model types.

Because of all this, we never promoted the 4-component blend, even though it offered a slightly
higher validation number. The 3-model blend was more robust, its valid/test relationship was
consistent, and nine rounds of trying to beat it honestly came back empty. Keeping the
conservative, already-verified model instead of chasing a marginal and unreliable gain was the
right call, and we think that decision is worth as much credit as any of the promotions above.

## What's next / limitations

The open question we didn't resolve is exactly the one the merge track surfaced: why validation and
test disagree on which blend is best once a 4th component enters the mix. We ruled out grid-search
overfitting via cross-validation, so the leading explanation is a genuine distributional shift
between the validation window and the later test window rather than a fixable modeling choice.
Resolving it properly would need either a validation scheme that spans a wider date range (so the
selection split looks more like the eventual test distribution) or more post-validation-window data
to check whether the gap holds at a third, independent time slice. We also never reached
user-history sequence modeling (DIN/SIM-style attention over a user's raw interaction sequence),
which was in the original plan. The recency-decay features are a lightweight, hand-built proxy for
that kind of signal, and a learned sequence model over each user's raw history remains the most
likely next lever if we picked this back up.

## Development tools used

**Claude Code** (Anthropic), used as the primary coding agent driving the entire iterative research
loop end to end: proposing hypotheses, writing the training/feature code, launching experiments,
reading back raw results, and deciding what to promote or reject. Across the four tracks combined
(main FM/GBM line, Yixi's independent branch, the merge-and-verify phase, and Xuxia's independent
verification lane) this covers 103 logged iterations and experiments. This is the actual subject
of what Track 2 evaluates, so we
want to be direct about it rather than understating it: the model and feature engineering here are
GBM-and-FM statistics, nothing exotic, and the interesting part of the submission is the process
that found and verified them, driven almost entirely by the agent rather than by us hand-tuning
hyperparameters.

## APIs used

None. No external APIs, no external model providers, no network calls at inference or training
time.

## Libraries/frameworks used

numpy, pandas, scikit-learn, LightGBM, XGBoost. The FM/BPR model is a from-scratch numpy
implementation, not a library. No PyTorch or other deep learning framework was needed; every
component here is either gradient-boosted trees or a from-scratch factorization machine with linear
blending on top.

## Datasets used

KuaiRand-Pure, organizer-provided, sourced from Kuaishou's short-video feed logs. No external
training data was used beyond what the starter kit shipped.

## Resource usage

**GPU-hours: 0.** The entire stack, FM training, LightGBM, XGBoost, and every experiment in the
loop, ran on CPU only, and that was a deliberate choice: gradient-boosted trees and a small
factorization machine are competitive on this problem without a GPU, and a full 5-seed FM ensemble
plus both GBM components trains in on the order of two minutes total on one core.

**Token consumption:** we don't have exact per-call input/output token counts from the orchestrator,
so we're reporting the honest proxy we do have: 103 logged iterations/experiments across the four
tracks, each one a full Claude Code session (hypothesis, implementation, evaluation, and a
written verdict with reasoning). That's the real cost basis of this submission, not a specific token
figure we'd have to make up to report.

# Devpost submission draft — Track 2: Within-User Ranking on KuaiRand-Pure

## Inspiration

Track 2 poses a narrow, well-specified problem: rank each user's own feed exposures by
`long_view` likelihood, scored by mean(GAUC, nDCG@5). The organizer-provided starter kit ships a
working FM baseline (test primary 0.5946) and a "from here" section naming untried directions.
That's an unusually good setup for an autonomous iteration loop — a fixed metric, a fixed
harness, and a documented set of open hypotheses to test systematically rather than guess at.

## What it does

A score-level blend of two independently-trained model families — an FM ranking model (pairwise
BPR loss, three layered additions: recency-decay/momentum features, activity-weighted BPR
sampling, retuned smoothing constants) and a LightGBM ranker (`linear_tree=True`, a linear model
per leaf rather than a flat constant, `learning_rate=0.10`) trained on a from-scratch, un-bucketed
encoding of the same causal features plus a decayed per-tab engagement *rate* feature — that
improves test primary from the FM baseline's 0.5946 to **0.65955** (+0.0650 absolute, +10.92%
relative), fully reproducible with `python3 make_submission.py`.

## How we built it

We ran an autonomous, orchestrator-driven iteration loop over 19 rounds / 63 iterations:

1. **Hypothesize** from the starter kit's own "organizer-suggested unexplored directions"
   (loss function, feature causality, sampling strategy, model architecture, multi-task learning,
   model capacity) rather than guessing blind.
2. **Implement and gradient-check.** Any newly-derived (not-copied) gradient formula was verified
   against finite-difference numerical gradients on a tiny synthetic example before being trusted
   in a real training run — this caught a real bug in a multi-task gradient derivation before it
   wasted compute.
3. **Harness-fidelity check.** Every new experiment first reproduced a known-good reference
   result bit-exact before its own sweep was trusted.
4. **Select on validation only**, confirm any >0.001 valid gain across 5 seeds, and independently
   re-verify every reported number by reading raw results and hand-computing means — never trust
   a script's self-reported summary.
5. **Converge formally.** We pre-declared a convergence rule (3 consecutive rounds with no
   validation improvement ≥0.002) rather than iterating until time ran out, and stopped exactly
   when that rule triggered — Rounds 10-12 all closed real, disjoint hypotheses (a feature fusion,
   a robustness check, a re-architected multi-task attempt with a gradient-verified
   reimplementation, and a model-capacity sweep) without finding a new best.

Everything runs on CPU, no GPU, no external APIs. The FM/BPR line uses numpy only, matching the
starter kit's own baseline; the final model's GBM component (added in a later round, see below)
additionally uses `pandas` and `lightgbm` — both pip-installable, and explicitly named as an
acceptable direction in the starter kit's own "from here" notes.

**Post-convergence, we kept going rather than stopping at "good enough."** After the loop
formally converged (see below), an explicit instruction not to give up on open-source/non-neural
model directions sent us back to two previously-REJECTed GBM attempts (LightGBM, CatBoost) that
had underperformed FM. The diagnosis for both had pointed at the *feature encoding* (both were
forced through FM's own bucketed representation, discarding exactly the continuous
ordering/magnitude signal a GBM is built to exploit), not at the model family — so rather than
treating "GBM underperforms FM" as closed, we gave the GBM its own un-bucketed encoding of the
same causal features. This closed the gap and then reversed it by a wide, seed-stable,
date-shift-robust margin (LightGBM alone: test 0.64794 vs. FM's 0.64187), and blending the two
model families' scores (10% FM / 90% GBM) gave a further gain from genuine model diversity,
becoming the final result through Round 15 (test 0.65197). Because this gain was unusually large
relative to everything else found across 44 iterations, it went through a longer verification
chain than usual before being trusted (ruled out as a stable-sort tie artifact, ruled out as
driven by a silently-added feature, confirmed stable across 5 seeds, confirmed to hold under a
date-shifted train/valid/test split) — see
[`experiments/iter44_gbm_native_features/RESULT.md`](experiments/iter44_gbm_native_features/RESULT.md).

**One further structural gain (Round 16) became the new final result.** At the GBM's
best-scoring capacity (`num_leaves=2`), every tree makes exactly one split and predicts a flat
constant on each side. Turning on LightGBM's `linear_tree=True` option fits a linear regression
per leaf instead, so the same one-split tree becomes piecewise-*linear* rather than
piecewise-constant — a structural change, not a hyperparameter retune, and untried across the
six other Round-15 methods (a second GBM library, a hyperparameter depth sweep, a stacking
meta-learner, a time-of-day feature, monotonic constraints, GOSS boosting) that had all landed as
clean rejects. It gained +0.0079 valid over the constant-leaf GBM on the first run, confirmed
tight across 5 seeds, and re-blending it with the unchanged FM ensemble (at a re-swept 8% FM /
92% GBM) pushed the result to **test 0.65643** — see
[`experiments/iter51_linear_tree/RESULT.md`](experiments/iter51_linear_tree/RESULT.md).

**One further lever (Round 17) pushed the result again.** After the `linear_tree=True` promotion,
we kept testing on explicit instruction. Three retests of whether the new tree type reopened
previously-closed directions (capacity, its own `linear_lambda` regularization knob, a
previously-rejected hour-of-day feature) all confirmed the existing configuration as a robust
local optimum along those axes — clean rejects, not gains. A fourth, genuinely new hypothesis —
`learning_rate`, tuned years earlier against the *old* constant-leaf tree and never re-checked
against the new piecewise-linear one — found a real further gain: `learning_rate=0.10` beat the
baseline by +0.00085 valid (5-seed confirmed, 5/5 seeds improving), and re-blending pushed the
result to **test 0.65832** — see
[`experiments/iter55_learning_rate_sweep/RESULT.md`](experiments/iter55_learning_rate_sweep/RESULT.md).

**Round 18 pivoted to the FM side and came up empty — instructively.** With the GBM
hyperparameter space confirmed exhausted, we resweept the two FM hyperparameters most analogous
to the levers that had just paid off on the GBM side (embedding dimension, learning rate) plus
BPR's negative-sampling weight. All three were clean rejects — but the FM `learning_rate` resweep
is worth noting on its own: it found a real, 5-seed-confirmed standalone FM gain (+0.00108 valid)
that simply didn't survive re-blending with the GBM (test even moved the wrong way). A useful
reminder that "real at the standalone level" and "real at the level that's actually submitted"
are two different bars, and only the second one matters.

**Round 19 found one more genuinely new feature and produced the current final result.** Every
feature set since early in the project had carried a decayed per-tab positive *count*, but never
the matching decayed-total denominator needed to turn it into a Laplace-smoothed *rate* — the same
count→rate upgrade that had already paid off once earlier in the project for a different feature.
Building that denominator and swapping the count for the rate gave a real, causally-verified,
5-seed-confirmed GBM gain (+0.00107 valid mean, 5/5 seeds) that — unlike the Round 18 FM finding —
propagated cleanly through to the blend: re-blending with the unchanged FM ensemble pushed the
final result to **test 0.65955** — see
[`experiments/iter63_decay_tab_rate/RESULT.md`](experiments/iter63_decay_tab_rate/RESULT.md).

## Challenges we ran into

- **Platform interruptions mid-experiment** (parallel background runs killed by session limits)
  cost an entire round's compute at one point. We hardened the process afterward: smaller,
  single-phase experiment scopes, results written to disk incrementally rather than buffered, and
  salvaging partial results from raw logs instead of blindly rerunning.
- **A subtle multi-task learning bug.** A second attempt at multi-task auxiliary heads (after a
  first design was rejected) needed a hand-derived gradient for a new per-task loss term. Finite-
  difference verification against a toy example caught a missing normalization factor before any
  real training run — the kind of bug that would otherwise have produced a plausible-looking but
  wrong result.
- **Knowing when to stop — and when "stopped" doesn't mean "done."** With no imposed limit, an
  iteration loop can run indefinitely, chasing noise. Pre-committing to a convergence rule (rather
  than deciding informally at the end) kept rounds honest about whether they were finding real
  signal — but convergence under a fixed feature representation isn't the same as convergence
  full stop. Two GBM attempts had been closed as REJECT earlier, and the temptation was to read
  that as "GBMs don't fit this problem" and move on. Revisiting the actual diagnosis (a
  feature-encoding mismatch, not a model-family mismatch) rather than the verdict label found a
  further +9.65%-relative gain that a less skeptical reading would have left on the table.
- **A result too good to trust at face value.** A monotonic trend running all the way to a
  hyperparameter's literal floor, with a bigger single-iteration gain than anything else found
  across 44 iterations, is exactly the shape of a bug, not a discovery. Before promoting it we
  specifically hunted for the two most likely culprits (an evaluation-harness sorting artifact, a
  silently-added confound feature) and only trusted the result once both came back clean and it
  held up across seeds and a shifted date split.

## Accomplishments that we're proud of

- +10.92% relative improvement over the official baseline (up from +7.45% at the first
  convergence point), the extra gain found entirely in a self-directed post-convergence phase
  rather than from any new instruction — including seven further methods tried after the GBM
  blend result above, six of which were clean rejects before the seventh (`linear_tree=True`)
  found the next real gain, three more rejects after that before an eleventh
  (`learning_rate=0.10` under `linear_tree=True`) found the next gain, three FM-side rejects after
  that (one of which found a real standalone-only gain that didn't survive blending — a useful
  negative result in its own right), and finally a new causal feature (a decayed per-tab
  engagement rate) that produced the currently submitted result.
- A disciplined negative-results record: two multi-task learning designs and a model-capacity
  sweep were tried, diagnosed, and closed with documented reasoning rather than silently dropped
  or retried indefinitely — and, distinctly, a documented *reopening* of a previously-closed
  direction (GBMs) once its closure reason was checked against a changed assumption.
- A verification discipline (harness-fidelity checks, finite-difference gradient checks,
  independent hand-computation of every reported number, and — for the final, largest-ever
  single gain — a four-part suspicion-driven verification chain) that caught real bugs and ruled
  out artifacts before they could produce a misleading result.

## What we learned

The biggest gain came from aligning the training objective with the evaluation objective
(pairwise BPR loss vs. a ranking metric) rather than from bigger models or more features — the
`user_id × video_id` interaction already captures most of the learnable signal, and pushing
either raw feature count or embedding capacity further past that point doesn't move the needle.
The second-biggest lesson came later: a "REJECT" verdict is only as good as the assumption it was
tested under, and the same feature set that starves an FM (needing discretized/embeddable inputs)
can starve a GBM trained the same way, for the opposite reason (a GBM needs the continuous signal
that discretization throws away). Matching each model family to its own native representation,
rather than reusing one family's pipeline for both, unlocked a second, independent source of
signal — and the two final models' blend outperforming either alone confirms they're capturing
genuinely different things, not just noisier copies of the same signal.

## What's next

The user-history sequence-modeling direction (DIN/SIM-style attention over a user's raw
interaction sequence) remains the most likely next lever — the current recency-decay features are
a lightweight proxy for recency, not a learned sequence model. A third GBM library (CatBoost) was
since tried on the correct native encoding and closed (underperformed LightGBM, added nothing in
a 3-way stack); the GBM's wider valid/test gap at its smallest, best-scoring tree size remains a
documented, seed-stable, date-shift-confirmed property worth further study rather than a red flag
to walk back the result.

## Built with

Python, numpy, LightGBM, pandas, Claude Code (autonomous multi-round orchestration and direct
implementation).

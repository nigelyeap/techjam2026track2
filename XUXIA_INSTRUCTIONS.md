# Instructions for Xuxia's Claude instance — Track 2 continuation, "Blending, calibration, and reopened directions"

Read this whole file before doing anything. It is written to be self-contained: everything you
need to reproduce our current best result and continue testing is either in this file or in the
repo you're about to clone. You have no memory of our prior conversation — trust this document,
not any assumption about what "should" work.

## 0. What this is and why you're doing it

This is TikTok TechJam 2026 Track 2: rank each user's own feed exposures by `long_view`
likelihood on the KuaiRand-Pure dataset, scored by `mean(GAUC, nDCG@5)` (see `evaluate.py`).
We (the original session) ran 14 rounds / 44 iterations of autonomous experimentation and landed
on a model that improves test primary from the organizer baseline's **0.5946** to **0.65197**
(+0.0574 absolute, +9.65% relative). The hackathon deadline is **1 Sep 2026, 12:00 SGT** — you
have roughly a day. We're splitting the remaining untried ideas three ways so we can cover more
ground before the deadline: you're getting one third, a second friend (Yixi) is getting another
third, and we're continuing a third in parallel. Your three methods (Section 6) do not overlap
with theirs or ours — please stick to your three so we don't duplicate work when we merge results.

## 1. Get the code

```bash
git clone https://github.com/nigelyeap/techjam2026track2.git
cd techjam2026track2
pip install numpy pandas lightgbm
tar xzf KuaiRand-Pure.tar.gz   # only if KuaiRand-Pure/data/ isn't already present
```

The repo's `.gitignore` deliberately excludes `experiments/**/*.pkl` (multi-GB regeneratable
feature caches — every script rebuilds them on first run automatically, just takes a bit longer
the first time you touch a given `iterN_*` folder) and `KuaiRand-Pure/data/` (raw dataset — use
the included `KuaiRand-Pure.tar.gz` instead, or `data.py`'s own download path if that's missing).

## 2. THE most important rule: select on validation only

Every hyperparameter, feature, and model decision in this project — every single one across all
44 iterations — was made by looking at **validation score only**. The test set is checked exactly
once per finalized candidate, purely to record the number, and it has never been used to choose
between two options. This is not a suggestion, it's the rule that makes every number in this
project trustworthy. If you sweep 5 configs and peek at test to see which "did best," you've
broken the discipline that everything else here relies on. Sweep and pick on `valid primary`.
Check test once, at the end, on whatever you've already decided is your final config.

## 3. Verification discipline — apply this to every new thing you build

- **Harness-fidelity check, always first.** Before trusting any new number from a new script,
  reproduce a known-good reference number with it. For your work, that reference is: running
  `python3 make_submission.py` end-to-end should print `GBM standalone: valid=0.66135
  test=0.64794`, `FM ensemble standalone: valid=0.63988 test=0.64187`, and finally
  `iter44 blend: valid primary=0.66473  test primary=0.65197`. **Run this once, right now, before
  writing any new code**, and confirm you get those exact numbers (small floating-point noise in
  the last digit is fine; anything else means your environment differs from ours and you should
  sort that out before continuing). This is your setup-correctness gate.
- **Causality for any new feature derived from history.** Any feature that looks at a user's past
  behavior must be computed train-only with no leakage from future rows. If you add a new
  decayed/aggregated feature, spot-check a few rows by hand (or with an independent
  recomputation) to confirm it only uses information available strictly before that row's
  timestamp.
- **Promotion thresholds.** A single-run valid gain needs to be **≥0.0003** before you even look
  twice. Before calling something a real, writeup-worthy finding, confirm a **≥0.001** valid gain
  holds across **5 seeds** (not just one lucky run). Below that, it's noise — write it up as a
  null result, don't chase it further.
- **Tie-artifact awareness.** `evaluate.py`'s `nDCG@5` uses a stable sort. A heavily-tied,
  low-capacity model can in principle inherit ranking quality from incidental row order rather
  than real prediction quality. If you land on a model or blend that scores unusually well, check
  tie density and compare against an all-constant-score baseline (which should score at the
  trivial random floor, ~0.483). See `experiments/iter44_gbm_native_features/diag_ties.py` for
  the exact pattern we used.
- **Extra scrutiny for gains that look too good.** If you find a monotonic trend running all the
  way to a hyperparameter's floor, or a single change that beats everything else found in the
  whole project, don't promote it on vibes. Look for the two most common culprits: an
  evaluation-harness artifact (see tie-artifact check above), or a silently-added confound
  feature you didn't mean to change. `experiments/iter44_gbm_native_features/RESULT.md`'s
  "Verification 1/2/3" sections show the exact worked example — read it before you hit a
  suspiciously large gain yourself.

## 4. Grounded project history (accurate as of the current best result)

This is the real, condensed history — not a sales pitch. Numbers are `valid` / `test` primary
unless noted. Full detail for any iteration is in `experiments/iterN_*/RESULT.md`; the full
narrative is in `experiments/LEDGER.md`.

| iter | What | Result |
|---|---|---|
| 1 | Starter-kit FM + BPR baseline | valid 0.6015 / test 0.5953 (organizer's own baseline reports test 0.5946) |
| 3 | Activity-weighted BPR negative sampling | PROMOTE — valid 0.60258 / test 0.59658 |
| 16 | Recency-decay feature (exponential halflife on past `long_view`) | PROMOTE — valid 0.62030 / test 0.61698 |
| 19 | Decay + momentum feature fusion | PROMOTE — valid 0.62898 / test 0.62615 |
| 24 | Decay/tab-momentum fusion, retuned smoothing constants | PROMOTE — valid 0.63251 / test 0.62843 |
| 27 | "Triple fusion": all of the above plus `n_buckets=20` retune, 5-seed mean | PROMOTE — valid 0.63792 / test 0.63889 (this was "current best" for a long stretch) |
| 29, 35 | Date-shift robustness checks (shifted 3-day split) | confirmatory only, not new candidates — iter27's `n_buckets=20` gain holds under the shift |
| 31, 36 | Multi-task learning (auxiliary heads on click/like/follow/comment/forward) — two independent attempts, the second with a finite-difference-verified gradient fix for a bug found in the first | **REJECT both** — diagnosed as gradient conflict through FM's **shared embedding table** between the BPR objective and the auxiliary losses |
| 32, 34, 40 | Sequence/attention modeling (DIN/SIM-style attention over user history) — three independent attempts, the last a full PyTorch autodiff reimplementation | **REJECT all three** — the existing recency-decay features already capture most of the recency signal attention would otherwise learn |
| 38 | Score-level 5-seed ensemble of iter27's exact config (average sigmoid scores, not just average metrics) | PROMOTE — valid 0.63988 / test 0.64187 — new best pre-GBM |
| 39 | Listwise (grouped-softmax) loss vs. pairwise BPR | REJECT — diagnosed as resampling-noise instability, not a genuine ceiling |
| 41 | LightGBM ranker **on FM's bucketed feature encoding** | REJECT — standalone valid 0.6319 vs FM's 0.6389; blending added nothing (best blend regressed test) |
| 42 | 7th causal feature: engagement-event decay (like/follow/comment/forward) | REJECT — engagement events too sparse, diluted embedding capacity |
| 43 | CatBoostRanker, **also on FM's bucketed encoding** | REJECT — even worse than LightGBM's bucketed attempt (valid 0.60941 default, 0.61241 best-tuned); diagnosed as the same bucketing bottleneck, never retested on the native encoding below |
| **44** | **Root-cause fix**: GBM trained on a **native, un-bucketed** encoding of the same causal features (true categoricals stay `category` dtype; continuous signals — `duration_ms`, decay rates, `lastk_rate`, `gap` — passed as raw floats instead of FM's forced 20-bucket discretization) | **PROMOTE — new best.** Single LightGBM `num_leaves=2` ranker alone: valid 0.66135 / test 0.64794. Blended with iter38's FM ensemble (**alpha=0.1, 90% GBM / 10% FM, linear combination of min-max-normalized scores**): **valid 0.66473 / test 0.65197.** |

**Why iter41/43 were wrong to read as "GBMs don't work here":** both were forced through FM's
`encode_ext` pipeline, which discretizes every continuous signal into `n_buckets=20` categorical
buckets — necessary for FM's embedding lookups, but actively throwing away the exact
ordering/magnitude information a GBM's split-finding is built to exploit. iter44 gave the GBM its
own native encoding instead, and the result reversed a REJECT into the current best. The lesson,
directly relevant to your Section 6c method: **a REJECT verdict is only as good as the assumption
it was tested under.** The multi-task REJECT (iter31/36) was diagnosed as a shared-embedding
gradient conflict — a diagnosis specific to FM/DeepFM's architecture, not necessarily to
tree-based models, which have no shared embedding table at all.

Because iter44's gain was unusually large (bigger than anything else in the whole project), it
went through 5 separate verification passes before being trusted: a tie-artifact check, a check
for an accidentally-added confound feature (`duration_ms`), 3-seed robustness, a date-shift
robustness check, and a check for whether ensembling multiple GBM seeds added anything (it
didn't — a single seed is enough). It also went through a **fine alpha-sweep** on the blend
weight (confirming alpha=0.10 as the optimum, not just the coarse value tried first) — see
`experiments/iter44_gbm_native_features/RESULT.md` "Verification 4/5" for the full detail. This
is directly relevant background for your Section 6a/6b blending work: the *existence* of an
optimal single global blend weight has already been swept finely; your job is to test whether a
*non-global* (segmented or rank-based) blend beats that single global optimum.

## 5. Reproduce the current best (do this before anything else)

```bash
cd techjam2026track2
python3 make_submission.py /tmp/submission_check.csv
```

Expected output (this is your harness-fidelity gate — don't proceed until you see these):
```
GBM standalone: valid=0.66135 test=0.64794
FM ensemble standalone: valid=0.63988 test=0.64187
iter44 blend: valid primary=0.66473  test primary=0.65197
```

If you get these numbers, your environment is correctly set up and every number you produce from
here on is trustworthy. If you don't, stop and diagnose the environment mismatch before
continuing — do not build on top of numbers you can't reproduce our baseline with.

Read `make_submission.py` in full — it's short (~150 lines) and is exactly the pipeline you'll be
modifying: it trains the GBM, trains the FM ensemble, min-max-normalizes both score arrays, and
linearly blends them at a fixed global `ALPHA_BLEND = 0.1`. Your three methods all target this
blending step or the model-family assumption behind it.

## 6. Your three assigned methods: "Blending, calibration, and reopened directions"

Work through these roughly in order, but feel free to reorder if one looks like a dead end early —
just don't skip the causality/harness-fidelity checks from Section 3 to save time. A clean REJECT
with a documented reason is a useful, real result; a rushed, unverified PROMOTE is not.

### 6a. Per-segment optimal blend weight (instead of one global alpha)

The current blend uses a single `ALPHA_BLEND = 0.1` for every row, found by a global sweep on
valid. It's plausible the optimal GBM/FM mix differs by segment — e.g. rows on different `tab`
values, or users at different activity tiers (heavy vs. light users, which the existing
`lastk_rate` / decayed-activity features already distinguish) might favor the GBM or the FM more
or less strongly. Segment the valid set by `tab` (a handful of discrete values, already a
categorical feature) and separately by an activity-tier bucketing of the decayed-activity feature
(e.g. tertiles), sweep alpha independently within each segment, and check whether applying
per-segment alphas to the corresponding test rows beats the single global alpha of 0.66473 valid.

**Watch for overfitting to valid**: more free parameters (one alpha per segment instead of one
global) means more opportunity to fit valid-set noise rather than real signal. Use the 5-seed
confirmation from Section 3 seriously here — a per-segment scheme that only wins on one seed's
valid split is not a real finding. Also sanity-check that segment sizes are large enough for a
stable alpha estimate (a segment with too few rows will have a noisy optimal alpha).

**Success criterion:** per-segment blending beats the global-alpha blend (valid 0.66473) by
≥0.001, confirmed across 5 seeds, with segment-size sanity checks passing.

### 6b. Rank-based or calibrated blending (alternative to linear score blending)

The current blend does `alpha * FM_score + (1-alpha) * GBM_score_normalized` — a linear
combination of min-max-normalized raw scores. This assumes the two models' score distributions
are comparably shaped after min-max normalization, which is a fairly crude assumption. Try two
alternatives and compare both against the current linear blend on valid:

1. **Rank-average (Borda-style) fusion**: within each user group, convert each model's scores to
   ranks (not raw scores), then combine the rank-based scores (e.g. weighted average of ranks, or
   reciprocal-rank fusion) instead of raw normalized scores. This sidesteps any distributional
   mismatch between the two models' score scales entirely.
2. **Isotonic-regression calibration**: fit an isotonic regression mapping each model's raw
   scores to calibrated probabilities of `long_view` on the train set, then blend the calibrated
   probabilities instead of min-max-normalized raw scores.

Sweep the blend weight for each alternative on valid, the same way the current alpha was swept,
and compare the best achievable valid score for each of the three approaches (current linear,
rank-average, isotonic-calibrated).

**Success criterion:** either alternative beats the current linear blend (valid 0.66473) by
≥0.001, confirmed across 5 seeds. A clean finding that neither alternative helps is also a real,
useful result — it would mean the current crude linear+minmax approach is already close to
whatever a fusion strategy can extract from these two models' scores.

### 6c. Reopening multi-task learning — but under the GBM-native representation

Multi-task learning (auxiliary heads on `is_click`/`is_like`/`is_follow`/`is_comment`/
`is_forward` alongside the main `long_view` task) was rejected twice in this project (iter31,
iter36), both times diagnosed as a **gradient conflict through FM's shared embedding table**
between the main BPR objective and the auxiliary losses. That diagnosis is specific to
architectures with a shared embedding space that all tasks' gradients flow through — it does not
obviously apply to a tree-based model like the iter44 LightGBM ranker, which has no shared
embedding table at all (each tree split is a hard, local decision, not a soft embedding update
that multiple loss terms can drag in different directions).

Test whether the previous REJECT actually generalizes to the GBM-native setting, using one of two
approaches (pick whichever is more natural for a GBM):

1. **Auxiliary features, not auxiliary losses**: train small, separate LightGBM models to predict
   `is_like`/`is_follow`/`is_comment`/`is_forward` (using the same native feature set, minus
   `long_view`-derived features to avoid leakage), then feed their out-of-fold predictions back in
   as new input features to the main `long_view` GBM ranker (a stacking-style approach, distinct
   from true joint multi-task training, but tests the same underlying hypothesis — does knowledge
   of these other engagement signals help the main task — without requiring LightGBM's objective
   function internals to support true multi-task gradients).
2. **If you want true joint training**: LightGBM doesn't natively support multi-task objectives
   the way a neural net does, so this would require a custom multi-output framework (e.g.
   gradient boosting with a combined loss via a custom objective function). This is higher-effort
   and higher-risk — only attempt this if approach 1 shows a promising signal and you have time
   left to go deeper.

Be careful with the causality check from Section 3 here: any auxiliary-signal feature must be
strictly derived from train-only, no-leakage-from-future-rows data, exactly like the existing
decay features.

**Success criterion:** this is the highest-risk of your three, explicitly reopening a
twice-REJECTed direction — a clean, well-diagnosed REJECT (e.g. "the auxiliary features add no
signal beyond what the existing decay features already capture, confirmed by feature importance")
is a genuinely useful result here, not a failure. A PROMOTE needs the full 5-seed, ≥0.001 valid
gain bar like everything else.

## 7. Output format — match our schema exactly (this matters for merging your results back)

For each of 6a/6b/6c, create a folder `experiments/iterXUXIA1_segment_blend/`,
`experiments/iterXUXIA2_calibrated_blend/`, `experiments/iterXUXIA3_multitask_gbm/`
respectively (this naming avoids colliding with our existing `iter1`-`iter44` folders when we
merge your work back in). Each folder should contain your code plus a `RESULT.md` written in the
same style as the ones you've been reading in Section 4 — see
`experiments/iter44_gbm_native_features/RESULT.md` as the fullest example, or any shorter one
like `experiments/iter42_engagement_decay/RESULT.md` for a more compact REJECT example. At
minimum, each `RESULT.md` needs: what you tried and why, the harness-fidelity check you ran
first, the exact valid/test numbers, and a clear **PROMOTE** or **REJECT** verdict with reasoning.

**Do not touch**, in the shared repo: `experiments/LEDGER.md`, `SUBMISSION.md`, `DEVPOST.md`,
`make_submission.py`, `submission.csv`, or any existing `iter1`-`iter44` folder. These get merged
by us afterward, from your `RESULT.md` files — editing them yourself risks a conflict we'd have
to untangle under time pressure.

## 8. Handing back

When you're done (or when you run out of time — partial results are fine, a documented REJECT is
a real result), write a single `XUXIA_SUMMARY.md` at the repo root: for each of 6a/6b/6c, one
paragraph — what you tried, the final valid/test numbers, and PROMOTE/REJECT with the one-line
reason. Send that file (or its contents) back to us directly; we'll merge anything that clears
the promotion bar (Section 3) into the master ledger ourselves.

Deadline: **1 Sep 2026, 12:00 SGT**. If you're not going to finish all three, prioritize 6a and
6b over 6c — 6c is explicitly the highest-risk, most-okay-to-fail of the three.

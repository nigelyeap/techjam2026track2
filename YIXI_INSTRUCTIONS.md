# Instructions for Yixi's Claude instance — Track 2 continuation, "GBM diversification & feature depth"

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
ground before the deadline: you're getting one third, a second friend (Xuxia) is getting another
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
  than real prediction quality. If you land on a model that scores unusually well, check tie
  density and compare against an all-constant-score baseline (which should score at the trivial
  random floor, ~0.483). See `experiments/iter44_gbm_native_features/diag_ties.py` for the exact
  pattern we used.
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
| 31, 36 | Multi-task learning (auxiliary heads on click/like/follow/comment/forward) — two independent attempts, the second with a finite-difference-verified gradient fix for a bug found in the first | **REJECT both** — diagnosed as gradient conflict through FM's shared embedding table between the BPR objective and the auxiliary losses |
| 32, 34, 40 | Sequence/attention modeling (DIN/SIM-style attention over user history) — three independent attempts, the last a full PyTorch autodiff reimplementation | **REJECT all three** — the existing recency-decay features already capture most of the recency signal attention would otherwise learn |
| 38 | Score-level 5-seed ensemble of iter27's exact config (average sigmoid scores, not just average metrics) | PROMOTE — valid 0.63988 / test 0.64187 — new best pre-GBM |
| 39 | Listwise (grouped-softmax) loss vs. pairwise BPR | REJECT — diagnosed as resampling-noise instability, not a genuine ceiling |
| 41 | LightGBM ranker **on FM's bucketed feature encoding** | REJECT — standalone valid 0.6319 vs FM's 0.6389; blending added nothing (best blend regressed test) |
| 42 | 7th causal feature: engagement-event decay (like/follow/comment/forward) | REJECT — engagement events too sparse, diluted embedding capacity |
| 43 | CatBoostRanker, **also on FM's bucketed encoding** | REJECT — even worse than LightGBM's bucketed attempt (valid 0.60941 default, 0.61241 best-tuned); diagnosed as the same bucketing bottleneck, **never retested on the native encoding below** |
| **44** | **Root-cause fix**: GBM trained on a **native, un-bucketed** encoding of the same causal features (true categoricals stay `category` dtype; continuous signals — `duration_ms`, decay rates, `lastk_rate`, `gap` — passed as raw floats instead of FM's forced 20-bucket discretization) | **PROMOTE — new best.** Single LightGBM `num_leaves=2` ranker alone: valid 0.66135 / test 0.64794. Blended with iter38's FM ensemble (alpha=0.1, 90% GBM weight): **valid 0.66473 / test 0.65197.** |

**Why iter41/43 were wrong to read as "GBMs don't work here":** both were forced through FM's
`encode_ext` pipeline, which discretizes every continuous signal into `n_buckets=20` categorical
buckets — necessary for FM's embedding lookups, but actively throwing away the exact
ordering/magnitude information a GBM's split-finding is built to exploit. iter44 gave the GBM its
own native encoding instead, and the result reversed a REJECT into the current best. The lesson,
directly relevant to your Section 6 methods: **a REJECT verdict is only as good as the assumption
it was tested under.** Check what representation something was actually tested on before treating
"we tried that" as closed.

Because iter44's gain was unusually large (bigger than anything else in the whole project), it
went through 5 separate verification passes before being trusted: a tie-artifact check, a check
for an accidentally-added confound feature (`duration_ms`), 3-seed robustness, a date-shift
robustness check, and a check for whether ensembling multiple GBM seeds added anything (it
didn't — a single seed is enough, see `experiments/iter44_gbm_native_features/RESULT.md`
"Verification 4/5"). Read that file if you want the full worked example of the verification
discipline in Section 3 applied end to end.

## 5. Reproduce the current best (do this before anything else)

```bash
cd kuairand-starter-kit
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

Read `experiments/iter44_gbm_native_features/train.py` (the `prepare()`/`run()` functions) to
understand the native-feature encoding — you'll be extending it, not reinventing it.

## 6. Your three assigned methods: "GBM diversification & feature depth"

Work through these roughly in order, but feel free to reorder if one looks like a dead end early —
just don't skip the causality/harness-fidelity checks from Section 3 to save time. A clean REJECT
with a documented reason is a useful, real result; a rushed, unverified PROMOTE is not.

### 6a. XGBoost-native ranker

We've now tried LightGBM (twice — bucketed REJECT in iter41, native PROMOTE in iter44) and
CatBoost (once — bucketed REJECT in iter43, **never retested on the native encoding**). XGBoost
has never been tried at all. Repeat iter44's exact recipe (same native encoding, same feature
set: `duration_ms, decay_rate_2.5, decay_act_2.5, decay_tab_3, lastk_rate, gap` plus the 5
categoricals) but with `xgboost.XGBRanker` (`objective='rank:pairwise'` or `rank:ndcg` — try
both, pick on valid) instead of `LGBMRanker`. Sweep tree capacity the same way iter44 did
(`num_leaves`-equivalent is `max_depth` for XGBoost — sweep it down toward very shallow trees,
mirroring iter44's finding that shrinking capacity kept helping all the way to the floor).
Compare against iter44's LightGBM standalone (valid 0.66135) and, if it's competitive or better,
try blending its scores in with the existing iter44 blend (3-way blend: FM + LightGBM + XGBoost)
using the same min-max-normalize-then-linear-combine approach as `blend.py`/`make_submission.py`,
sweeping the weights on valid.

**Success criterion:** a documented valid/test number for standalone XGBoost-native, and if a
3-way blend beats the current 0.66473 valid by ≥0.001 (5-seed confirmed), that's a promotable
finding.

### 6b. Feature-engineering depth on the native-GBM encoding

iter44's causal feature set was carried over unchanged from iter24/27 (FM-era decisions). Since
the GBM now has its own native encoding that can use continuous signals directly, there's likely
room that was never explored because FM's bucketing would have thrown it away anyway. Concretely:

- Sweep additional halflife values beyond the existing 2.5-day and 3-day decay halflives (e.g.
  1d, 5d, 7d, 14d) for the decay-rate features, both individually and as extra parallel features
  (not just replacements) — a GBM can use multiple halflives as separate split candidates in a
  way an FM's shared embedding space couldn't cheaply.
- Add author-popularity-decay and video-popularity-decay features (same exponential-decay
  machinery as `compute_decay_features` in `data_ext.py`, but aggregated by `author_id`/`video_id`
  instead of by `user_id` — i.e. "how much has this author's/video's engagement been trending,"
  not "how engaged is this user").
- Try 2-3 simple pairwise cross-features that a GBM's native splits can't easily reconstruct on
  its own from the raw features (e.g. `decay_rate_2.5 / (duration_ms + eps)`, or a
  user-activity-tier × tab interaction).

Add these one at a time (or in small groups), always with the causality check from Section 3 —
anything aggregated by `author_id`/`video_id` needs the same train-only, no-leakage treatment as
the user-level decay features already get.

**Success criterion:** any single addition or small combination that beats iter44's standalone
GBM (valid 0.66135) by ≥0.001, confirmed across 5 seeds.

### 6c. Lightweight attention-pooling feature fed into the native-GBM encoding

Standalone attention/sequence models have been rejected three independent times in this project
(iter32, iter34, iter40) as full end-to-end models — the conclusion each time was that the
existing decay features already capture most of the useful recency signal that attention would
learn. **Do not repeat those as standalone models — that's been tried enough.** Instead, test a
narrower, different hypothesis: can a *small, frozen or lightly-trained* attention-pooling
summary of a user's raw interaction sequence produce a single new scalar/small-vector *feature*
that a GBM can use alongside its existing features, even though a full attention *model*
couldn't beat FM on its own? This is meaningfully different from what's been tried: it's feature
engineering for a different, non-embedding-based model family, not another standalone
attention/embedding model competing on the same terms as FM.

Concretely: compute a simple additive or scaled-dot-product attention pool over each user's last
K interactions (K=20-40, similar to iter40's window), using fixed or cheaply-learned weights
(e.g. a single learned query vector, trained separately and quickly — this does not need to be
end-to-end differentiable with the GBM), producing one or a few extra scalar features per row.
Add them to the native-GBM feature set from Section 6b and see if the GBM benefits from having
this pre-computed summary available as a split candidate, even though the previous attempts
found attention couldn't beat FM as a full replacement model.

**Success criterion:** this is the highest-risk of your three — a clean REJECT with a clear
diagnosis (e.g. "the pooled feature correlates too highly with the existing decay features, no
new information") is a perfectly good outcome here. Don't force a positive result if the signal
isn't there; document why.

## 7. Output format — match our schema exactly (this matters for merging your results back)

For each of 6a/6b/6c, create a folder `experiments/iterYIXI1_xgboost_native/`,
`experiments/iterYIXI2_feature_depth/`, `experiments/iterYIXI3_attention_pool_feature/`
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
a real result), write a single `YIXI_SUMMARY.md` at the repo root: for each of 6a/6b/6c, one
paragraph — what you tried, the final valid/test numbers, and PROMOTE/REJECT with the one-line
reason. Send that file (or its contents) back to us directly; we'll merge anything that clears
the promotion bar (Section 3) into the master ledger ourselves.

Deadline: **1 Sep 2026, 12:00 SGT**. If you're not going to finish all three, prioritize 6a and
6b over 6c — 6c is explicitly the highest-risk, most-okay-to-fail of the three.

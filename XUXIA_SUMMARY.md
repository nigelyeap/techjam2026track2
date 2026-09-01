# Xuxia's Claude instance — Track 2 continuation summary

## Environment note (read this first)

`XUXIA_INSTRUCTIONS.md`'s Section 5 harness-fidelity reference is
`iter44 blend: valid=0.66473 test=0.65197`. Cloning the repo fresh and
running `python3 make_submission.py` instead reproduced `iter63 blend:
valid=0.67606 test=0.65955` — `main` had advanced 19 commits past iter44
between when the instructions were written and when this work started
(iter45-iter63: `linear_tree=True`, `learning_rate=0.10` retune, a
decayed per-tab rate feature, alpha retuned 0.1→0.14). The FM ensemble
standalone number (valid=0.63988/test=0.64187) reproduced exactly, so the
environment itself was not the problem — just a stale baseline reference
in the handoff doc. Per direct user instruction, **iter63's numbers are
used as the baseline to beat throughout this file**, not iter44's. All
three methods below reproduced this iter63 baseline via their own
harness-fidelity checks before any new result was trusted.

## 6a — per-segment blend alpha (by `tab` and by activity tertile)

Tried replacing the single global blend alpha (0.14) with per-segment
alphas, swept independently on valid then applied to the matching test
rows, segmented two ways: by `tab` (15 distinct values) and by tertiles of
the decayed-activity feature. Both regressed relative to the global alpha
in every one of 5 GBM seeds (tab: mean −0.00499 valid, 0/5 seeds ≥+0.001;
tier: mean −0.00568 valid, 0/5 seeds ≥+0.001). The tab result traces to
overfitting on small segments (several tabs have <300 valid rows and land
on degenerate grid-boundary alphas); the tier result is more informative
since all three tiers are large and balanced (~41.6k rows each) yet still
regress, suggesting no exploitable GBM/FM heterogeneity across activity
levels — the single global alpha is already close to a real optimum.
**Verdict: REJECT.** Full detail: `experiments/iterXUXIA1_segment_blend/RESULT.md`.

## 6b — rank-based / calibrated blending

Tried two alternatives to the current linear+minmax blend: rank-average
(Borda) fusion and reciprocal-rank fusion (both vs. the current blend),
and isotonic-regression calibration (fit on train only). All three
regressed relative to the current blend in every one of 5 GBM seeds.
Rank fusion lost a small, consistent margin (Borda mean −0.00146 valid,
RRF mean −0.00133) — plausible, since ranks throw away magnitude
information the linear blend still uses. Isotonic calibration collapsed
catastrophically (mean −0.03584 valid): a follow-up diagnostic found
isotonic regression pools the GBM's ~123k-unique-value raw score into
just **37** distinct calibrated levels (vs. FM's much gentler 278), a
tie-artifact that alone drags GBM standalone from 0.67168 to 0.54189 —
a structural mismatch between lambdarank-style raw scores and isotonic's
monotonic-step-function calibration, not a tuning problem.
**Verdict: REJECT (both alternatives)** — a clean, useful null result:
the current linear+minmax blend is already close to what any of these
fusion strategies can extract from these two models' scores. Full detail:
`experiments/iterXUXIA2_calibrated_blend/RESULT.md`.

## 6c — reopening multi-task learning under the GBM-native representation

Tried approach 1 (auxiliary features via stacking, not auxiliary losses):
trained 4 small LightGBM classifiers to predict `is_like`/`is_follow`/
`is_comment`/`is_forward` from the same native feature set (5-fold OOF on
train, full-train-fit on valid/test — leakage-free by construction, with
an independent alignment spot-check against a fresh CSV re-parse), then
fed their predictions back into the main `long_view` GBM as 4 new columns.
Single-seed result: valid 0.67168→0.67192 (**+0.00024**), below Section
3's 0.0003 "even look twice" threshold, so no 5-seed confirmation was run
(per protocol). Feature importance independently confirms the same
conclusion: across the 48 trees actually built, 3 of the 4 auxiliary
columns were never chosen as a split feature at all, and the 4th
(`aux_is_like`) only once. The original iter31/36 REJECT mechanism
(shared-embedding gradient conflict) doesn't apply here — there's no
shared loss/score in this stacking design — but the auxiliary signals
still add essentially nothing, pointing to the engagement signals
themselves being low-information for `long_view` prediction rather than
an architecture-specific artifact. **Verdict: REJECT.** Approach 2 (true
joint multi-task training) was not attempted, per the instructions'
guidance to only pursue it if approach 1 showed a promising signal. Full
detail: `experiments/iterXUXIA3_multitask_gbm/RESULT.md`.

## Overall

All three methods produced clean, well-diagnosed REJECTs — no PROMOTE to
report. The current best (iter63 blend: valid=0.67606/test=0.65955,
alpha=0.14) is unchanged. Nothing in `experiments/LEDGER.md`,
`SUBMISSION.md`, `DEVPOST.md`, `make_submission.py`, `submission.csv`, or
any pre-existing `iterN_*` folder was touched — all new work lives in
`experiments/iterXUXIA1_segment_blend/`, `experiments/iterXUXIA2_calibrated_blend/`,
and `experiments/iterXUXIA3_multitask_gbm/`.

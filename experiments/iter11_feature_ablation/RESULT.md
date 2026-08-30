# iter11 — feature ablation / minimal effective set (on top of iter9)

## Question
iter9 tested only `{activity}`, `{activity,tab}`, and `{activity,tab,rate}` (all at
Laplace smoothing `ALPHA=1.0`, hardcoded). It never isolated `rate` on its own or
`{tab,rate}` without `activity`, and never tuned `ALPHA`. iter11 fills those gaps to
find the minimal feature set that captures iter9's gain, and to check whether the
smoothing constant matters.

## Setup
`experiments/iter11_feature_ablation/{data_ext.py,train.py}` are copies of iter9's
files (iter9's originals untouched), with two additions:
- `encode_ext()` / `run_bpr_ext()` take an `alpha` kwarg (was a hardcoded module
  constant `ALPHA=1.0` in iter9), so the Laplace smoothing constant in `rate`'s
  formula `(prior_pos+alpha)/(prior_total+2*alpha)` can be swept.
- `run_bpr_ext()` accepts an optional pre-loaded `splits` dict so a sweep driver
  (`sweep.py`) can call `load_ext()` once and reuse it across all configs/seeds.
Everything else (FM, activity-weighted BPR loss, hyperparams) is unchanged from
iter9/iter3.

## Phase 1 — feature-set ablation (3 seeds: 0,1,2; alpha=1.0)
| feature set | valid primary (mean) | test primary (mean) | source |
|---|---|---|---|
| activity only | 0.6031 | 0.5967 | iter9 (cited, not rerun) |
| activity + tab | 0.6046 | 0.5986 | iter9 (cited, not rerun) |
| **rate only** | **0.60905** | **0.60348** | iter11 (new) |
| **activity + rate** | **0.60985** | **0.60460** | iter11 (new) |
| **tab + rate** | **0.61045** | **0.60530** | iter11 (new) |
| activity + tab + rate | 0.6103 | 0.6057 | iter9 (cited, not rerun) |

Key finding: **`rate` alone (a single field) already beats `activity+tab` combined**
(0.60905 vs 0.6046 valid) and gets to within 0.0013 of the full 3-feature combo.
Adding either `tab` or `activity` to `rate` closes almost all of the remaining gap,
and `tab+rate` (dropping `activity` entirely) is *higher* than the full 3-feature
combo's 3-seed number (0.61045 vs 0.6103). This confirms and sharpens iter9's own
conclusion — `rate` is the dominant signal, `activity`/`tab` mostly just give it a
non-degenerate context — and suggests `activity` may be largely redundant once
`tab`+`rate` are both present.

## Phase 2 — Laplace smoothing sweep (3 seeds: 0,1,2, at best Phase-1 combo = tab+rate)
| alpha | valid primary (mean) | test primary (mean) |
|---|---|---|
| 0.5 | 0.61015 | 0.60485 |
| 1.0 | 0.61045 | 0.60530 |
| **2.0** | **0.61107** | 0.60522 |
| 5.0 | 0.61034 | 0.60518 |

alpha=2.0 edges out alpha=1.0 on 3-seed valid by +0.0006, but the sweep is fairly
flat (all four alphas land within ~0.001 valid of each other) — this axis looks
close to noise-level, similar to iter7's finding that iter3's original
hyperparameters were already near-optimal.

## 5-seed confirmation: tab+rate, alpha=2.0 vs iter9's activity+tab+rate, alpha=1.0
Per the task protocol, since `tab+rate @ alpha=2.0` had the best 3-seed valid mean
of everything tested, seeds 3 and 4 were run to complete a 5-seed set:

| seed | tab+rate α=2.0 valid | tab+rate α=2.0 test | iter9 (activity+tab+rate, α=1.0) valid | iter9 test |
|---|---|---|---|---|
| 0 | 0.61053 | 0.60544 | 0.61028 | 0.60562 |
| 1 | 0.61166 | 0.60556 | 0.61038 | 0.60567 |
| 2 | 0.61102 | 0.60465 | 0.61006 | 0.60585 |
| 3 | 0.61054 | 0.60372 | 0.60963 | 0.60525 |
| 4 | 0.61072 | 0.60489 | 0.61029 | 0.60559 |
| **mean** | **0.61090** | **0.60485** | **0.61013** | **0.60560** |
| **std** | 0.00042 | 0.00066 | 0.00027 | 0.00020 |

**Valid**: tab+rate@α=2.0 wins by +0.00077, and wins in all 5/5 individual seed
pairs (a consistent, if small, direction).

**Test**: iter9's original config wins by +0.00075 (i.e. tab+rate@α=2.0 is *worse*
on test), also in all 5/5 individual seed pairs — the same magnitude, opposite
direction.

## Verdict: **iter9's exact config (activity+tab+rate, alpha=1.0) remains current
best — NOT promoting the leaner tab+rate/alpha=2.0 config**, despite it nominally
winning on valid.

Reasoning:
1. **Magnitude is below the noise-worthy threshold.** +0.00077 valid is roughly
   1.5-2x the combined std of the two configs, and far below the ~0.002 mark this
   AutoML run treats as a "real" jump elsewhere (e.g. iter9's own +0.0075 vs iter3
   was 25-45x noise). This is a small, marginal effect, not a clear signal.
2. **Test disagrees, by the same magnitude, in the same consistent way.** A
   genuine improvement should not systematically reverse sign between valid and
   test when both are 5-seed means with non-overlapping data. The fact that valid
   consistently favors the leaner config while test consistently favors iter9's
   is the signature of mild overfitting to valid via the alpha grid search
   (alpha=2.0 was picked from a 4-point grid specifically because it scored
   highest on a 3-seed valid estimate — a classic winner's-curse setup) rather
   than a true generalizable gain.
3. Per protocol, model selection is valid-only and test is never used to pick
   among configs — that was followed correctly here (alpha and feature-set choice
   were both made on valid). But the RESULT writeup's job is to report both
   splits honestly for whoever reviews/promotes; the test regression is a
   legitimate reason for caution even though it wasn't used to *choose* a config.

**Positive finding worth keeping regardless of promotion status**: `rate` is by
far the single most important of iter9's three added features — it alone beats
`activity+tab` combined, and `tab+rate` (2 fields, dropping `activity`) gets
statistically indistinguishable performance from the full 3-feature set. If a
future iteration needs to reduce feature count/compute (e.g. for a production
constraint), `{tab, rate}` is a defensible near-minimal substitute for
`{activity, tab, rate}` — but on pure metric grounds with current evidence, there
is no case to switch away from iter9's original config.

The Laplace-smoothing-constant axis (alpha in {0.5, 1.0, 2.0, 5.0}) is
inconclusive/flat and not worth pursuing further, consistent with iter7's finding
that this run's other hyperparameter axes are near-exhausted.

## Code
`experiments/iter11_feature_ablation/{data_ext.py,train.py,sweep.py}`,
raw sweep results in `experiments/iter11_feature_ablation/sweep_results.json`
(18 sweep configs + 2 extra confirmation seeds = 20 rows total).

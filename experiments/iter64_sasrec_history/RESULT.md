# iter64 — SASRec-style self-attention history encoder (independent model, score-blended)

## Hypothesis

iter40's own closing diagnosis (three prior sequence-modeling attempts —
iter32, iter34, iter40 — all REJECTed) pinned the failure mode on
*mechanism*, not on sequence modeling being inherently useless for this
task: routing a DIN-style attention signal back through the FM's own shared
item-embedding table created a second gradient path that conflicted with
the FM's rank-invariant BPR objective. iter40 explicitly named "a non-FM-
family model" combined only by score-blending as the untried, structurally
different lever. Following the user's explicit permission to use any
open-source library/paper/pretrained weights, this iteration builds exactly
that: a SASRec-style (Kang & McAuley, 2018) self-attentive sequence encoder
over each user's own causally-ordered interaction history, trained fully
standalone (own item-embedding table, own BPR loss, own optimizer — no
parameter or gradient sharing with the FM or GBM), combined with the
current best (iter63) blend only via post-hoc score-blending.

## Implementation

- `data_ext.py`: builds per-row, causally-ordered history sequences. Same
  `(time_ms, orig_idx)` strict total-order discipline as iter18's
  `compute_momentum_features` (imported, not copied): for each user, sort
  ALL rows across train+valid+test by `(time_ms, orig_idx)`; a row's history
  is the video_ids of that user's rows strictly earlier in this order
  (right-aligned, left-padded to `max_len=20`), read BEFORE the row's own
  item is pushed onto the rolling window. Item-id vocabulary fit on TRAIN
  rows only (UNK fallback), matching `data.py`'s own `encode()` convention.
- `model.py`: item embedding (`padding_idx=0`, forced-zero pad vector) +
  learned positional embedding → one manual multi-head scaled-dot-product
  self-attention block (mask-safe: an all-padding row — a user's first-ever
  interaction — produces a well-defined zero context vector, no NaN) →
  masked mean-pool over real positions → dot product with the candidate
  item's own embedding = score.
- `train.py`: reuses `build_pos_neg_index`/`sample_pairs`
  (`iter27_triple_fusion/train.py`, imported verbatim) for the same
  per-user positive/negative BPR sampling scheme the FM itself uses, TRAIN
  split only. Adam, early-stopped on valid primary (patience=4).

**Causality verification** (`verify_causality.py`): 900 rows sampled across
train/valid/test, each row's history independently rebuilt from scratch
(brute-force per-user resort + prefix slice) and compared field-for-field
against the vectorized output. 0/900 mismatches.

## Standalone result (seed 0, d=32, 2 heads, dropout=0.2, lr=1e-3, bs=4096,
max_len=20, 25 epochs capped / patience=4, best at epoch 4)

| | valid | test |
|---|---|---|
| SASRec standalone | 0.58333 | 0.57797 |
| iter63 GBM standalone (for reference) | 0.67168 | 0.65353 |
| iter38 FM ensemble standalone (for reference) | 0.63988 | 0.64187 |

Standalone score is well below not just the GBM/FM standalone scores but
also the project's original raw FM baseline (test 0.5946) — and the
training curve shows classic overfitting to a weak signal (train loss falls
monotonically epoch 1→8, 0.627→0.503; valid peaks at epoch 4 then degrades
every epoch after).

## Blend-level check (`blend_check2.py`, pure-numpy re-check against the
unchanged iter63 blend, sweeping `beta`: `(1-beta)*iter63_blend +
beta*SASRec`, both min-max normalized)

| beta | valid | test |
|---|---|---|
| 0.00 (no SASRec) | **0.67606** | **0.65955** |
| 0.02 | 0.67182 | 0.65801 |
| 0.05 | 0.66191 | 0.65210 |
| 0.08 | 0.65317 | 0.64615 |
| 0.10 | 0.64840 | 0.64277 |
| 0.15 | 0.63900 | 0.63528 |
| 0.20 | 0.63177 | 0.62943 |
| 0.30 | 0.62059 | 0.61821 |

**Monotonically decreasing at every tested beta > 0, on both valid and
test.** This is a clean, unambiguous reject — not a borderline result that
needs 5-seed confirmation (the project's 5-seed-confirm bar exists to guard
against a promising-looking *positive* single-seed result being noise; a
monotonic degradation with no interior optimum near zero is not that case).
The SASRec model's errors are not sufficiently uncorrelated with (or are
actively worse-calibrated than) the existing blend's to help even at a
tiny weight — unlike the FM, whose much-lower standalone score (0.63988)
still contributes a real gain at 14% blend weight, because its errors are
genuinely complementary to the GBM's.

## Diagnosis

Unlike iter32/34/40 (which shared the FM's embedding table and were
REJECTed for a gradient-conflict reason), this model was fully independent
by construction — ruling out that specific failure mode. The actual
bottleneck here looks more mundane: a from-scratch item-embedding table
learned purely from BPR pairs, with only 7,551 distinct videos and ~1.1M
train rows split across ~27k users, an average of ~40 history events per
user (most well under `max_len=20`), and no side information (author_id,
tab, duration) — the model has to learn everything about an item from
co-occurrence in BPR pairs alone, with far less signal per parameter than
the FM (which uses `user_id × video_id × author_id × tab × dur_bucket`,
richer even before recency features) or the GBM (rich causal + decay
features). Likely underpowered given the data scale, not fundamentally
wrong as an architecture — but fixing that (richer per-item side features
in the attention block, a next-item pretraining phase before BPR
fine-tuning, larger embeddings) is a multi-experiment undertaking that
doesn't fit the remaining time budget for this cycle.

## Infrastructure note (worth recording for any future PyTorch experiment
in this repo)

Importing `torch` and then training a LightGBM model **in the same Python
process** reliably segfaults (exit code 139) in this environment
(`lightgbm==4.7.0`, `torch==2.13.0`, Python 3.14, miniconda on macOS) — the
process dies silently mid-training with no Python traceback, only
recognizable via the OS exit code. Setting `KMP_DUPLICATE_LIB_OK=TRUE` did
**not** fix it (ruled out as a simple OpenMP-duplicate-init issue). Worked
around by splitting GBM/FM training and SASRec training into two separate
processes (`gen_iter63_scores.py`, torch-free; `gen_sasrec_scores.py`,
lightgbm-free) that each save raw score arrays to `.npz`, combined by a
third, pure-numpy process (`blend_check2.py`). Any future experiment mixing
`torch` with `lightgbm`/`catboost` in this repo should keep them in
separate processes from the start.

## Verdict: **REJECT**

Standalone score far below both existing model families and even the
original baseline; blend-level effect is a clean, monotonic negative at
every tested weight. Mechanistically different from all three prior
sequence-modeling attempts (fully independent model, no shared gradient
path) — this rules out iter40's specific "shared-embedding-table conflict"
diagnosis as the cause here, and points instead at the model being
underpowered for the available data/feature scale, a separate and much
larger undertaking to fix than fits the remaining time budget.

Full artifacts: `experiments/iter64_sasrec_history/{data_ext.py,model.py,
train.py,verify_causality.py,gen_iter63_scores.py,gen_sasrec_scores.py,
blend_check2.py,blend_check2_results.json,gen_iter63_scores.log,
gen_sasrec0.log}`.

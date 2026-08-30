# iter31 — Multi-task auxiliary loss (is_click/is_like/is_follow/is_comment/is_forward)

## Idea

The README's "未探索" (unexplored) headroom list, item 3, and the official
Track 2 problem statement both name **multi-task learning** as an
organizer-suggested, never-tried direction: KuaiRand's logs carry
`is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward` labels
alongside the main `long_view` label used for the primary metric. This is
the first of 31 iterations to use any of them.

## Method

Starts from **iter24's exact feature set and pipeline** (current best,
valid 0.63251 5-seed): `decay_rate_2.5`, `decay_act_2.5`, `decay_tab_3`,
`last1`, `lastk_rate`, `gap`, on top of the base fields
`user_id/video_id/author_id/tab/dur_bucket`. `experiments/iter31_multitask/data_ext.py`
is a **verbatim copy** of iter24's `data_ext.py` with one addition:
`load_aux_labels(data_dir)`, which reads the same two raw log CSVs
(`log_standard_4_08_to_4_21_pure.csv`, `log_standard_4_22_to_5_08_pure.csv`)
already used by `data.py`/`_load_raw_time`, and exposes the 5 auxiliary
columns — currently dropped by the starter kit's loader — as
per-split arrays aligned 1:1 with `encode_ext`'s row order (see its
docstring for the exact alignment argument, and PART E of `data_ext.py`'s
`__main__` self-test for a brute-force verification against a fresh,
independent CSV read).

**Model / training design** (`experiments/iter31_multitask/train.py`,
`mtl_bpr_step`): the simplest faithful multi-task design for a
linear/FM model trained via manual numpy gradients — **no new
architecture, no new parameters, no new forward pass**. The exact same FM
logit `z = b + W[X].sum + inter(V)` that the main BPR loss already computes
for a sampled `(Xpos, Xneg)` pair is *also* used as the score for a
pointwise logistic (BCE) loss against each of the 5 auxiliary engagement
labels, on the **same rows already sampled for the BPR pair this step**.
The auxiliary BCE gradient (`sigmoid(z) - y_aux`, averaged over the 5
tasks so `aux_weight`'s scale doesn't depend on task count) is scaled by a
single scalar `aux_weight` (swept 0.1/0.2/0.3) and summed directly into
the same gradient tensors (`gV`, `gW`, `g` for `b`) that the BPR gradient
populates, **before** the single shared Adam update — i.e. one Adam step
per training iteration optimizing
`L = L_bpr(long_view) + aux_weight · mean_t BCE_t(engagement_t)`.
This shares representations (`V`, the user/item latent factors — the
actual point of multi-task learning) across the main ranking task and the
5 auxiliary engagement tasks, without introducing a second scoring head.

Base engagement rates on the train split (1,141,112 rows): `is_click`
46.35%, `is_like` 1.87%, `is_follow` 0.10%, `is_comment` 0.26%,
`is_forward` 0.10%, vs. `long_view` 33.66%. `is_click`/`is_like` are dense
enough to carry real signal; `is_follow`/`is_comment`/`is_forward` are
sparse but still have 1,100-2,900 positive rows each in-window, so
contribute a (small) nonzero gradient rather than being pure noise.

## Causality / leakage argument

Per protocol, any new signal touching training needs a leakage argument.
The auxiliary labels here are **train-split-only signals feeding
gradients during training** — never features, never seen at inference —
by the same construction as iter22/23's decay-aware BPR sampling weight:

1. `load_aux_labels()` is a **separate code path** from `encode_ext()`.
   The 5 auxiliary arrays are never joined into any `X` feature matrix,
   for any split, anywhere in this codebase.
2. `train.py`'s `mtl_bpr_step` only ever reads `aux_mat[Xpos_rows]` /
   `aux_mat[Xneg_rows]`, where `aux_mat` is built exclusively from
   `aux_cache['train']` and `Xpos_rows`/`Xneg_rows` are themselves drawn
   from `build_pos_neg_index(ytr, utr)` — i.e. indices into the **train**
   split only (same index space `bpr_step`/`sample_pairs` have always used
   since iter3).
3. Evaluation (`evaluate(uva, yva, m.predict(Xva))` / `evaluate(ute, yte,
   m.predict(Xte))`) calls only `m.predict`, i.e. the frozen FM's own
   logits on the feature matrix — it has no reference to any `aux` array
   at all, for any split. There is no code path by which `aux['valid']` or
   `aux['test']` could reach a prediction; they aren't even read by
   `run_bpr_ext` except to build `aux_mat` from `aux_cache['train']`.
4. Verified empirically: PART E of `data_ext.py`'s self-test spot-checks
   20 random train rows against a wholly independent fresh CSV read,
   confirming index alignment, plus cross-checks `user_id`/`video_id`
   against the causal-feature row tuple to rule out a coincidental label
   match.

This is safe by construction, not merely empirically — the same argument
shape the ledger has used for iter22/23's training-time-only sampling
weight.

## Harness-fidelity check

With `aux_weight=0.0`, `run_bpr_ext` uses the **unmodified** `bpr_step`
(byte-for-byte copy of iter24's), so the entire multi-task code path is
bypassed. Verified against iter24's own published per-seed numbers
(`experiments/iter24_decay_tab_refine/results.json`, tag
`decay_rate_2.5+decay_act_2.5+decay_tab_3+mom`):

| seed | valid (iter31, aux_weight=0) | valid (iter24 ref) | test (iter31) | test (iter24 ref) |
|---|---|---|---|---|
| 0 | 0.6326001882553101 | 0.6326001882553101 | 0.6283901929855347 | 0.6283901929855347 |

All 3 seeds are **bit-exact** matches (max abs err `0.00e+00` on both
valid and test, per `driver_log.txt`'s Phase 0 output):

| seed | valid (iter31, aux_weight=0) | valid (iter24 ref) | test (iter31) | test (iter24 ref) |
|---|---|---|---|---|
| 0 | 0.6326001883 | 0.6326001883 | 0.6283901930 | 0.6283901930 |
| 1 | 0.6330777407 | 0.6330777407 | 0.6294211149 | 0.6294211149 |
| 2 | 0.6317896843 | 0.6317896843 | 0.6268689632 | 0.6268689632 |

Harness confirmed non-drifted before evaluating any multi-task change.

## Sweep: aux_weight (all 5 tasks, equal-weighted mean BCE), 3 seeds

Selection is on **valid only** per protocol; test is reported here for
completeness but was never used to pick a config.

| config | aux_weight | valid mean (3-seed) | valid std | test mean (3-seed) | test std | Δ valid vs iter24 (0.63251) |
|---|---|---|---|---|---|---|
| harness (reproduction) | 0.0 | 0.63249 | 0.00053 | 0.62823 | 0.00105 | −0.00002 (noise) |
| mtl_all5_w0.1 | 0.1 | 0.62713 | 0.00053 | 0.62700 | 0.00046 | **−0.00538** |
| mtl_all5_w0.2 | 0.2 | 0.62390 | 0.00052 | 0.62171 | 0.00137 | **−0.00861** |
| mtl_all5_w0.3 | 0.3 | 0.62006 | 0.00055 | 0.61825 | 0.00084 | **−0.01245** |

Per-seed values in `results.json`. The regression is **monotonic and
consistent across all 3 seeds at every weight tested** — no seed at any
`aux_weight > 0` comes close to a harness-level seed, and the gap widens
smoothly and monotonically as `aux_weight` increases from 0.1 to 0.3, with
no sign of a beneficial region or a crossover back toward parity. Both
splits move in the same direction at every weight (no valid/test
disagreement of the kind seen in the Round 7 four-way crossover) — this is
a real, unambiguous regression, not overfitting-to-valid noise.

Given a margin of **−0.00538 at the least-harmful weight tested** (already
~10x iter24's own valid std of 0.00050, in the wrong direction) and a
strictly worsening trend with no crossover in sight, there is no case for
extending to smaller weights, alternative task subsets, or a 5-seed
confirmation — the sweep result is already unambiguous with 3 seeds per
point and a clean monotonic trend, exactly the situation the protocol
allows reporting as a 3-seed non-promotion without further runs.

## Diagnosis

The regression is monotonic in `aux_weight`, present on both splits, and
present at every seed — this points to a genuine objective-conflict
between the ranking task and the auxiliary tasks under this particular
sharing scheme, not noise or undertraining. The most likely mechanism:

`mtl_bpr_step` shares the **exact same absolute score** `z` between the
BPR pairwise loss (which only ever needs `z_pos` and `z_neg` to be
correctly *ordered*, and is invariant to any shift/rescaling of `z` that
preserves that order) and the pointwise BCE terms (which need `sigmoid(z)`
to sit at an *absolute* calibrated probability matching each auxiliary
label's own base rate). `is_click` alone has a 46.35% train base rate —
roughly **14 points higher** than `long_view`'s 33.66% — so the shared
`z` is being pulled, every single step, toward a scale/bias that makes
`sigmoid(z)` hover near "typically clicked," which is a much less
selective, much higher-recall signal than "will be long-viewed." Because
click, and even like/follow/comment/forward to a lesser extent, are only
loosely correlated with `long_view` at the individual-row level (most
clicks do not lead to a long view), calibrating the *same* absolute score
to also satisfy the click-rate prior actively fights the BPR objective's
need for well-separated relative scores within a user's own exposure set
— rather than reinforcing it. This is consistent with the effect scaling
smoothly with `aux_weight`: more weight, more pull toward the auxiliary
tasks' calibration, more damage to the pairwise ranking signal. The dense,
high-base-rate `is_click` term likely dominates the pull (it has ~25x the
positive rows of `is_like`, ~400x of `is_follow`/`is_forward`), but the
core issue is architectural — sharing the raw scalar score across a
ranking loss and pointwise calibration losses, rather than sharing only
the underlying embeddings `V` behind two separately-scaled scores — not
merely a base-rate imbalance among the 5 auxiliary tasks that reweighting
them differently would fix.

(This diagnosis is offered as the most likely mechanism given the data in
hand; a task-specific linear head per auxiliary task — sharing `V` but
giving each task its own `W_t`/`b_t` so only the *embeddings*, not the
absolute score, are shared — is the natural next design to test this
hypothesis in a future iteration, but is out of scope for this session's
single-focused-design mandate.)

## Comparison vs iter24

No tested configuration comes within 0.005 of iter24's 5-seed valid
reference (0.63251), let alone beats it. iter24 remains current best,
unchanged.

## Code

`experiments/iter31_multitask/{data_ext.py,train.py,driver.py,results.json}`.
`data_ext.py` is iter24's feature pipeline verbatim + `load_aux_labels`.
`train.py` adds `mtl_bpr_step` alongside the untouched `bpr_step` (used for
`aux_weight=0.0`, which is how harness fidelity was verified bit-exact).

## Verdict: REJECT

Multi-task auxiliary loss (is_click/is_like/is_follow/is_comment/
is_forward, shared-score pointwise BCE design, `aux_weight` swept
0.1/0.2/0.3, all 3 seeds each) produces a real, monotonic, both-split
regression vs iter24 at every tested weight — worst case −0.01245 valid at
`aux_weight=0.3`, best case (least-harmful, `aux_weight=0.1`) still
−0.00538 valid, ~10x iter24's own seed noise. No 5-seed confirmation
run was warranted (protocol threshold for extending is a candidate
*beating* the reference by >0.001; this candidate underperforms by >5x
that margin at its best point, with a trend that only worsens at higher
weight). iter24 remains current best. The negative result is still
informative: it rules out the simplest "same-score, multiple losses"
multi-task design at this repo's model capacity, and localizes the likely
failure mode (absolute-score sharing between a rank-invariant loss and
base-rate-calibrated pointwise losses) for any future attempt at this
organizer-suggested direction.

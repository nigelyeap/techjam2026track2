# iter4 — BPR with 4 negatives per positive

## What changed vs iter2 (`experiments/iter2_bpr_uniform/train.py`)

iter2 samples one (user, positive row) pair per training example, then exactly one
random negative row from that same user, and trains with BPR loss
`-log(sigmoid(score_pos - score_neg))`.

iter4 keeps the same sampling scaffolding (`build_pos_neg_index`, uniform-user
sampling of the positive) but samples **4 negatives per positive** (with
replacement) instead of 1, via `sample_pairs_multineg`. The loss for each positive
is the *average* of its 4 pairwise BPR losses. Correspondingly, in
`bpr_step_multineg`:

- each positive row's total gradient = the **average** (not sum) of its gradient
  across the 4 `(pos, neg_i)` comparisons: `gpos = sum_j (sig(d_ij)-1)/(4*B)`
- each of the 4 negative rows individually gets **1/4** the gradient weight it
  would carry in the iter2 1:1 case

so the gradient magnitude per positive stays comparable to iter2 rather than
scaling up 4x just because more rows are touched per step (per assignment spec).

`zpos` is computed once per positive (single `m.logits(Xpos)` call over the
8192-row positive batch) and `zneg` is computed once over the flattened
`8192*4` negative rows, then reshaped to `(8192, 4)` to form all 4 pairwise
differences per positive — no redundant logits recomputation.

Everything else (FM architecture, `k=16`, `lr=0.001`, `patience=4`, Adam
optimizer, eligible-user construction) is identical to iter2.

## Batch-size choice

Positive batch size `bs=8192` is **unchanged** from iter2 (not reduced). Each
step therefore touches `8192` positives + `8192*4=32768` negatives = `40960`
rows total (2.5x iter2's `16384` rows/step). `steps_per_epoch` is still keyed off
positive-row count only (same formula as iter2: `ceil(pos_count/bs)` = 47
steps/epoch), so "one epoch" still means one pass over positives. This was the
recommended option in the assignment: more signal per step should be worth it on
its own, rather than deliberately holding total-rows-per-step fixed by shrinking
the positive batch to `2048`.

## Results, 5 seeds (0–4), `--epochs 40 --patience 4` (all early-stopped between
epoch 8–10)

| seed | valid primary | test primary |
|------|---------------|--------------|
| 0    | 0.5986        | 0.5928       |
| 1    | 0.5974        | 0.5917       |
| 2    | 0.5977        | 0.5916       |
| 3    | 0.5983        | 0.5930       |
| 4    | 0.5984        | 0.5926       |

**valid**: mean **0.5981**, std **0.00045**
**test**: mean **0.5923**, std **0.00058**

## Comparison

| | valid mean | test mean | test std |
|---|---|---|---|
| iter1 (pointwise, current best) | 0.6015 | 0.5953 | 0.0008 |
| iter2 (BPR, 1 neg/pos) | 0.5988 | 0.5923 | 0.0003 |
| **iter4 (BPR, 4 neg/pos)** | **0.5981** | **0.5923** | **0.00058** |

## Verdict

**REJECTED.** iter4's test mean (0.5923) is statistically indistinguishable from
iter2's (0.5923) — literally the same to 4 decimal places — and both are ~0.003
below iter1's 0.5953, well outside the ~0.0003–0.0008 noise floor established by
iter1/iter2. iter4's valid mean (0.5981) is even marginally *below* iter2's
(0.5988), i.e. adding more negatives per positive did not help and if anything
trended slightly the wrong way on valid (within noise of iter2, but certainly no
improvement).

Averaging 4 negatives per positive did reduce the *variance* of the single-step
BPR gradient estimate as intended (that part of the hypothesis holds), but it did
not close the gap to iter1. This supports the iter2 hypothesis in the ledger: the
mismatch is not primarily gradient noise from a single random negative, but a
training-*distribution* mismatch — uniform per-user sampling (each user
contributes pairs with equal probability regardless of activity level) vs
pointwise's implicit weighting by user/row activity. Reducing gradient noise
around an already-mismatched sampling distribution doesn't fix that distribution
mismatch, so BPR (in any of the 1-neg or 4-neg variants tried so far) still
underperforms plain pointwise FM on this task/metric.

Code: `experiments/iter4_bpr_multineg/train.py`. Per-seed logs:
`experiments/iter4_bpr_multineg/seed{0..4}.log`.

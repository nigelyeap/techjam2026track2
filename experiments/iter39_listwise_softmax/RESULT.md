# iter39 — listwise (grouped-softmax) loss vs. pairwise BPR

## Idea

The starter kit's README names loss-function choice (pointwise -> pairwise
*or listwise*) as the most-likely-to-help untried direction. Pairwise BPR
is already fused into iter27/current-best; listwise had never been tried.
Reopened as a new score-maximization angle after convergence, per explicit
instruction to keep pushing on score.

## Method

For each sampled user (group), take all their train-split positives plus a
random-subsample-capped set of negatives (`M_max=16` rows/group), compute
grouped softmax over FM logits, target `t_i = y_i / n_pos_in_group`
(uniform over positives, 0 elsewhere), loss = cross-entropy, gradient
`dL/dz_i = p_i - t_i` (standard multinomial-logistic result). All other
config (features, formula constants, decay-weighted user sampling for which
users get selected into a group) held identical to iter27, for a clean
loss-function-only ablation.

## Gradient check

Verified against finite-difference numerical gradients on a tiny synthetic
FM (2 groups, sizes 3/4) before any real run — max abs error ~1.7e-11 on V,
~1.1e-11 on W, ~5.6e-17 on b. See
`/private/tmp/claude-501/.../scratchpad/grad_check_listwise.py`.

## Result

Single-seed (seed=0) runs at 3 learning rates spanning a 10x range, full
epoch budget with early stopping (patience picks the best-valid epoch, so
the reported number already reflects the best checkpoint each run ever
saw, not a stale final epoch):

| lr | epoch-1 valid primary | best valid primary (any epoch) | pattern |
|---|---|---|---|
| 0.001 | 0.6162 | 0.6162 (epoch 1) | peaks epoch 1, monotonic decline every epoch after |
| 0.0003 | 0.6298 | 0.6298 (epoch 1) | same pattern |
| 0.0001 | 0.6380 | 0.6380 (epoch 1) | same pattern |

All three settings peak at epoch 1 and monotonically *worsen* every epoch
thereafter -- not noise, a consistent direction across a 10x lr range. The
best any setting ever reaches (0.6380, at the lowest lr, stopped after a
single epoch) is roughly at parity with iter27's BPR baseline (~0.638) but
does not exceed it, and requires stopping training almost immediately to
achieve even that -- a fragile, non-robust operating point, not a real gain.

## Diagnosis

The grouped-softmax objective is being computed over a randomly
resubsampled, capped (`M_max=16`) subset of each user's impressions every
step, unlike BPR's simple uniform pos/neg pair draw. For users with many
negatives, a different random subset is judged "the field" each time they
are sampled, so the model is chasing a moving, high-variance objective
rather than converging toward a stable ranking of the *fixed* population
BPR effectively marginalizes over via random pairwise draws. This produces
fast initial progress (the aggregate gradient direction is still roughly
correct early on) followed by a slow degradation as the model starts
overfitting to whichever arbitrary subset composition happened to dominate
recent steps -- consistent with the monotonic-after-epoch-1 decline holding
across a 10x lr sweep (ruling out plain step-size divergence as the cause).

A larger `M_max` (using ALL of a user's negatives rather than capping) would
remove this specific noise source but was not attempted -- for the highest-
degree users this would mean per-group sizes in the hundreds, materially
changing the compute profile, and the diagnosis here already explains the
observed pattern well enough to not warrant that follow-up before the
already-confirmed iter38 ensemble win.

## Verdict: REJECT

No configuration of this listwise loss beats iter27's BPR-based current
best. iter27 (+ iter38's ensembling on top) remains the selected model.

## Code

`experiments/iter39_listwise_softmax/train.py`, gradient check in
`/private/tmp/claude-501/-Users-nigelyeap/0b355fc9-bbc3-4b4d-88ee-ee430c3f4e64/scratchpad/grad_check_listwise.py`.

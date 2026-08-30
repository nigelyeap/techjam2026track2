# iter5 — FM + listwise (ListNet-style softmax cross-entropy) loss

## What was implemented

`train.py` trains the same `FM` architecture as `baseline.py` (k=16, lr=0.001,
patience=4, same Adam optimizer / same `logits()`) but replaces the pointwise
logloss update with a listwise softmax cross-entropy update, mirroring the
group structure of nDCG@5:

- **Batch construction (per step)**: sample `users_per_batch=256` users
  uniformly (with replacement) from the pool of train users who have >=1
  positive row *somewhere* in their train history (a user with zero
  positives overall can never pass the per-group positivity filter below,
  so excluding them from the sampling pool upfront is equivalent to
  filtering them out every step, just cheaper).
- For each sampled user, take up to `cap` of their train rows (all rows if
  they have <= cap, else a uniform random subset of `cap` without
  replacement).
- Drop any sampled group that ends up with zero positive rows after
  subsampling (possible for an eligible user with many rows and few
  positives when cap is small) — mirrors nDCG@5's convention that a
  zero-positive group has no well-defined target ranking.
- For each surviving group of size `g`: `target[i] = 1/n_pos` if row `i` is
  positive else `0`; predicted distribution = `softmax(scores)` over the
  group (scores = `FM.logits()` raw z). Loss = ListNet-style softmax
  cross-entropy `-sum(target * log(softmax(scores) + eps))`.
- **Gradient**: per-row `d(loss)/d(z_i) = softmax(scores)_i - target_i`
  (exact for softmax-CE with a target that sums to 1 within its group). All
  groups in the step are concatenated into one flat batch (rows +
  `group_id` array), softmax/target/gradient computed vectorized via
  `np.add.at`/`np.maximum.at` keyed by `group_id`, and the accumulated
  gradient is averaged over the number of *groups* (users) in the step —
  not the number of rows — since a user's list is the natural "unit" of a
  listwise loss (mirrors how nDCG@5 itself averages per user). This
  gradient vector is then applied to `V`/`W`/`b` via the exact same
  Adam-update mechanics as `experiments/iter2_bpr_uniform/train.py`'s
  `bpr_step` (copied pattern, not copied file — the loss and its gradient
  are new; the parameter-update math is loss-agnostic).

**Batch-size semantics differ from the row-batch-size used elsewhere**:
`users_per_batch=256` means a step touches ~256×cap rows spread over 256
independent softmax groups, not 8192 i.i.d. rows like the pointwise
baseline or 8192 pairs like BPR. `steps_per_epoch = ceil(n_eligible_users /
users_per_batch)` (~98 steps/epoch here, one pass over the ~24,881 eligible
train users per epoch), per the spec's guidance.

**Hyperparameter note — `cap`**: the spec suggested `cap=20` as an example.
Empirically (seed 0, valid primary): cap=20 -> 0.5936-0.5958 plateau,
cap=50 -> 0.5963, cap=100 -> 0.5969 (diminishing and noisier). Settled on
**cap=50** as the reported config — enough per-user context for the softmax
target to be informative without the extra cost of cap=100 for negligible
gain. `users_per_batch=256` kept as given.

**Confirmed not undertraining**: re-ran seed 0 with `--steps_mult 2` and
`--steps_mult 3` (2x/3x steps per epoch at cap=20 and cap=50) — valid
primary plateaus at the same ~0.594-0.596 level and then overfits faster
(early-stops sooner), it does not surpass the 1x-steps plateau. The ceiling
is a property of the loss/batching, not a training-budget artifact.

## 5-seed results (k=16, lr=0.001, users_per_batch=256, cap=50, patience=4)

| seed | valid primary | test primary |
|------|---------------|---------------|
| 0    | 0.5963        | 0.5910        |
| 1    | 0.5963        | 0.5908        |
| 2    | 0.5967         | 0.5906        |
| 3    | 0.5975        | 0.5907        |
| 4    | 0.5964        | 0.5900        |
| **mean** | **0.5966** | **0.5906** |
| **std**  | 0.0005     | 0.0003     |

(Full per-epoch logs in `run_log.txt`.)

## Verdict vs iter1 (0.6015 valid / 0.5953 test, current best)

**REJECTED.** Listwise softmax CE mean primary is 0.6015 → 0.5966 valid
(−0.0049) and 0.5953 → 0.5906 test (−0.0047), both well outside the ~0.0008
noise floor established by iter1's 5-seed run. It is also worse than
iter2's rejected BPR pairwise loss (0.5988 valid / 0.5923 test mean), so
listwise is not just worse than the pointwise baseline but the worst of the
three loss families tried so far. Aligning the loss family with the group
structure of nDCG@5 (as opposed to BPR's arbitrary pairwise sampling) did
**not** help relative to BPR — if anything it did slightly worse, echoing
iter2's hypothesis: uniform-per-user sampling (whether pairwise or
listwise) changes the effective training distribution away from what
pointwise implicitly weights by (raw impression frequency), and that
distributional shift appears to cost more than the loss-metric alignment
gains back. A likely contributing factor specific to listwise: with only
~256 users per step and `cap` rows each, the softmax normalization is
computed over a fairly small, randomly-capped context per user, giving a
noisier gradient signal per step than pointwise's dense per-row logloss
over the full row-batch.

## Files
- `train.py` — implementation (fresh training loop; imports `FM`/`sigmoid`
  from `baseline.py`, reuses the Adam-update code pattern from
  `experiments/iter2_bpr_uniform/train.py`).
- `run_log.txt` — full stdout across the 5 seeds.

# iter3 — FM + BPR pairwise loss, activity-weighted user sampling

## What changed vs iter2 (`experiments/iter2_bpr_uniform/train.py`)

Only the per-step **user** sampling in `sample_pairs()`. iter2 picked the user for
each training pair uniformly at random over BPR-eligible users
(`rng.integers(0, n_users, size=bs)`). iter3 instead samples users proportional to
their positive-row count in train (`pos_len`), i.e. more active users are sampled
more often — matching the implicit weighting that plain pointwise SGD gets for
free by just iterating over all rows.

Implementation: rather than `rng.choice(n_users, size=bs, p=weights)` (slow for
large weight arrays), built a cumulative-sum table once per run
(`user_cumw = np.cumsum(pos_len)`, `user_totalw = user_cumw[-1]`) and drew each
batch's user picks with `np.searchsorted(user_cumw, rng.random(bs) * user_totalw,
side='right')`, clipped to `n_users - 1`. This is O(bs log n_users) per step and
the table is built once, not per-batch — negligible overhead vs. the uniform
version (both ran at ~0.9–1.2s/epoch, 47 steps/epoch).

Everything else identical to iter2: same `FM` class from `baseline.py` (k=16,
lr=0.001, l2=1e-6), same BPR loss (`-log(sigmoid(score_pos - score_neg))`), same
pos/neg-per-user sampling within a chosen user, same hyperparameters
(bs=8192, patience=4, epochs cap 40), same `build_pos_neg_index` eligibility
filter (users with >=1 positive AND >=1 negative row in train; 24,290 eligible
users, 47 steps/epoch).

Code: `experiments/iter3_bpr_weighted/train.py`. Import paths fixed with
`sys.path.insert(0, ...)` pointing two levels up (`../../` from this subfolder)
to reach `data.py`/`baseline.py`/`evaluate.py`; `--data_dir` default corrected to
`../../KuaiRand-Pure/data` (also passed explicitly on the CLI for every run
below).

## Results, 5 seeds (0-4), same protocol as iter2

| seed | valid primary | test primary | early-stop epoch |
|------|---------------|---------------|-------------------|
| 0    | 0.6025        | 0.5963        | 8                 |
| 1    | 0.6025        | 0.5972        | 11                |
| 2    | 0.6021        | 0.5962        | 11                |
| 3    | 0.6028        | 0.5965        | 13                |
| 4    | 0.6030        | 0.5967        | 11                |

**valid**: mean **0.60258**, std **0.00031**
**test**:  mean **0.59658**, std **0.00035**

(full per-epoch logs in `run_log.txt` in this directory)

## Comparison to current best (iter1) and iter2

| iter | valid primary (mean) | test primary (mean) | test std |
|------|-----------------------|----------------------|----------|
| iter1 (pointwise baseline, official) | 0.6015 (seed0 only) | 0.5946 | 0.0008 |
| iter2 (BPR, uniform-user sampling)   | 0.5988 (5-seed)      | 0.5923 | 0.0003 |
| **iter3 (BPR, activity-weighted)**   | **0.60258** (5-seed) | **0.59658** | 0.00035 |

## Verdict

**PROMOTED.** iter3 beats iter1's official 5-seed test mean (0.5946) by **+0.0020**
(0.59658 vs 0.5946), and beats iter2's test mean (0.5923) by +0.0043. The gain over
iter1 is ~2.5x iter1's own noise floor (std 0.0008) and ~6x iter3's own std
(0.00035) — outside the noise band established by iter1/iter2, not a wash. Valid
primary also improves (0.6026 mean vs iter1's single-seed 0.6015).

This confirms the working hypothesis: BPR's ranking-loss alignment with the
GAUC/nDCG@5 metric was directionally right in iter2, but uniform-per-user
sampling threw away the implicit activity-weighting that pointwise SGD gets for
free, which cost more than the loss-alignment gained. Restoring that weighting
(sample users proportional to their positive-row count) recovers the loss
alignment's benefit and turns BPR from a regression (iter2: -0.0023 vs iter1)
into a genuine, noise-floor-clearing improvement (iter3: +0.0020 vs iter1).

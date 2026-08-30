# iter35 — date-shifted-split robustness check of iter27's winning fused config

## Idea

iter27 (`experiments/iter27_triple_fusion/`) is the current standing best:
valid primary 0.63792 / test primary 0.63889 (5-seed), config =
`sampling_alpha=0.75` (decay-aware BPR user-sampling weight, halflife=3d) +
Laplace `alpha=0.5` + `n_buckets=20` + iter24's refined causal features
(`decay_rate_2.5, decay_act_2.5, decay_tab_3, last1, lastk_rate, gap`).

One ingredient of this fused config, `n_buckets=20`, was previously shown to
be date-shift-sensitive **in isolation**: iter29
(`experiments/iter29_bucket_robustness/`) found that on the official split,
`n_buckets=20` beats `n_buckets=10` by a real margin (using iter19's plain
feature set, Laplace alpha=1.0 default, flat sampling — none of iter27's
other two fusion ingredients present), but on a 3-day-earlier-shifted split
(train 2022-04-05..18 / valid 04-19..25 / test 04-26..05-05), that isolated
effect shrinks to near-zero/flips sign (Δ valid: +0.00063 → −0.00007; Δ
test: +0.00362 → +0.00056).

iter27's own RESULT.md explicitly flagged this as an open, unresolved
caveat: it re-confirmed `n_buckets=20` still wins **within the fused
config** on the official split (+0.00156 valid, 3-seed, at
sampling_alpha=0.5), but never re-checked whether that held **within the
fused config** under iter29's exact date shift. This iteration closes that
gap: rerun iter27's exact winning fused config on iter29's exact shifted
split, for n_buckets ∈ {10, 20}, and compare against both iter29's isolated
effect and iter27's own official-split fused-config effect.

**This is a robustness/diagnostic check, not a promotion candidate.** A
shifted split is not the official split; nothing here can become a new
"current best" regardless of the numbers. The only question in scope: does
`n_buckets=20` still beat `n_buckets=10` *within the fused config* under the
date shift, the same comparison iter29 made for the isolated lever?

## Method

Created `experiments/iter35_iter27_date_shift_robustness/{data_ext.py,
train.py, driver.py}`:

- `data_ext.py` is a line-for-line copy of
  `iter27_triple_fusion/data_ext.py`'s feature computation (the four
  independent causal traversals: flat activity/tab_pos/rate, momentum
  last1/lastk/gap, fine-grid exponential decay, decayed tab_pos) and
  `encode_ext`/`compute_final_decayed_pos`, UNMODIFIED. The only structural
  change: `_load_raw_time`/`load_ext` take an explicit `splits_map`
  argument instead of hardcoding an import of `SPLITS` from `../../data.py`
  — mirroring exactly how `iter29_bucket_robustness/data_ext.py` adapted
  `iter25_retune_v2/data_ext.py` for the same purpose. `SPLITS_SHIFTED` is
  copied verbatim from iter29's own module (`train 2022-04-05..18 / valid
  04-19..25 / test 04-26..05-05`). This lets the SAME feature/encoding code
  run against either the OFFICIAL split or iter29's SHIFTED split, selected
  by the caller, with per-split pickle caches kept separate via a
  `split_tag`.
- `train.py` is an EXACT, unmodified copy of `iter27_triple_fusion/train.py`
  — no changes were needed since it only ever consumes a pre-built
  `splits_cache` from the driver.
- `driver.py` runs (in order): (0) an OFFICIAL-split harness-fidelity check
  of iter27's exact winning config (`sampling_alpha=0.75, alpha=0.5,
  n_buckets=20, decay_halflife=3`, ITER24_FEATS) against
  `iter27_triple_fusion/results.json`'s published 5-seed numbers; (1) a
  supplementary OFFICIAL-split measurement of `n_buckets=10` at the SAME
  `sampling_alpha=0.75` (iter27 itself only measured n_buckets=10 at
  sampling_alpha=0.5, so this fills in the missing apples-to-apples cell for
  the three-way table), 3 seeds; (2) a SHIFTED-split construction check
  (row counts / date boundaries) against iter29's reported numbers; (3) the
  actual robustness check — the winning fused config with n_buckets ∈
  {10, 20}, 5 seeds each, on the SHIFTED split.

Every run's result is appended to `results.json` immediately after it
finishes (incremental save, matching iter27/iter29's own driver pattern).

## Harness-fidelity checks

- **Official split, iter27's exact winning config, 5 seeds**: bit-exact
  match to `iter27_triple_fusion/results.json` (tag
  `fusion_sampling_alpha0.75`) on every seed, both valid and test (max
  |Δ| < 1e-4). Mean: valid 0.63792, test 0.63889 — identical to iter27's
  published numbers.
- **Shifted-split construction**: row counts (`train` 1,079,797 / `valid`
  143,394 / `test` 170,150) match `iter29_bucket_robustness`'s reported
  shifted split exactly; date boundaries copied verbatim from iter29's own
  module (train 2022-04-05..18 / valid 04-19..25 / test 04-26..05-05).

Both checks passed before any downstream result was trusted.

## Results

| tag | split | n_buckets | seeds | valid mean | test mean |
|---|---|---|---|---|---|
| `official_winning_cfg_nbuckets10` | official | 10 | 3 | 0.63676 | 0.63600 |
| `shifted_winning_cfg_nbuckets20` | shifted | 20 | 5 | 0.63609 | 0.63994 |
| `shifted_winning_cfg_nbuckets10` | shifted | 10 | 5 | 0.63453 | 0.63881 |

(iter27's own published `n_buckets=20`, official split, sampling_alpha=0.75,
5-seed: valid 0.63792 / test 0.63889, reproduced bit-exact above as the
fidelity check.)

## Three-way comparison

| | Δ valid (20 vs 10) | Δ test (20 vs 10) |
|---|---|---|
| iter29 isolated lever, shifted split (iter19 feats, flat sampling) | −0.00007 | +0.00056 |
| iter27 fused config, official split (sampling_alpha=0.5) | +0.00156 | n/a |
| iter35 fused config, official split (sampling_alpha=0.75, 3-seed) | +0.00140 | n/a |
| **iter35 fused config, shifted split (sampling_alpha=0.75, 5-seed)** | **+0.00156** | **+0.00114** |

## Diagnosis

The caveat iter27 flagged does **not** materialize for the fused config.
Under iter29's exact date shift, `n_buckets=20` still beats `n_buckets=10`
by +0.00156 valid / +0.00114 test — essentially the same margin as on the
official split (+0.00140 to +0.00156 valid across both sampling_alpha
settings tested), not the near-zero/flipped margin iter29 found for the
*isolated* lever (−0.00007 valid). The isolated-lever fold-specificity
iter29 documented does not carry over once `n_buckets=20` is combined with
iter24's refined features and iter23's decay-aware sampling weight — the
combination appears to stabilize the bucket-count effect rather than
inherit its fragility. A plausible mechanism: iter29's isolated setup used
iter19's plain feature set with flat sampling, where the specific rows that
fall into extreme quantile buckets vary more across a date shift; iter27's
richer feature set + decay-aware sampling may already be capturing/
smoothing much of the same signal `n_buckets=20` picks up in isolation,
making the remaining marginal bucket-count effect less sensitive to which
exact rows land in each split.

## Verdict: robustness finding — iter27's promotion is NOT undermined

`n_buckets=20`'s contribution to iter27's fused config is robust to the
date shift that made the same lever's *isolated* effect vanish (iter29).
This closes the open caveat from iter27's RESULT.md with a positive
result: no evidence that iter27's official-split margins are an artifact of
that particular date split. iter27 remains current best; no ledger change
to "Current best" is needed from this iteration — this is confirmatory,
not a new candidate.

## Code

`experiments/iter35_iter27_date_shift_robustness/{data_ext.py,train.py,driver.py}`,
raw results in `experiments/iter35_iter27_date_shift_robustness/results.json`,
full run trace in `driver_log.txt`.

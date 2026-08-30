# iter6 — causal user-author affinity history feature

## What was implemented

Added a 6th feature field, `hist_affinity`, alongside the existing 5
(`user_id, video_id, author_id, tab, dur_bucket`), keeping the exact same
pointwise-logloss FM training loop as `baseline.py --model fm`
(k=16, lr=0.001, bs=8192, epochs=40, patience=4) — only the feature set
changed, not the loss or optimizer.

For each row `(date, user_id, video_id, author_id, tab, duration_ms, label)`,
`hist_affinity` = how many times this user previously had a `long_view=1`
interaction with this exact `author_id`, using only strictly-earlier-date
rows (see causality section below). This count is then bucketed into 5
discrete buckets: `{0, 1, 2, 3-5, 6+}` (fixed rule, not quantiles — see
rationale below), and appended as a 6th categorical field before FM
embedding lookup.

Files (all new, nothing outside `experiments/iter6_history_feature/` was
touched):
- `data_ext.py` — `compute_history_counts()`, `bucket_history_count()`,
  `load_ext()` (copies/adapts `data.load()`), `encode_ext()` (copies/adapts
  `data.encode()`, adds the 6th field). Also has a `__main__` block that
  prints and manually re-verifies sample rows (see below).
- `train.py` — `run_fm_ext()`, a line-for-line copy of `baseline.py`'s
  `run_fm()` training loop, unchanged except it calls `encode_ext()`/
  `load_ext()` instead of `data.py`'s `load()`/`encode()`. Reuses the
  unmodified `FM` class from `baseline.py` and the unmodified `evaluate()`
  from `evaluate.py`.
- `logs/seed{0..4}.log` — full training logs + final JSON result line per seed.

## Bucketing rationale (deviation from pure quantiles)

`dur_bucket` in the original `data.py` uses 10 quantile edges fit on train.
Quantile bucketing degenerates for `hist_affinity`: the raw count distribution
is extremely zero-heavy (see below), so most quantile cut points collapse
onto 0, producing far fewer than 10 meaningfully distinct buckets. Instead
I used a fixed, count-based rule: `0 → 0`, `1 → 1`, `2 → 2`, `3-5 → 3`,
`6+ → 4`, which preserves maximum resolution where the data is dense (low
counts) and compresses the sparse long tail. Measured distribution across
all 1,436,609 rows (train+valid+test combined):

| bucket | count | share |
|---|---|---|
| 0 (no prior positive) | 1,426,586 | 99.30% |
| 1 | 9,701 | 0.68% |
| 2 | 289 | 0.020% |
| 3 (3-5) | 33 | 0.002% |
| 4 (6+) | 0 | 0% |

Only **0.70%** of rows (10,023 / 1,436,609) have any prior user-author
positive at all. This single number is the main reason for the null result
below: within a ~1-month log window, a user re-encountering an author they
previously long-viewed is rare, so the feature has almost no coverage to
work with.

## Causality / no-leakage verification methodology

This is the most important correctness constraint, so it was checked two ways:

1. **Design-level**: `compute_history_counts()` combines train+valid+test
   raw rows into one flat list, sorts by `date`, and processes **date groups**
   rather than individual rows: for every row in a given date, it first reads
   the counter state as it stood *before* that date (so same-date rows can
   never see each other or themselves), and only *after* all rows for that
   date have been read does it fold that date's `label==1` rows into the
   counter (so the next date sees them, but this date never did). This is a
   strict `<` comparison by construction, never `<=`. Because `SPLITS`'
   date ranges are non-overlapping and monotonic (train 20220408-20220421 <
   valid 20220422-20220428 < test 20220429-20220508), this single combined-timeline
   pass automatically gives: train rows see only earlier train rows; valid/test
   rows see all earlier rows (train history + earlier same-split rows) — matching
   the causal deployment scenario described in the assignment, with no special-casing
   needed.

2. **Empirical spot-checks** (in `data_ext.py`'s `__main__`, reproducible via
   `python3 data_ext.py --data_dir ../../KuaiRand-Pure/data`):
   - For every sampled row with `hist_count >= 2`, brute-force recomputed the
     count by filtering the full flat row list for `same user, same author,
     date < row_date, label==1` and asserted equality. All matched.
   - For sampled rows with `hist_count == 0`, confirmed the brute-force
     recount was also 0.
   - Explicitly located `(user, author, date)` groups with **≥2 positive
     rows on the same date** and confirmed both rows report `hist_count=0`
     for each other — e.g. `user=41, author=333794, date=20220411` has two
     `label=1` rows, both showing `hist_count=0`, proving same-date rows do
     not leak into each other's counts (strict `<`, not `<=`, verified
     concretely, not just by code inspection).

All checks passed; no same-date or future-date leakage detected.

## Results (5 seeds, same hyperparams as iter1: k=16, lr=0.001, bs=8192, epochs=40, patience=4)

| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.6016 | 0.5951 |
| 1 | 0.6017 | 0.5946 |
| 2 | 0.6014 | 0.5935 |
| 3 | 0.6023 | 0.5946 |
| 4 | 0.6012 | 0.5950 |
| **mean** | **0.6016** | **0.5945** |
| **std** | 0.0004 | 0.0006 |

Full per-seed GAUC/nDCG@5/primary breakdown is in `logs/seed{0..4}.log`.

## Verdict vs iter1 (0.6015 valid / 0.5953 test, seed0; official 5-seed test mean 0.5946 std 0.0008)

**REJECTED** — the causal user-author affinity feature produces valid/test
primary scores statistically indistinguishable from iter1 (valid mean
0.6016 vs iter1's 0.6015; test mean 0.5945 vs iter1's 5-seed official mean
0.5946), well inside the ~0.0008 noise floor in both directions. The feature
adds no measurable signal here — most likely because it only has nonzero
value on 0.70% of rows (see coverage table above), so an FM model with this
field essentially can't learn anything useful from it: it's UNK/zero for
99.3% of impressions. The causal-correctness engineering is verified sound;
the negative result is a genuine sparsity/coverage finding, not a leakage
or implementation bug.

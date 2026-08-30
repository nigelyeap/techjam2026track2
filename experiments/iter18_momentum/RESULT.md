# iter18 — fine-grained timestamp-level causal momentum features

## Summary

iter9's causal traversal only resolves ordering at DAY granularity (strict
`<` on `date`): two rows on the same date are treated as simultaneous even
if they happened hours apart. This iteration adds a finer-grained,
`time_ms`-level causal traversal per user and derives three new SHORT-TERM
"session momentum" features on top of iter9's coarse `activity`/`tab_pos`/
`rate`:

- **`last1`** — was the user's immediately-preceding row (strict `time_ms`
  order) a `long_view`? Categorical `0`/`1`/`UNK` (UNK = user's first row in
  the combined train+valid+test timeline).
- **`lastk_rate`** — Laplace-smoothed positive rate over the user's last
  K=5 rows strictly before this one: `(sum(last 5 labels)+1)/(min(5,avail)+2)`.
- **`gap`** — bucketed time gap (ms) since the user's immediately-preceding
  row — pure recency-of-engagement, independent of what that prior row's
  label was. `UNK` for the user's first row.

**Result: stacking all three momentum features (`last1+lastk_rate+gap`) on
top of iter9's `{activity,tab,rate}` gives a large, consistent win — 5-seed
valid primary mean 0.61417 (std 0.00114) vs iter9's 0.61013 (std 0.00027),
+0.00404 (~15x iter9's std); test primary mean 0.60927 (std 0.00125) vs
iter9's 0.60560, +0.00367 (~18x iter9's std). All 5/5 seeds improve on BOTH
splits. This is the largest single gain found since iter9 itself.**

**Verdict: PROMOTE.**

## 1. Causal correctness — verification

### Method

`compute_momentum_features()` in `data_ext.py`:
1. Combines train+valid+test raw rows into one flat timeline (same pattern
   iter9 used for its date-grouped features).
2. For each user, sorts ALL of that user's rows across every split by
   `(time_ms, orig_idx)` ascending — `time_ms` is a genuine millisecond
   epoch timestamp read directly from the raw log (`log_standard_*.csv`,
   which `data.py`'s `load()` currently discards); `orig_idx` is the row's
   position in the raw CSV read order (file1 then file2, BEFORE any
   date-range split filtering), used purely as a **stable, split-independent
   tiebreak** for the (empirically near-nonexistent) case of an exact
   `time_ms` tie.
3. Walks that per-user sequence once. At each row, features are **READ
   first** (from state built by strictly-earlier rows in this total order),
   and the row's own label is folded into the state only **after** the read.
   This makes leakage structurally impossible — a row can never see its own
   label or a later row's label, and the tiebreak makes the order strict and
   total so same-`time_ms` rows can't see each other either.

`_load_raw_time()` in `data_ext.py` mirrors `data.py`'s `load()` line for
line (same files, same read order, same `vid2author` join, same
`SPLITS`-based date filtering imported directly from `data.py` to stay in
sync) but additionally keeps `hourmin`, `time_ms`, and assigns `orig_idx`.
Because both loaders derive from the *identical* sequential file-read order
and filter with the *identical* predicate, the resulting per-split row lists
line up index-for-index with `data.py`'s own splits by construction — no
fuzzy joining by key was needed (and would have been ambiguous given
possible duplicate rows).

### Spot-check results (full output in `spotcheck_output.txt`)

Brute-force recount against 3 real users' **complete chronological
sequences** (13, 13, and 10 rows respectively), printing every row's
`date`/`time_ms`/`label` alongside both the computed and manually-recounted
`last1`/`lastk_sum`/`lastk_cnt`/`gap_ms` — **all 36 rows across the 3 users
matched exactly, zero mismatches.** Representative excerpt (user 1):

```
pos= 3 date=20220412 time_ms=1649706052290 label=1 | last1: got=0 manual=0 | lastk_sum: got=1 manual=1 ...
pos= 4 date=20220412 time_ms=1649706789917 label=1 | last1: got=1 manual=1 | lastk_sum: got=2 manual=2 ...
```
Note pos=3 and pos=4 share the same `date` (20220412) but different
`time_ms` 12 minutes apart — under iter9's date-level traversal these two
rows would be simultaneous (pos=4 could not see pos=3's label at all); under
iter18's `time_ms` traversal, pos=4 correctly sees pos=3's label=1 as its
`last1`. This is exactly the extra resolution this iteration adds.

A **synthetic same-`time_ms` tie stress test** (two rows sharing an exact
`time_ms=5000`, plus a third row at a later `time_ms=5500` but a *smaller*
`orig_idx` than the tied pair) confirmed: (a) the tiebreak correctly orders
the two tied rows deterministically and neither leaks into the other in the
wrong direction, and (b) real chronological order always wins over the
`orig_idx` tiebreak — the later-`time_ms` row is correctly placed after both
tied rows despite having a smaller `orig_idx`. All assertions passed.

Real-world exact-`time_ms` ties turned out to be vanishingly rare (not
observed in the 3 sampled users at all), consistent with `time_ms` being a
genuine millisecond epoch column, but the tiebreak logic was still verified
correct via the synthetic case per the task's requirement.

## 2. Coverage

Over all 1,436,609 rows (train+valid+test combined):

| Feature | Coverage (not user's first row) | Notes |
|---|---|---|
| `last1` | **98.12%** (1,409,532 / 1,436,609) | denser than iter9's `rate` (92.29%) |
| `gap` | **98.12%** (1,409,532 / 1,436,609) | same denominator as `last1` (both undefined only for a user's very first row) |
| `lastk_rate` | 100% (always defined via Laplace smoothing, falls back to 0.5 neutral prior when 0 prior rows exist) | — |

Of covered `last1` rows, 33.15% have `last1==1` (prior row was a
long_view) — close to the overall label base rate, as expected.
`lastk_cnt` distribution: `{0: 27077, 1: 26742, 2: 26412, 3: 26017,
4: 25634, 5(full window): 1,304,727}` — the vast majority of rows have a
full 5-row window available.

Gap magnitude: median 2385s (~40 min) between consecutive user actions,
p25=317s, p75=28302s (~7.9h) — a wide spread, consistent with the mix of
within-session and cross-session gaps this feature is meant to capture.

## 3. Sweep (3 seeds: 0,1,2), valid primary mean

All configs stack on top of iter9's `{activity,tab,rate}` base (re-derived
here for parity — confirmed to match iter9 within noise, see below).

| Config | valid mean | valid std | test mean | test std |
|---|---|---|---|---|
| `base` (=iter9 re-derived) | 0.61024 | 0.00013 | 0.60572 | 0.00010 |
| `+last1` | 0.60595 | 0.00031 | 0.60178 | 0.00035 |
| `+lastk_rate` | 0.60909 | 0.00018 | 0.60418 | 0.00015 |
| `+last1+lastk_rate` | 0.60580 | 0.00066 | 0.60197 | 0.00096 |
| `+gap` | 0.61134 | 0.00049 | 0.60633 | 0.00023 |
| `+last1+gap` | 0.61242 | 0.00023 | 0.60737 | 0.00066 |
| `+lastk_rate+gap` | 0.61150 | 0.00121 | 0.60625 | 0.00033 |
| **`+last1+lastk_rate+gap`** | **0.61472** | 0.00108 | **0.61002** | 0.00108 |

**Parity check**: `base` (0.61024 valid / 0.60572 test, 3-seed) matches
iter9's own 5-seed mean (0.61013 valid / 0.60560 test) within noise —
confirms the re-derivation is faithful and any downstream deltas are
attributable to the new features, not to a re-implementation drift.

**A striking, non-additive pattern**: `last1` and `lastk_rate` each *hurt*
performance in isolation (both individually and combined: -0.004 to -0.006
valid vs base) — they appear to be too noisy/collinear with iter9's existing
`rate` on their own, or the FM's linear+pairwise structure fits them
worse without the recency-conditioning that `gap` provides. `gap` alone is
mildly positive (+0.0011 valid). But once `gap` is present, `last1` and
`lastk_rate` both become net-positive contributors, and the full 3-feature
combination is dramatically better than any subset — +0.00448 valid vs base,
more than double the best 2-feature combo (`last1+gap` at +0.00218). This
suggests `gap` (recency) acts as an enabling context: momentum features
("was engagement high/positive recently") are more useful to the model when
it also knows *how* recently, letting it calibrate how stale/fresh that
momentum signal is — a form of feature interaction the FM's pairwise terms
can exploit once all three are present together.

## 4. 5-seed confirmation of the winning combo

`+last1+lastk_rate+gap` (fields = `user_id, video_id, author_id, tab,
dur_bucket, activity, tab, rate, last1, lastk_rate, gap`), seeds 0-4:

| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.61349 | 0.60871 |
| 1 | 0.61457 | 0.60998 |
| 2 | 0.61612 | 0.61136 |
| 3 | 0.61390 | 0.60797 |
| 4 | 0.61277 | 0.60834 |
| **mean** | **0.61417** | **0.60927** |
| **std** | 0.00114 | 0.00125 |

vs iter9 (5-seed): valid mean 0.61013 (std 0.00027), test mean 0.60560
(std 0.00020).

- **valid delta: +0.00404** (~15x iter9's std, ~3.5x this config's own std)
- **test delta: +0.00367** (~18x iter9's std, ~3x this config's own std)
- Per-seed comparison vs iter9's matching seed: seed0 +0.00319/+0.00311,
  seed1 +0.00417/+0.00428, seed2 +0.00602/+0.00546, seed3 +0.00430/+0.00267,
  seed4 +0.00247/+0.00274 (valid/test) — **improvement is consistent 5/5
  seeds on both splits**, not driven by one lucky run.

This clears the task's ~0.001-0.002 "real signal" bar by 2-4x, and — unlike
iter11's rejected valid-only edge that reversed on test — the gain shows up
consistently on **both** valid and test, which is the strongest evidence
this run has produced against overfitting the sweep.

## 5. Verdict: PROMOTE

`experiments/iter18_momentum` (FM + activity-weighted BPR, fed iter9's
`{activity,tab,rate}` plus the new time_ms-level `{last1,lastk_rate,gap}`
momentum features) becomes the new candidate best:

- valid primary mean **0.61417** (5-seed, std 0.00114) — vs iter9's 0.61013
- test primary mean **0.60927** (5-seed, std 0.00125) — vs iter9's 0.60560

Hypothesis confirmed: day-granularity causal features (iter9) miss
within-day/short-term session dynamics that hour/minute-resolution
`time_ms` ordering can recover. Unlike iter9's `rate` (a long-run aggregate
over the user's entire history), `last1`/`lastk_rate` capture the user's
*current* session mood, and `gap` supplies the recency context that lets the
model calibrate how much to trust that momentum signal — together they
carry information genuinely orthogonal to iter9's day-level features.

Code: `experiments/iter18_momentum/{data_ext.py,train.py,driver.py,
driver_extra_seeds.py}`. Raw per-run results: `results.json`. Causal
spot-check transcript: `spotcheck_output.txt`.

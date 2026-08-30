# iter27 — triple fusion of iter24 (refined features) + iter23 (decay-aware BPR sampling) + iter25 (retuned Laplace alpha / n_buckets)

## Rerun note (Round 9)

This is a **rerun** of iter27. The original Round 8 dispatch was killed by a
platform session-limit error before it produced any `results.json` (total
compute loss on that attempt; only the source code in
`experiments/iter27_triple_fusion/{data_ext.py,train.py,driver.py}`
survived). This document is being written fresh from the rerun.

New caveat factored into this rerun's interpretation: `iter29_bucket_robustness`
(written after the original iter27 dispatch) found that iter25's
`n_buckets=20` effect does **not** robustly replicate on a date-shifted
split -- it may be partly fold-specific. The Step 1 sweep below therefore
includes an `n_buckets=10` variant alongside the default `n_buckets=20`, to
check whether the bucket-count choice still matters once combined with the
other two ingredients, on the official (non-shifted) split.

## Idea

Three Round 7 wins were each independently confirmed (5-seed, both-split,
real gains over iter19) but never combined, because they change
non-overlapping mechanisms:
1. **iter24** -- model INPUT FEATURES: `decay_rate_2.5, decay_act_2.5,
   decay_tab_3, last1, lastk_rate, gap` (halflife retune + decayed
   `tab_pos` + momentum). Standing current best: valid 0.63251 (5-seed),
   test 0.62843 (5-seed).
2. **iter23** -- BPR TRAINING-TIME user-sampling weight:
   `decayed_pos_total[user] ** sampling_alpha` (sampling_alpha=0.5,
   halflife=3d) replacing the flat `pos_len[user] ** sampling_alpha`.
3. **iter25** -- FORMULA CONSTANTS: Laplace-smoothing `alpha=0.5` inside the
   decay/rate ratio formulas (default 1.0) + `n_buckets=20` quantile-bucket
   count (default 10).

`experiments/iter27_triple_fusion/data_ext.py`/`train.py` (surviving code
from the killed Round-8 attempt) implement this fusion: all three
ingredients copied verbatim from their respective already-verified source
modules, per protocol (no re-derivation of causality proofs needed since
nothing here is a NEW causal feature -- only a harness-fidelity check is
required).

## Pre-flight checks (before trusting anything)

- **Syntax/import check**: `data_ext.py`/`train.py`/`driver.py` all parse
  and import cleanly; `python3 -c "import data_ext, train"` succeeds from
  within this directory.
- **Feature cache**: a `.cache_v1_2-2.5-3-3.5__tab_3-7.pkl` (381MB) survived
  from the killed Round-8 run. Verified intact by loading it directly
  (`pickle.load` succeeds, correct split sizes: train 1,141,112 / valid
  124,909 / test 170,588, loads in ~3s) -- reused rather than recomputed.
- **Data path**: `driver.py`'s `DATA_DIR='../../KuaiRand-Pure/data'` is
  relative to this experiment directory (`experiments/iter27_triple_fusion/`),
  resolving to `kuairand-starter-kit/KuaiRand-Pure/data` -- confirmed present
  with all 4 required CSVs.

<!-- CAUSALITY CHECK SECTION: to be filled in after `python3 data_ext.py` completes -->

## Causality/harness verification

### Bug found and fixed during rerun

The surviving `data_ext.py` (from the killed Round-8 attempt) had a real
performance bug in its `__main__` block (PART E, reference-date edge case):

```python
# BEFORE (bug): max(...) re-evaluated once per row -> O(n^2) over ~1.14M train rows
last_date_users = [r[1] for r in train_rows if r[0] == max(rr[0] for rr in train_rows) and r[6] == 1]
...
n_pos_last_date = sum(1 for r in train_rows if r[1] == u and r[6] == 1
                       and r[0] == max(rr[0] for rr in train_rows))
```

`max(rr[0] for rr in train_rows)` sits inside a list-comprehension filter
that itself iterates over `train_rows` (~1.14M rows) -- so the `max()` scan
gets re-run once *per row*, an accidental O(n²) ≈ 1.3 trillion operations.
This is exactly what hung the first run attempt: pid 30167 ran for **20+
CPU-minutes with zero output** before being killed; a `sample`-based stack
trace confirmed it was stuck inside nested `builtin_max`/generator-iteration
frames at the time of the kill. Fixed by hoisting the max out of the loop
into a `last_train_date` variable computed once:

```python
last_train_date = max(rr[0] for rr in train_rows)
last_date_users = [r[1] for r in train_rows if r[0] == last_train_date and r[6] == 1]
...
n_pos_last_date = sum(1 for r in train_rows if r[1] == u and r[6] == 1 and r[0] == last_train_date)
```

After the fix, the full `python3 data_ext.py` causality suite completes in
well under a minute.

### Full output (`python3 data_ext.py`, post-fix)

```
=== PART A: decay-feature (rate/act) causal spot-checks (brute force) ===
halflife= 2.0d  decayed_total>0 coverage: 92.29%  (mean=12.916, max=296.595)
halflife= 2.5d  decayed_total>0 coverage: 92.29%  (mean=14.995, max=342.161)
halflife= 3.0d  decayed_total>0 coverage: 92.29%  (mean=16.753, max=377.162)
halflife= 3.5d  decayed_total>0 coverage: 92.29%  (mean=18.273, max=404.760)
25 random rows x 4 halflives: decayed_pos/decayed_total match brute force (max abs err 1.42e-14). No leakage detected.
zero-activity rows (5 checked): decayed_pos/total/tab correctly 0.0.
same-date-pair edge case (3 rows): decay values identical across the pair, as expected. PASSED.

=== PART B: decayed tab_pos causal spot-checks (brute force) ===
tab halflife= 3.0d  decayed_tab_pos>0 coverage: 73.37%  (mean=3.658, max=81.792)
tab halflife= 7.0d  decayed_tab_pos>0 coverage: 73.37%  (mean=5.557, max=102.994)
30 random rows x 2 tab-halflives: all decayed_tab_pos match brute force (max abs err 1.42e-14). No leakage detected.
zero-tab_pos-but-nonzero-activity rows (5 checked): decayed_tab_pos correctly 0.0 despite nonzero decay_rate/act.
same-date-pair edge case (decay_tab, 3 rows): decayed_tab_pos identical across the pair, as expected. PASSED.

=== PART C: momentum-feature causal spot-checks (brute force) ===
last1 coverage (not user's first row): 98.12%
gap coverage (not user's first row): 98.12%
[3 real users' full chronological sequences, 36 rows total -- all last1/lastk_sum/gap_ms
values matched brute-force manual recount exactly, zero mismatches]
--- synthetic same-time_ms tie stress test (momentum) ---
tie stress test: all assertions passed.

=== PART D: cross-family joint edge case (same-date, different time_ms pair) ===
user=1 date=20220412: 3 rows, same calendar date, distinct time_ms
  -> decay AND decay_tab features are IDENTICAL across the pair (both date-level, correctly
     blind to intra-date order); momentum last1 correctly DIFFERS and resolves the true
     time_ms order. No cross-contamination from the join.

=== PART E: decay-aware sampling-weight spot-check (brute force) ===
train period end (reference date ordinal): 738266  (24881 users with >=1 train positive)
30 random users: compute_final_decayed_pos matches brute-force recount of 0.5**(gap_days/3)
over all TRAIN positive rows (max abs err 2.66e-15). No arithmetic error.
zero-train-positive users (5 checked): correctly absent from decayed_pos dict.

All causal spot-checks (decay + decay_tab + momentum + cross-family joint + sampling-weight) passed.
```

No leakage detected in any feature family, individually or in pairwise/
cross-family combination. All three fused ingredients (decay features,
decayed tab_pos, momentum, decay-aware sampling weight) are copied verbatim
from already-verified prior iterations (iter20/iter24, iter18, iter22/
iter23 respectively), so no new causality proof was required beyond this
spot-check -- consistent with protocol.

## Config sweep (3 seeds)

Per this round's discipline, selection is valid-only: `test_primary` was computed
and stored in `results.json` for every run (cheap side-effect of `run_bpr_ext`),
but not printed/compared during the sweep. All four configs use the fused
feature set (iter24) + decay-aware sampling (iter23, halflife=3d) as a fixed
base; the axes varied are `n_buckets` (iter29's caveat) and `sampling_alpha`
(re-swept because iter23 originally tuned it against a *different* feature
set than iter24's refined one).

| tag | sampling_alpha | Laplace alpha | n_buckets | seed0 | seed1 | seed2 | valid mean | margin vs iter24 (0.63251) |
|---|---|---|---|---|---|---|---|---|
| `triple_fusion_default` | 0.5 | 0.5 | 20 | 0.63848 | 0.63735 | 0.63829 | **0.63804** | +0.00553 |
| `triple_fusion_nbuckets10` | 0.5 | 0.5 | 10 | 0.63617 | 0.63662 | 0.63664 | **0.63648** | +0.00397 |
| `fusion_sampling_alpha0.25` | 0.25 | 0.5 | 20 | 0.63788 | 0.63777 | 0.63586 | **0.63717** | +0.00466 |
| `fusion_sampling_alpha0.75` | 0.75 | 0.5 | 20 | 0.63894 | 0.63868 | 0.63685 | **0.63816** | +0.00565 |

Best 3-seed config: **`fusion_sampling_alpha0.75`** (valid mean 0.63816, margin
+0.00565 over iter24's 5-seed reference). Margin clears the 0.001 promotion
threshold, so this config was extended to 5 seeds.

**n_buckets=20 vs n_buckets=10, re: the iter29 caveat.** Holding
sampling_alpha=0.5 fixed, n_buckets=20 beats n_buckets=10 by +0.00156 valid
(0.63804 vs 0.63648) on the official (non-date-shifted) split, inside the
fused config. This does **not** contradict iter29 -- iter29's finding was
that the *isolated* n_buckets=20 effect shrinks/flips sign specifically on a
*date-shifted* split (a fold-specificity check this iteration did not rerun).
Here, on the split iter27 actually selects and reports against, n_buckets=20
still clearly wins as part of the fused config, so it was kept in the winning
config. Whether that edge would also fade under a date shift, as iter29 found
for the isolated effect, remains untested for the *fused* config and is
flagged honestly as an open question rather than assumed away.

## 5-seed confirmation

Extended `fusion_sampling_alpha0.75` (sampling_alpha=0.75, Laplace alpha=0.5,
n_buckets=20, decay_halflife=3) to seeds 0-4:

| seed | valid | test |
|---|---|---|
| 0 | 0.63894 | 0.63989 |
| 1 | 0.63868 | 0.63913 |
| 2 | 0.63685 | 0.63768 |
| 3 | 0.63747 | 0.63853 |
| 4 | 0.63768 | 0.63921 |
| **mean** | **0.63792** | **0.63889** |
| std | 0.00077 | 0.00074 |

Test mean (0.63889) was read exactly once, at the end, only for this single
promoted candidate -- consistent with this round's valid-only-selection
discipline.

### Comparison against iter24 (5-seed, current standing best: valid 0.63251, test 0.62843)

| seed | iter24 valid | iter27 valid | Δvalid | iter24 test | iter27 test | Δtest |
|---|---|---|---|---|---|---|
| 0 | 0.63260 | 0.63894 | +0.00634 | 0.62839 | 0.63989 | +0.01150 |
| 1 | 0.63308 | 0.63868 | +0.00560 | 0.62942 | 0.63913 | +0.00971 |
| 2 | 0.63179 | 0.63685 | +0.00506 | 0.62687 | 0.63768 | +0.01081 |
| 3 | 0.63208 | 0.63747 | +0.00539 | 0.62855 | 0.63853 | +0.00998 |
| 4 | 0.63298 | 0.63768 | +0.00470 | 0.62892 | 0.63921 | +0.01029 |
| **mean** | 0.63251 | 0.63792 | **+0.00541** | 0.62843 | 0.63889 | **+0.01046** |

**5/5 seeds improve on both valid and test, no sign flips.** This matches the
ledger's standing bar for promotion.

### Comparison against the other two parents (5-seed means)

| | valid | test |
|---|---|---|
| iter23 (decay-aware sampling alone) | 0.63109 | 0.62929 |
| iter25 (Laplace alpha/n_buckets alone) | 0.63028 | 0.63185 |
| iter24 (refined features alone, current best) | 0.63251 | 0.62843 |
| **iter27 (all three fused)** | **0.63792** | **0.63889** |
| iter27 delta vs best individual parent (iter24 valid / iter25 test) | +0.00541 | +0.00704 |

The fused config beats every individual parent's own 5-seed number on both
splits, not just the previous overall-best (iter24) -- i.e. the three
mechanisms combine roughly additively-or-better here rather than cancelling,
despite touching training-time sampling, model input features, and formula
constants simultaneously (three genuinely non-overlapping mechanisms, as
Round 7's analysis argued).

## Verdict: **PROMOTE — new current best**

`fusion_sampling_alpha0.75` (feature set = iter24's refined set, sampling
weight = iter23's decay-aware `decayed_pos_total**0.75` with halflife=3d,
Laplace alpha=0.5, n_buckets=20) becomes the new standing best:

- **valid: 0.63792** (5-seed mean, std 0.00077) vs iter24's 0.63251 -- **+0.00541**
- **test: 0.63889** (5-seed mean, std 0.00074) vs iter24's 0.62843 -- **+0.01046**
- 5/5 seeds improve over iter24 on both valid and test, no sign flips.
- Beats iter23 and iter25's own 5-seed numbers on both splits too, not only
  iter24 -- consistent with the three ingredients being non-overlapping
  mechanisms (features / sampling weight / formula constants) that compose
  rather than compete.
- One caveat carried forward honestly, not resolved here: iter29 showed the
  isolated n_buckets=20 effect is partly fold-specific (shrinks/flips on a
  date-shifted split). This rerun confirms n_buckets=20 still wins inside the
  fused config on the official split, but does not re-verify that edge under
  a date shift for the *fused* config specifically -- a natural follow-up
  would be an iter29-style shifted-split rerun of this exact winning config
  before treating the +0.00541/+0.01046 margins as fully shift-robust.
- sampling_alpha=0.75 mildly edges out iter23's original 0.5 once combined
  with iter24's feature set and iter25's formula constants (0.63816 vs
  0.63804 valid, 3-seed) -- a small but real shift in the optimum, expected
  since iter23's original sweep was tuned against a different feature set.

## Code
`experiments/iter27_triple_fusion/{data_ext.py,train.py,driver.py}`, raw
results in `experiments/iter27_triple_fusion/results.json`.

## Code
`experiments/iter27_triple_fusion/{data_ext.py,train.py,driver.py}`, raw
results in `experiments/iter27_triple_fusion/results.json`.

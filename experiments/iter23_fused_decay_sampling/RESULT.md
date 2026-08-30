# iter23 — fused decay+momentum features ⊕ decay-aware BPR sampling weight (iter19 ⊕ iter22)

## Idea
iter19 (current best, valid 0.62898 / test 0.62615) fused iter16's recency-
decay INPUT features with iter18's session-momentum INPUT features. iter22
(valid 0.62274 / test 0.62101, superseded by iter19 but not overlapping with
it) independently found a real gain from replacing the BPR training loop's
flat `pos_len[user] ** alpha` user-sampling weight with a recency-decayed
version (`decayed_pos_total[user] ** alpha`, halflife=3d matching
`decay_act_3`/`decay_rate_3`, alpha=0.5) — but iter22 only tested this on top
of iter16's feature set, never on iter19's fuller (feature+momentum) set.
iter19 changes model INPUT FEATURES; iter22 changes the BPR TRAINING-TIME
user-SAMPLING weight — non-overlapping mechanisms that had never been
combined. This iteration stacks both.

## Feature set / config
- **Model input features (unchanged from iter19, verbatim)**: `decay_rate_3,
  decay_act_3, tab, last1, lastk_rate, gap` — iter16's decay features +
  iter18's momentum features, fed to the same FM model.
- **BPR training-time user-sampling weight (iter22's mechanism, reused)**:
  `decayed_pos_total[user] ** alpha`, `decayed_pos_total` = the recency-
  decayed count of a user's TRAIN positive rows, decayed to the end of the
  train period (halflife=3d, same time constant as `decay_act_3`/
  `decay_rate_3`), replacing iter19's/iter16's/iter9's flat
  `pos_len[user] ** alpha`. Alpha swept over {0.25, 0.5, 0.75, 1.0}.
- `data_ext.py` = iter19's data_ext.py (fused feature computation,
  unmodified) + `compute_final_decayed_pos` copied verbatim from
  iter22/data_ext.py (works unmodified against this module's extended row
  tuples since it only reads the shared `(date, user_id, ..., label)`
  prefix at indices 0/1/6).
- `train.py` = iter19's train.py (BPR loop, `sample_pairs`/`bpr_step`/
  `build_pos_neg_index`, all copied verbatim) + iter22's `sampling_mode`/
  `alpha`/`decay_halflife` branch in `run_bpr_ext`, copied verbatim from
  iter22/train.py.

## Harness-fidelity check (protocol step 1, required before the sweep)
Before touching the fused (momentum-inclusive) feature set, the new
train.py/data_ext.py combo was validated against iter22's own
already-published numbers on iter16's EXACT feature set (`decay_rate_3,
decay_act_3, tab` — no momentum fields):

```
$ python3 train.py --features decay_rate_3,decay_act_3,tab --sampling_mode flat --alpha 1.0 --seed 0 --quiet
  valid  primary 0.6194308996200562
  test   primary 0.6176092624664307
```
Bit-exact match to iter22's own reported flat/alpha=1.0 seed-0 fidelity
check (`0.6194308996200562` / `0.6176092624664307`, itself bit-exact to
iter16's original numbers).

```
$ for s in 0 1 2; do python3 train.py --features decay_rate_3,decay_act_3,tab --sampling_mode decay --alpha 0.5 --seed $s --quiet; done
  seed 0: valid 0.6217743158340454  test 0.6217406988143921
  seed 1: valid 0.6223915219306946  test 0.6206650733947754
  seed 2: valid 0.6234520673751831  test 0.6207175254821777
```
Bit-exact match, all 3 seeds, to iter22's own `results.json` for
`decay_sampling_alpha0.5` seeds 0-2. **This confirms the fused harness is a
faithful reuse of iter22's sampling mechanism, not a reimplementation
drift**, before any momentum fields were added to the feature set.

## Causality verification
`python3 data_ext.py` (full output; PART A-C are iter19's original
decay/momentum/cross-family checks re-run unmodified against this dir's
copy, PART D is the new sampling-weight-specific check):

```
=== PART A: decay-feature causal spot-checks (brute force) ===
halflife= 3d  decayed_total>0 coverage: 92.29%  (mean=16.753, max=377.162)
25 random rows x 1 halflives: decayed_pos/decayed_total match brute force
(max abs err 1.42e-14). No leakage detected.
zero-activity rows (5 checked): decayed_pos/total correctly 0.0.
same-date-pair edge case (3 rows): decay values identical across the pair,
as expected (same-date rows never see each other). PASSED.

=== PART B: momentum-feature causal spot-checks (brute force) ===
last1 coverage (not user's first row): 98.12%
gap coverage (not user's first row): 98.12%
[3 real users' full chronological sequences, 36 rows total -- all
last1/lastk_sum/gap_ms values matched brute-force manual recount exactly]
--- synthetic same-time_ms tie stress test (momentum) ---
tie stress test: all assertions passed.

=== PART C: cross-family joint edge case (same-date, different time_ms pair) ===
user=1 date=20220412: 3 rows, same calendar date, distinct time_ms
  -> decay features IDENTICAL across the pair (date-level, correctly blind
     to intra-date order); momentum last1 correctly DIFFERS and resolves
     the true time_ms order. No cross-contamination from the join.

=== PART D: decay-aware sampling-weight spot-check (brute force) ===
train period end (reference date ordinal): 738266  (24881 users with >=1 train positive)
30 random users: compute_final_decayed_pos matches brute-force recount of
0.5**(gap_days/3) over all TRAIN positive rows (max abs err 2.66e-15). No
arithmetic error, matches iter22's formula.
zero-train-positive users (5 checked): correctly absent from decayed_pos
dict (never contribute sampling weight).
reference-date edge case (user 550, 1 positives all on last train date
20220421): decayed_pos == raw count exactly (1.000000). PASSED.

All causal spot-checks (decay + momentum + cross-family joint + sampling-weight) passed.
```

`compute_final_decayed_pos` is a TRAINING-TIME-only quantity (per-user
scalar controlling BPR sampling frequency) — it never enters a row's
feature vector, so there is no leakage risk by construction, exactly as in
iter22. PART D confirms the arithmetic itself is correct (matches iter22's
brute-force-verified formula) on top of that structural guarantee.

## Sweep: alpha grid on the FULL fused feature set (3 seeds: 0,1,2)

| config | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|
| `combo_full_flat` (control: fused feats + iter19's flat sampling) | 0.62933, 0.00040 | 0.62632, 0.00037 |
| `combo_full_decay_alpha0.25` | 0.62787, 0.00185 | 0.62684, 0.00165 |
| **`combo_full_decay_alpha0.5`** | **0.63112, 0.00103** | **0.62864, 0.00097** |
| `combo_full_decay_alpha0.75` | 0.63042, 0.00048 | 0.62903, 0.00077 |
| `combo_full_decay_alpha1.0` | 0.62982, 0.00055 | 0.62806, 0.00032 |

`combo_full_flat` (3-seed mean valid 0.62933 / test 0.62632) reproduces
iter19's own published 3-seed `combo_full` phase-1 number
(0.62933/0.62632) **exactly** — confirming this harness's BPR loop and
feature encoding are a faithful, non-drifted copy of iter19's, not just at
seed 0 but across the full 3-seed mean.

alpha=0.5 wins on valid by a clear margin over every other alpha in the
grid (+0.00325 over the next-worst point at alpha=0.25, +0.00070 over the
next-best at alpha=0.75) and over the flat-sampling control (+0.00179
valid / +0.00232 test). The alpha response is **not monotonic** here (unlike
iter22's own sweep on iter16-alone features, which was monotonically
decreasing over {0.5,1.0,1.5,2.0}) — 0.5 and 0.75 are close and both clearly
beat 0.25 and 1.0, consistent with a broad optimum around alpha≈0.5-0.75
rather than a sharp one. Per protocol (select the valid-winner), alpha=0.5
was extended to 5 seeds.

## 5-seed confirmation (`combo_full_decay_alpha0.5`)

| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.62977 | 0.62938 |
| 1 | 0.63133 | 0.62727 |
| 2 | 0.63226 | 0.62928 |
| 3 | 0.63009 | 0.62936 |
| 4 | 0.63203 | 0.63118 |
| **mean** | **0.63109** | **0.62929** |
| **std** | 0.00101 | 0.00124 |

### vs iter19 alone (5-seed, matched seeds)

| seed | Δ valid (iter23 − iter19) | Δ test (iter23 − iter19) |
|---|---|---|
| 0 | +0.00075 | +0.00339 |
| 1 | +0.00226 | +0.00043 |
| 2 | +0.00236 | +0.00315 |
| 3 | +0.00218 | +0.00418 |
| 4 | +0.00304 | +0.00455 |

**5/5 seeds improve on both valid and test, no sign flips.** Mean deltas:
**+0.00211 valid / +0.00314 test** — the test delta alone is ~5.4x iter19's
own test std (0.00058), well clear of noise, and the smallest single-seed
test gain (seed 1, +0.00043) is still a genuine same-direction improvement,
not a wash.

### vs iter22 alone (5-seed)
iter22 (decay-aware sampling on iter16-alone features, no momentum):
valid 0.62274 (std 0.00084) / test 0.62101 (std 0.00050).
**iter23 beats iter22 by +0.00835 valid / +0.00828 test** — expected, since
iter23 additionally carries iter19's momentum features that iter22 never had.

### vs combo_full_flat (in-harness ablation: same features, flat sampling only)
combo_full_flat (3-seed): valid 0.62933 / test 0.62632.
**Δ = +0.00179 valid / +0.00232 test at 3 seeds** — the sampling-weight
change alone (holding the fused feature set fixed) is a real, positive
effect on top of iter19's already-strong feature fusion, not just noise
from re-running with a different seed set.

## Why it likely helps
This is consistent with iter22's own explanation for why decay-aware
sampling helps on iter16-alone features, but now compounds with iter19's
momentum features too: iter19's flat `pos_len`-weighted sampling still
over-emphasizes users who were highly active AT ANY POINT during the log
window, even if their activity (and hence their `decay_act_3`/`decay_rate_3`
AND `last1`/`lastk_rate`/`gap` momentum signals) has since faded to a low,
uninformative value at prediction time. Aligning the sampling weight with
the same decayed-activity notion realigns *which* rows get oversampled as
BPR anchors with *which* rows the model's own (now doubly time-aware, decay
+ momentum) features actually describe well.

## Verdict: **PROMOTE — new current best**

iter23 (`decay_rate_3, decay_act_3, tab, last1, lastk_rate, gap` features +
decay-aware BPR sampling, `decayed_pos_total ** 0.5`, halflife=3d) beats
iter19's published test primary (0.62615) by **+0.00314** (5-seed mean
0.62929), with **all 5 seeds improving on both valid and test, no sign
flips** — a real, consistent-direction margin that clears the promotion bar
this run has used throughout (iter19's test gain over iter16 was +0.00887;
this is a smaller but still clean, non-noise, non-cancelling gain on top of
an already-strong config). This supersedes iter19 as the standing best.

- valid primary mean **0.63109** (std 0.00101), 5-seed — vs iter19 0.62898
  (std 0.00063): **+0.00211**
- test primary mean **0.62929** (std 0.00124), 5-seed — vs iter19 0.62615
  (std 0.00058): **+0.00314**
- Consistent 5/5 seeds, both splits, no sign flips.
- Confirms the Round 6 residual finding (iter19 and iter22 are genuinely
  non-overlapping and stack additively): feature-side recency information
  and sampling-side recency information were each independently useful and
  remain useful when combined, rather than one subsuming the other.

## Code
`experiments/iter23_fused_decay_sampling/{data_ext.py,train.py,driver.py}`,
full sweep + 5-seed data in
`experiments/iter23_fused_decay_sampling/results.json` (15 rows: 3-seed
sweep over `combo_full_flat` + 4 alpha values; +2 extra seeds for
`combo_full_decay_alpha0.5` to complete the 5-seed confirmation).

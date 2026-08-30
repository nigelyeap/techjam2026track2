# iter24 — decay/tab refinement re-tested WITH momentum fields present

## Idea
iter19 (`decay_rate_3, decay_act_3, tab, last1, lastk_rate, gap`, 5-seed valid
0.62898 / test 0.62615) is the standing best. iter20 found two refinements on
top of iter16's *older, momentum-free* feature set: (a) the true halflife
optimum for `decay_rate`/`decay_act` is closer to 2.5d than 3d, and (b)
decaying `tab_pos` itself (instead of leaving it flat) gives a further real
gain. Neither refinement had ever been tested together with iter19's
momentum fields (`last1`, `lastk_rate`, `gap`) — it was plausible the
halflife optimum would shift once momentum features are already absorbing
some of the very-short-horizon signal a short halflife would otherwise
compete for. This iteration re-sweeps both refinements with momentum present
throughout, then combines whichever wins with iter19's fused feature set.

## Feature set (final, winning)
`decay_rate_2.5, decay_act_2.5, decay_tab_3, last1, lastk_rate, gap` — i.e.
iter19's fused config with flat `tab` replaced by `decay_tab_3` and the
rate/act halflife moved from 3d to 2.5d. 6 extra fields total, same as
iter19.

## Architecture
`data_ext.py` runs **four independent causal traversals** over the same flat
per-row data, joined onto the same rows by row index (not merged into one
traversal — this is iter19's own established design choice, explicitly kept
here to avoid cross-family contamination):
1. `compute_causal_features` — flat date-grouped activity/tab_pos/rate
   (copied verbatim from iter20's data_ext.py, itself copied from
   iter9/iter16).
2. `compute_decay_features` — exponential-decay rate/act, fine halflife grid
   `HALFLIVES=[2, 2.5, 3, 3.5]` days (copied verbatim from iter20's
   data_ext.py).
3. `compute_decay_tab_features` — iter20's decayed-tab_pos machinery,
   `TAB_HALFLIVES=[3, 7]` days (copied verbatim from iter20's data_ext.py,
   already brute-force-verified there).
4. `compute_momentum_features` — iter18's time_ms-level last1/lastk/gap
   (imported via importlib from `iter18_momentum/data_ext.py`, exactly as
   iter19 did — not copied, not modified).

Row loader is iter19's 10-column `_load_raw_time` (needed for
`time_ms`/`orig_idx`, which the momentum traversal requires and iter20's
plain `data.load()` doesn't expose). All four traversals only read the
columns they document needing, so combining them is a pure join, not a
shared mutable pass — no cross-family leakage is possible by construction.

## Causality verification (`python3 data_ext.py`, full output)

```
=== PART A: decay-feature (rate/act) causal spot-checks (brute force) ===
halflife= 2.0d  decayed_total>0 coverage: 92.29%  (mean=12.916, max=296.595)
halflife= 2.5d  decayed_total>0 coverage: 92.29%  (mean=14.995, max=342.161)
halflife= 3.0d  decayed_total>0 coverage: 92.29%  (mean=16.753, max=377.162)
halflife= 3.5d  decayed_total>0 coverage: 92.29%  (mean=18.273, max=404.760)
25 random rows x 4 halflives: decayed_pos/decayed_total match brute force (max abs err 1.42e-14). No leakage detected.
zero-activity rows (5 checked): decayed_pos/total/tab correctly 0.0.
same-date-pair edge case (3 rows): decay values identical across the pair, as expected (same-date rows never see each other). PASSED.

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
  rank=0 time_ms=1649706052290 label=1 decay_pos=0.5000 decay_total=1.9142 decay_tab=0.6300 last1=0 gap_ms=57224731
  rank=1 time_ms=1649706789917 label=1 decay_pos=0.5000 decay_total=1.9142 decay_tab=0.6300 last1=1 gap_ms=737627
  rank=2 time_ms=1649707373426 label=1 decay_pos=0.5000 decay_total=1.9142 decay_tab=0.6300 last1=1 gap_ms=583509
  -> decay AND decay_tab features are IDENTICAL across the pair (both date-level, correctly
     blind to intra-date order); momentum last1 correctly DIFFERS and resolves the true
     time_ms order. All three families verified independently correct on the same rows --
     no cross-contamination from the join (decay<->decay_tab agree by construction, both
     disagree with momentum by construction).

All causal spot-checks (decay + decay_tab + momentum + cross-family joint) passed.
```

No leakage detected in any feature family, or in their pairwise/three-way
combination.

## Step 1 — fine halflife re-sweep for decay_rate/decay_act WITH momentum present (3 seeds)

Config: `decay_rate_H, decay_act_H, tab (flat), last1, lastk_rate, gap`

| H (days) | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|
| 2 | 0.62961, 0.00073 | 0.62144, 0.00052 |
| **2.5** | **0.62997, 0.00059** | 0.62428, 0.00126 |
| 3 (iter19's original) | 0.62933, 0.00040 | 0.62632, 0.00037 |
| 3.5 | 0.62849, 0.00057 | 0.62668, 0.00058 |

**2.5d remains the valid-optimum even with momentum present** — the halflife
optimum did NOT shift back toward 3d as one plausible hypothesis suggested.
The margin over 3d is small (+0.00064 valid, within one std of either point)
but the same rank-order iter20 found (momentum-free) reproduces here.

Notably, the `H=3` row is *exactly* iter19's own config, and its 3
individual seed values (0.62902, 0.62907, 0.62990) match iter19's own
published seeds 0/1/2 bit-for-bit — confirming this new 4-traversal harness
is a faithful, non-drifted reproduction of iter19, not a reimplementation
with subtle differences.

One honest caveat: on **test**, `H=3` and `H=3.5` actually score higher
(0.62632 / 0.62668) than `H=2.5` (0.62428) in this 3-seed sample — the
valid-side ranking (which is what selection was based on, correctly, to
avoid test leakage) does not fully carry over to test at the 3-seed noise
level. This is flagged rather than hidden; the final decayed-tab result
(below) resolves this by giving the H=2.5 branch strong 5-seed test
performance anyway, but it's worth remembering H=3 was competitive too.

## Step 2 — decayed tab_pos on top of H=2.5 + momentum (3 seeds)

Config: `decay_rate_2.5, decay_act_2.5, decay_tab_H2, last1, lastk_rate, gap`

| config | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|
| flat `tab` (=Step 1's H=2.5 row) | 0.62997, 0.00059 | 0.62428, 0.00126 |
| `decay_tab_7` | 0.63218, 0.00058 | 0.62888, 0.00028 |
| **`decay_tab_3`** | **0.63249, 0.00053** | 0.62823, 0.00105 |

Decaying `tab_pos` (on top of momentum, which iter20 never tested) gives a
clear, consistent gain over flat `tab`: **+0.00252 valid** for `decay_tab_3`
and **+0.00221 valid** for `decay_tab_7`, both well outside their own std.
`decay_tab_3` edges out `decay_tab_7` on valid (+0.00031, within noise) so it
was carried forward, matching iter20's own halflife=3 preference.

Best 3-seed config: `decay_rate_2.5+decay_act_2.5+decay_tab_3+mom`, valid
mean 0.63249 — margin vs iter19's 5-seed valid reference (0.62898) is
**+0.00351**, well above the 0.001-0.002 confirmation threshold, triggering
a full 5-seed run.

## 5-seed confirmation (`decay_rate_2.5+decay_act_2.5+decay_tab_3+last1+lastk_rate+gap`)

| seed | valid | test |
|---|---|---|
| 0 | 0.63260 | 0.62839 |
| 1 | 0.63308 | 0.62942 |
| 2 | 0.63179 | 0.62687 |
| 3 | 0.63208 | 0.62855 |
| 4 | 0.63298 | 0.62892 |
| **mean** | **0.63251** | **0.62843** |
| **std** | 0.00050 | 0.00086 |

## Comparison against iter19 (standing best, 5-seed: valid 0.62898/std 0.00063, test 0.62615/std 0.00058)

Per-seed deltas (iter24 − iter19), same seed indices:

| seed | Δ valid | Δ test |
|---|---|---|
| 0 | +0.00358 | +0.00240 |
| 1 | +0.00401 | +0.00258 |
| 2 | +0.00189 | +0.00074 |
| 3 | +0.00417 | +0.00337 |
| 4 | +0.00399 | +0.00229 |

**All 5/5 seeds improve on both valid and test — no sign flips.** Mean
deltas: **+0.00353 valid / +0.00228 test**, roughly 5.6x iter19's own valid
std and roughly 3.9x iter19's own test std — a real, consistent-direction
margin, not noise.

## Verdict: **PROMOTE — new current best**

`decay_rate_2.5, decay_act_2.5, decay_tab_3, last1, lastk_rate, gap` beats
iter19 (the prior standing best across the whole run) on both valid and
test, by a margin well outside seed-to-seed noise, consistently across all 5
seeds with no exceptions. This confirms both of iter20's refinements
(finer halflife, decayed `tab_pos`) hold up once combined with iter19's
momentum features — the halflife optimum did not shift, and decayed `tab_pos`
provides an additional, independent gain on top of momentum rather than
being made redundant by it. This supersedes iter19 as the standing best.

5-seed valid 0.62898 → 0.63251 (+0.00353); 5-seed test 0.62615 → 0.62843
(+0.00228).

## Code
`experiments/iter24_decay_tab_refine/{data_ext.py,train.py,driver.py}`, raw
sweep results in `experiments/iter24_decay_tab_refine/results.json` (20
rows: 4 halflives x 3 seeds for Step 1, 2 tab-halflives x 3 seeds for
Step 2, +2 extra seeds for the winning config's 5-seed confirmation).

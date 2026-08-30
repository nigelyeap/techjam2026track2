# iter20 — finer halflife grid + decayed tab_pos, refining iter16

Note: the dispatched agent's session was terminated mid-run by a platform
session-limit error (not a task failure). The underlying sweep (restarted
once after an earlier silent driver crash, per the orchestrator's
intervention) ran to completion and wrote all results to `results.json`;
the orchestrator re-ran the causality verification directly
(`python3 data_ext.py`) and wrote this file from the completed sweep data.

## Axis A — finer halflife grid around iter16's 3-day peak (3 seeds each)

iter16's original sweep only tested {1,3,7,14,30} days and found a peak at
3d. This axis tests a finer grid: {1.5, 2, 2.5, 3, 3.5, 4, 5} days for both
`decay_rate_H`/`decay_act_H` together (same halflife, matching iter16's
config structure), + `tab` (flat).

| halflife | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|
| 1.5d | 0.61953, 0.00006 | 0.61131, 0.00193 |
| 2d | 0.62095, 0.00097 | 0.61453, 0.00154 |
| **2.5d** | **0.62121, 0.00018** | 0.61632, 0.00317 |
| 3d (iter16 original) | 0.62028, 0.00061 | 0.61727, 0.00190 |
| 3.5d | 0.61983, 0.00006 | 0.61726, 0.00229 |
| 4d | 0.61948, 0.00014 | 0.61825, 0.00123 |
| 5d | 0.61790, 0.00048 | 0.61777, 0.00056 |

The finer grid reveals the true valid-optimum sits at **2.5 days**, not
exactly 3 — a small refinement, +0.00093 valid over iter16's original 3d
point (well within the noise range these 3-seed measurements show, e.g. 2d's
own std is nearly as large as this gap). Confirms iter16's coarse-grid
finding was directionally correct (peak near 3d, non-monotonic, falls off on
both sides) but the exact optimum is closer to 2.5d.

## Axis B — decaying `tab_pos` itself (previously left flat in iter16)

Tested `decay_rate_2.5 + decay_act_2.5 + decay_tab_H` (tab halflife H swept
over {3, 7} days, guided by Axis A's 2.5d winner) vs iter16-style flat `tab`:

| config | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|
| `decay_rate_2.5+decay_act_2.5+tab` (flat tab) | 0.62121, 0.00018 | 0.61632, 0.00317 |
| `decay_rate_2.5+decay_act_2.5+decay_tab_7` | 0.62185, 0.00023 | 0.61937, 0.00122 |
| **`decay_rate_2.5+decay_act_2.5+decay_tab_3`** | **0.62226 (3-seed), 5-seed: 0.62268, 0.00055** | 5-seed: **0.61938, 0.00188** |

Decaying `tab_pos` (halflife=3d, matching the rate/act halflife family) does
help modestly on top of the 2.5d rate/act combo — a further +0.00147 valid
over the flat-tab version, consistent in direction on test as well
(+0.00306). This answers iter16's open question: `tab_pos` benefits from
decay too, just like `rate`/`activity` did.

## Causality verification (re-run by orchestrator, full output)

```
--- causal spot-checks: decayed_pos/decayed_total (rate/act, brute force) ---
25 random rows x 3 halflives: all decayed_pos/decayed_total match brute
force (max abs err 1.42e-14). No leakage detected.

--- causal spot-checks: decayed_tab_pos (NEW, brute force) ---
30 random rows x 2 tab-halflives: all decayed_tab_pos match brute force
(max abs err 1.42e-14). No leakage detected.
zero-activity rows (5 checked): decayed_pos/total/tab correctly 0.0.
zero-tab_pos-but-nonzero-activity rows (5 checked): decayed_tab_pos
correctly 0.0 despite nonzero decay_rate/act.

--- same-date-pair edge case (decay_tab) ---
tab halflife=3d, (user,tab,date) triple with 3 same-date positives:
  identical decayed_tab_pos across the same-date pair, as expected. PASSED.

All causal spot-checks passed (rate/act fine grid + NEW decay_tab). No
same-date or future leakage detected.
```

## 5-seed confirmation (`decay_rate_2.5+decay_act_2.5+decay_tab_3`)

| seed | valid | test |
|---|---|---|
| 0 | 0.62261 | 0.62097 |
| 1 | 0.62311 | 0.61737 |
| 2 | 0.62190 | 0.62094 |
| 3 | 0.62345 | 0.61681 |
| 4 | 0.62230 | 0.62082 |
| **mean** | **0.62268** | **0.61938** |
| **std** | 0.00055 | 0.00188 |

vs iter16 (5-seed, 3d config): valid 0.62030 / test 0.61698. **Δ = +0.00238
valid / +0.00240 test** — a real, small-to-moderate, causally-clean gain
over iter16's original config, generalizing to test.

## Verdict: real gain over iter16, but **superseded by iter19**

`decay_rate_2.5+decay_act_2.5+decay_tab_3` beats iter16 by a real margin
(+0.00238 valid), but iter19 (`experiments/iter19_decay_momentum/`, decay +
momentum feature fusion, valid 0.62898/test 0.62615) beats it by a much
wider one. **Not promoted as current best** — iter19 remains the standing
best. This iteration's two findings are logged as residuals for a future
round: (1) the true halflife optimum is closer to 2.5d than 3d; (2)
`tab_pos` benefits from decay too, an untested addition on top of iter19's
already-promoted feature set (iter19 kept `tab` flat, inherited unchanged
from iter16) — combining iter20's decayed-tab finding with iter19's momentum
fusion is a natural next step.

## Code
`experiments/iter20_decay_refine/{data_ext.py,train.py,driver.py}`,
raw sweep results in `experiments/iter20_decay_refine/results.json`.

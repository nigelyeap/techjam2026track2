# iter28 — DeepFM (iter26) stacked on iter24's refined feature set

Note: the dispatched agent's session was terminated mid-run by the same
Round-8 platform session-limit event that killed iter27/29/30
simultaneously. `driver_log.txt` shows all planned phases (0-1) completed
and printed their own summary lines before the kill; only the final
RESULT.md writeup was lost (the file left behind was a 66-byte placeholder
stub). The orchestrator wrote this file directly from `results.json`/
`driver_log.txt` — no rerun was needed.

## Idea
iter24 (`decay_rate_2.5, decay_act_2.5, decay_tab_3, last1, lastk_rate,
gap`) is the current valid-best (5-seed valid 0.63251/test 0.62843). iter26
independently found that adding a DeepFM-style MLP tower on top of iter19's
(older) feature set gave a real, if noisier, gain (5-seed valid
0.63079/test 0.63033 vs iter19's 0.62898/0.62615). The two changes are
non-overlapping (input features vs. model architecture) and had never been
combined. This iteration stacks iter26's DeepFM tower on iter24's refined
feature set.

## Harness-fidelity check (Phase 0, MLP disabled)
5-seed run with `use_deep=False` on iter24's exact feature set:

valid mean=0.63251 std=0.00050 | test mean=0.62843 std=0.00086

**Bit-exact match to iter24's own published 5-seed numbers** (0.63251/
0.62843) — confirms this harness is a faithful, non-drifted reproduction of
iter24 before the deep component is added.

## Phase 1 — width sweep {16, 32, 64}, single hidden layer (3 seeds)

| width | valid primary (mean) | test primary (mean) |
|---|---|---|
| 16 | 0.63221 | 0.62980 |
| **32** | **0.63268** | 0.62984 |
| 64 | 0.63184 | 0.62934 |

Width 32 wins on valid by a 3-seed margin of **+0.00017** over iter24's
5-seed reference (0.63251) — well below the ~0.001-0.002 confirmation
threshold used throughout this project. Per the dispatch instructions
(report the default width at 5 seeds regardless), width 32 was extended to
5 seeds anyway.

## 5-seed confirmation (`deep_h32` on iter24's refined features)

| seed | valid | test |
|---|---|---|
| 0 | 0.63196 | 0.62798 |
| 1 | 0.63484 | 0.63132 |
| 2 | 0.63125 | 0.62979 |
| 3 | 0.63183 | 0.63051 |
| 4 | 0.63234 | 0.63020 |
| **mean** | **0.63244** | **0.62996** |
| **std** | 0.00125 | 0.00111 |

## Comparison against iter24 (standing valid-best, 5-seed: valid
0.63251/std 0.00050, test 0.62843/std 0.00086)

| | iter24 (5-seed) | iter28 deep_h32 (5-seed) | delta |
|---|---|---|---|
| valid primary | 0.63251 | 0.63244 | **−0.00007** |
| test primary | 0.62843 | 0.62996 | **+0.00153** |

**Valid is flat-to-negative** (−0.00007, well inside noise — 2/5 seeds
higher, 3/5 lower than iter24's own seed values) while **test shows a
modest, fairly consistent gain** (+0.00153, all 5 seeds at or above iter24's
mean). This is the same pattern that produced the project's four-way
valid/test crossover in Round 7 (iter23/24/25/26): a change that helps test
without a corresponding valid signal.

## Verdict: **REJECT — does not clear the valid-only promotion bar**

Per this project's protocol, promotion decisions are made on **valid only**;
test is monitoring, not selection criteria. iter28's deep_h32 does not beat
iter24 on valid (in fact slightly below, within noise), so despite the
tempting test-side gain it is **not promoted**. This is exactly the
scenario the project's test-peeking discipline exists to guard against —
promoting on a test-side signal here would risk overfitting to test via
repeated peeking, which the project's own ledger protocol explicitly warns
against. Logged as a residual finding: DeepFM's benefit (established
independently in iter26) does not clearly stack with iter24's specific
refined-feature set on the metric that matters for selection, though it may
be worth a fresh look if a future architecture-search round finds a
DeepFM variant with a genuine valid-side signal.

## Code
`experiments/iter28_deepfm_refined_features/{data_ext.py,train.py,model.py,driver.py}`,
raw results in `experiments/iter28_deepfm_refined_features/results.json`
(19 rows: 5-seed Phase 0 + 3-seed width sweep + 2 extra seeds for the
deep_h32 5-seed confirmation), full run trace in `driver_log.txt`.

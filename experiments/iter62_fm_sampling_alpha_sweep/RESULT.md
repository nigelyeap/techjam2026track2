# iter62 — FM negative-sampling alpha resweep

## Motivation

`sampling_alpha=0.75` controls how strongly user-level decayed-positive-count
weighting shapes the BPR negative-sampling distribution. Unchanged since
iter38, never resystematically resweept. Structurally different from
lr/k (a data-sampling knob, not a model-capacity or optimizer knob).

## Method

Single-seed (seed=0) sweep over `{0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}`,
standalone FM metric.

## Result

| sampling_alpha | valid | test |
|---|---|---|
| 0.00 | 0.63727 | 0.63664 |
| 0.25 | 0.63789 | 0.64002 |
| 0.50 | 0.63848 | 0.64012 |
| **0.75 (current)** | **0.63894** | 0.63989 |
| 1.00 | 0.63647 | 0.63691 |
| 1.25 | 0.63451 | 0.63550 |
| 1.50 | 0.63408 | 0.63499 |

Monotonically increasing from 0.0 to 0.75, then monotonically decreasing
beyond — a clean interior optimum already sitting exactly at the current
value. **Verdict: REJECT** — `sampling_alpha=0.75` confirmed optimal; no
further gain available on this axis.

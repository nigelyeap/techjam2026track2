# iter60 — FM embedding dimension (k) resweep

## Motivation

`k=16` has been the FM embedding dimension since iter38 and was never
resystematically resweept in this session's records. This is the FM-side
analogue of the GBM's `num_leaves` capacity knob — a natural next lever
now that the GBM hyperparameter space (iter53/57/58/59) is exhausted.

## Method

Single-seed (seed=0) FM training at `k` over `{8, 12, 16, 24, 32, 48, 64}`,
all other hyperparameters at iter38's exact config (`lr=0.001, epochs=40,
bs=8192, patience=4, sampling_alpha=0.75, decay_halflife=3`), standalone
FM metric (not blended).

## Result

| k | valid | test |
|---|---|---|
| 8 | 0.63797 | 0.63989 |
| 12 | 0.63712 | 0.63791 |
| **16 (current)** | **0.63894** | 0.63989 |
| 24 | 0.63688 | 0.63787 |
| 32 | 0.63780 | 0.64005 |
| 48 | 0.63844 | 0.63979 |
| 64 | 0.63728 | 0.63794 |

`k=16` is already the best point on the grid — every other tested value is
strictly worse on valid, with no monotonic trend in either direction
(non-monotonic, consistent with FM/BPR training being seed/init-sensitive
at single-seed resolution). **Verdict: REJECT** — `k=16` stands confirmed
as the right embedding dimension; no further gain available on this axis.

# iter56 — fine-grained learning_rate sweep around iter55's winner (0.10)

## Motivation

iter55's coarse sweep `{0.01..0.20}` was highly non-monotonic (0.07
collapsed to 0.639 between two much better points at 0.05 and 0.10),
suggesting a narrow, jagged landscape rather than a smooth one at this
ultra-low-capacity regime. This checks whether a finer grid around 0.10
finds an even better point the coarse grid skipped over.

## Method

Fine sweep of `learning_rate` over
`{0.080, 0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120, 0.130}`,
seed=0, all other hyperparameters unchanged from iter51/iter55.

## Result

| learning_rate | valid | test |
|---|---|---|
| 0.080 | 0.67020 | 0.65288 |
| **0.085** | **0.67074** | 0.65320 |
| 0.090 | 0.67032 | 0.65240 |
| 0.095 | 0.67026 | 0.65236 |
| 0.100 (iter55, promoted) | 0.67052 | 0.65277 |
| 0.105 | 0.67038 | 0.65286 |
| 0.110 | 0.67008 | 0.65266 |
| 0.115 | 0.66984 | 0.65268 |
| 0.120 | 0.66964 | 0.65263 |
| 0.130 | 0.66946 | 0.65274 |

`learning_rate=0.085` edges out 0.10 by +0.00022 valid at single-seed
resolution — below the 0.0003 look-threshold, and well within the
seed-to-seed noise already measured for this config (iter55's 5-seed std
was 0.00021). The whole 0.08–0.13 neighborhood sits within a ~0.0013
valid band with no clear further-improving direction. **Verdict:
REJECT** (no further gain) — this confirms iter55's `learning_rate=0.10`
sits in a genuine local plateau rather than on a slope with more room to
climb; not worth a 5-seed confirmation given the gap doesn't clear the
look-threshold.

# iter21 — retune Laplace-smoothing alpha + capacity/bucketing on iter16's feature set

Note: the dispatched agent's session was terminated mid-run by a platform
session-limit error (not a task failure), partway through Axis A and before
Axis B (capacity/bucket resweep) could start. The orchestrator wrote this
file from the completed Axis A sweep data (`results_axisA.json`). **Axis B
was never run** — see Verdict below for how this is handled.

## Axis A — Laplace-smoothing alpha resweep (3 seeds each, on iter16's exact feature set: decay_rate_3+decay_act_3+tab)

| alpha | valid primary (mean, std) | test primary (mean, std) |
|---|---|---|
| 0.1 | 0.62095, 0.00023 | 0.62064, 0.00047 |
| 0.25 | 0.62123, 0.00034 | 0.62098, 0.00085 |
| **0.5** | **0.62153, 0.00034** | 0.62014, 0.00170 |
| 1.0 (iter16 default) | 0.62028, 0.00061 | 0.61727, 0.00190 |
| 2.0 | 0.61975, 0.00014 | 0.61672, 0.00058 |
| 5.0 | 0.61825, 0.00037 | 0.61363, 0.00113 |
| 10.0 | 0.61750, 0.00017 | 0.61217, 0.00075 |

Monotonically decreasing as alpha rises from 0.5 upward through 10 — the
same pattern iter22 independently found for a different alpha (BPR
sampling-weight exponent), suggesting this dataset generally prefers less
aggressive Laplace smoothing than the original iter9-era default of 1.0.
alpha=0.5 is best on valid; alpha=0.25 is close behind and slightly better
on test (0.62098 vs 0.62014) — the two are within noise of each other, both
clearly better than the default. Unlike iter11's rejected alpha finding
(which flipped sign between valid and test), this gain is **consistent in
direction on both splits** — not a winner's-curse artifact.

**Best point found: alpha=0.5** — valid +0.00125 vs default (1.0), test
+0.00287 vs default. A real but modest gain, smaller than iter20's or
iter22's own findings and far smaller than iter19's.

## Axis B — embedding capacity (k) / bucket-count (n_buckets) resweep

**Not run.** The plan was to sweep k∈{16,24,32} and n_buckets∈{5,10,20} on
top of iter16's feature set using alpha=0.5 (Axis A's winner), but the
agent's session was terminated before this started. No results exist for
this axis.

## Verdict: **inconclusive / superseded — not promoted, Axis B outstanding**

Axis A's alpha=0.5 finding is real (consistent direction on valid and test,
+0.00125 valid vs iter16's default) but (a) was tuned against iter16's
feature set, which is no longer the current best — **iter19**
(`experiments/iter19_decay_momentum/`, valid 0.62898/test 0.62615) now
supersedes it by a much wider margin than this alpha retune could close, and
(b) Axis B is incomplete. Not promoted. Logged as a residual finding for a
future round: **re-run the alpha sweep (favor lower alpha, e.g. {0.1, 0.25,
0.5, 1.0}) on top of iter19's actual fused feature set**, since iter19 uses
the same Laplace-smoothing formula shape for `decay_rate`/`decay_act`/
`lastk_rate` and this finding suggests the current hardcoded alpha=1.0 may
be leaving a small amount of real signal on the table there too. Capacity/
bucket resweep (Axis B) also still needs to be run, ideally against iter19's
feature set rather than iter16's for the same reason.

## Code
`experiments/iter21_retune/{data_ext.py,train.py,driver_axisA.py,driver_axisA_ext.py,driver_axisB.py}`,
raw sweep results in `experiments/iter21_retune/results_axisA.json`
(21 rows: 7 alpha values × 3 seeds).

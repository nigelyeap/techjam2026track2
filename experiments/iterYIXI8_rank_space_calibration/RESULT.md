# iterYIXI8 — rank-space calibration and top-weighted blending

## Final verdict: REJECT

The established within-user percentile transform remains the validation
winner. None of the seven non-reference monotonic transforms clears the
project's +0.0003 preliminary threshold at the frozen reference weights, so
no alternative is eligible for weight optimization, model-specific
refinement, shifted-split refitting, five-seed confirmation, or promotion.

```text
selected transform = percentile p (unchanged reference)
selected weights   = 0.24 FM / 0.40 LightGBM / 0.36 XGBoost
valid primary      = 0.68237525
published test     = 0.66948622
delta              = +0.00000000
```

The strongest new transform is `p^0.5`, but it scores **0.68235779**, a
delta of **-0.00001746**. It slightly improves GAUC while reducing nDCG@5,
leaving official primary fractionally worse.

YIXI7 remains the repository-wide best system at **0.68415534 valid /
0.67437255 test**. A 6h candidate would need to beat the requested YIXI5
reference first and then survive comparison with that newer best; no new
transform reaches even the first gate.

## File and condition separation

Following iter44's separation of experimental conditions, this iteration
contains:

- `freeze_predictions.py`: exact component/reference fidelity and immutable
  validation prediction cache; it never predicts test;
- `phase_a_transforms.py`: predeclared common transforms at fixed weights,
  using only frozen arrays and no model training;
- `phase_b_weight_calibration.py`: preliminary-gated weight optimization plus
  a mandatory reference-grid/plateau reproduction;
- `phase_c_shifted_robustness.py`: shifted-split eligibility gate and the
  predeclared earlier date boundaries;
- `diagnose_ties.py`: post-selection validation-only stable-sort/tie checks;
- `common.py`: label-free transformations, grids, hashes, and shared checks.

Machine-readable outputs are `frozen_predictions.json`,
`phase_a_results.json`, `phase_b_results.json`, `phase_c_results.json`, and
`diagnostic_results.json`. The regeneratable frozen validation arrays live in
`.frozen_valid_predictions.npz` inside this directory.

No existing experiment, ledger, submission, summary, or shared source file
was modified.

## Mandatory harness-fidelity check

Before any 6h code was written, the unmodified repository harness ran as:

```text
python3 make_submission.py /tmp/yixi6h_submission_check.csv
```

The current repository harness points to iter63 and reproduced:

```text
LightGBM standalone: valid=0.67168 test=0.65353
FM standalone:       valid=0.63988 test=0.64187
iter63 blend:        valid=0.67606 test=0.65955
submit.py format/alignment check: PASSED (170588 rows)
```

The dedicated 6h prediction harness then reproduced both relevant percentile
systems within `1e-8`:

| frozen validation reference | weights FM/LGB/XGB | published | reproduction |
|---|---:|---:|---:|
| requested YIXI5 | 0.24 / 0.40 / 0.36 | **0.68237525** | **0.68237525** |
| current YIXI7 | 0.24 / 0.42 / 0.34 | **0.68415534** | **0.68415534** |

It also reproduced each component:

| component | valid primary |
|---|---:|
| unchanged five-seed FM | 0.63987792 |
| YIXI5 LightGBM | 0.67167872 |
| unchanged YIXI5/YIXI7 XGBoost | 0.66755420 |
| YIXI7 tuned LightGBM | 0.67689133 |

There were no prior saved component predictions in YIXI5/YIXI7. The harness
therefore fit each specified model once to materialize the validation arrays,
then SHA-256 hashed them. Phase A and every calibration runner loaded and
verified those exact immutable arrays; they did not retrain a base model.
Test predictions were not computed by the frozen harness.

## Predeclared label-free transforms

For each model independently, average-tie rank state was computed inside each
user only. Higher model scores map to larger percentile `p` and descending
rank 1. No label enters any transform.

The fixed transform set was:

```text
percentile:       p
power:            p^0.5, p^1.5, p^2, p^3
clipped logit:    log((p + 0.001) / (1 - p + 0.001))
reciprocal rank:  1 / descending_rank
NDCG rank:        1 / log2(descending_rank + 1)
```

Power transforms already lie in `[0,1]`. The clipped logit is rescaled by its
fixed theoretical `p=0` and `p=1` endpoints. Reciprocal and NDCG curves are
rescaled inside each user so that the bottom and top ranks map to 0 and 1
(single-row groups map to 1). These normalizations are deterministic,
label-free, and identical for FM, LightGBM, and XGBoost.

## Validation-only policy and gates

- Phase A used the exact frozen YIXI5 predictions and fixed 0.24/0.40/0.36
  weights.
- All models received the same common transform initially; there was no
  model-specific Cartesian search.
- A non-reference transform required at least +0.0003 valid primary before
  receiving a Phase B weight sweep.
- Phase B's allowed grid was predeclared as the established 0.10 simplex plus
  local 0.02 refinement.
- A tiny model-specific refinement was predeclared only after a common
  transform cleared the gate: retain the selected transform on two models and
  restore percentile on exactly one. The gate never opened.
- A new selected calibration would require +0.001 and five-seed confirmation
  before being called real.
- Temporal robustness was reserved for the frozen selected non-reference
  calibration, using the earlier date shift. There was no such candidate.
- No runner used test for selection. Because the unchanged reference remained
  selected, no new test prediction was made.

## Phase A — common monotonic transforms

The comparison below holds component predictions and weights fixed:

| common transform | GAUC | nDCG@5 | primary | delta vs percentile |
|---|---:|---:|---:|---:|
| **percentile `p`** | **0.77264076** | **0.59210974** | **0.68237525** | — |
| `p^0.5` | 0.77277076 | 0.59194475 | 0.68235779 | -0.00001746 |
| `p^1.5` | 0.77173936 | 0.59189510 | 0.68181723 | -0.00055802 |
| `p^2` | 0.77084315 | 0.59164828 | 0.68124568 | -0.00112957 |
| `p^3` | 0.76882249 | 0.59104574 | 0.67993414 | -0.00244111 |
| clipped logit | 0.76910657 | 0.59006268 | 0.67958462 | -0.00279063 |
| reciprocal rank | 0.76728851 | 0.58959448 | 0.67844152 | -0.00393373 |
| NDCG rank | 0.76793945 | 0.58984667 | 0.67889309 | -0.00348216 |

### Metric diagnosis

`p^0.5` is the only near-tie. Relative to percentile:

```text
GAUC delta     = +0.00013000
nDCG@5 delta   = -0.00016499
primary delta  = -0.00001746
```

Flattening the percentile curve slightly helps whole-user ordering but harms
the top five. That trade is smaller than the noise threshold and has negative
official primary, so it cannot be pursued.

Increasing the power exponent worsens monotonically: the more the blend
concentrates score differences near the top percentile, the more GAUC falls,
without a compensating nDCG gain. The explicitly top-weighted reciprocal and
NDCG curves are worst. This provides a consistent diagnosis rather than an
isolated grid result: the existing linear percentile scale already balances
whole-list GAUC and top-five nDCG better than more aggressive top weighting.

**Phase A finding: no promising non-reference transform.**

## Phase B — gated weight calibration

No new transform reached +0.0003, so optimizing its weights would violate the
project rule that sub-threshold single-run results are not chased. The common
transform weight sweeps and model-specific refinement were therefore skipped.

The percentile reference itself was rerun through the exact permitted grid as
a fidelity and plateau check. It returned the published optimum exactly:

```text
weights = 0.24 FM / 0.40 LightGBM / 0.36 XGBoost
valid   = 0.68237525
```

Nearby three-model points are smooth and close rather than an isolated spike:

| FM / LGB / XGB | GAUC | nDCG@5 | primary | delta |
|---:|---:|---:|---:|---:|
| **0.24 / 0.40 / 0.36** | **0.77264076** | 0.59210974 | **0.68237525** | — |
| 0.24 / 0.34 / 0.42 | 0.77249843 | **0.59224164** | 0.68237007 | -0.00000519 |
| 0.24 / 0.42 / 0.34 | 0.77255875 | 0.59208709 | 0.68232292 | -0.00005233 |
| 0.22 / 0.42 / 0.36 | 0.77237916 | 0.59221989 | 0.68229949 | -0.00007576 |
| 0.26 / 0.36 / 0.38 | 0.77255446 | 0.59194934 | 0.68225193 | -0.00012332 |

The unchanged percentile reference also remains **-0.00178009** below the
newer YIXI7 system. Since no new transform was selected, transferring or
re-optimizing it on YIXI7 would be an identity operation, not a 6h finding.

## Phase C — temporal robustness gate

The earlier split was predeclared exactly as in prior project checks:

```text
train: 2022-04-05 .. 2022-04-18
valid: 2022-04-19 .. 2022-04-25
test:  2022-04-26 .. 2022-05-05
```

A shifted refit was not eligible. Phase B selected the exact reference
transform on every model with the exact reference weights. Candidate and
reference are therefore the same mathematical function on any prediction
array and any date split:

```text
shifted candidate - shifted reference = 0 exactly
```

There is no positive calibration effect whose temporal robustness could be
tested. Running three expensive shifted base-model refits to compare an array
with itself would not add evidence and would conflict with the preliminary
gate. If a non-reference transform had survived Phase A/B, the shifted refit
would have been mandatory before promotion.

## Tie and stable-sort diagnostics

Every transform is monotonic and uses average ranks, so it preserves existing
component ties. The resulting blend rankings remain highly differentiated:

| score source | mean within-user unique fraction |
|---|---:|
| FM percentile | 0.99993939 |
| LightGBM percentile | 0.98643366 |
| XGBoost percentile | 0.93628129 |
| selected percentile blend | **0.99877159** |
| minimum across all tested transform blends | **0.99877159** |

The selected blend emits 13,995 distinct floating-point values across
validation. Alternative curves emit 27,585-34,457 distinct blend values; none
is heavily tied.

Trivial ranking controls reproduce the expected floor:

| control | GAUC | nDCG@5 | primary |
|---|---:|---:|---:|
| all constant | 0.50000000 | 0.46734253 | 0.48367125 |
| seeded random | 0.49867553 | 0.46709988 | 0.48288769 |

The selected score is almost fully unique within user and far above both
controls. Neither its score nor the negative transform findings are created
by stable sorting of tied ranks.

## Test discipline

No non-reference candidate passed validation, so no new test prediction was
computed. The unchanged selected reference already has the previously frozen
test score **0.66948622**. Re-running it would provide no new information and
would violate the intent of one-time test access.

For clarity, the systems left after 6h are:

| system | valid | test | status |
|---|---:|---:|---|
| YIXI5 percentile reference retained by 6h | 0.68237525 | 0.66948622 | unchanged |
| **YIXI7 current best** | **0.68415534** | **0.67437255** | remains promoted |
| new 6h transform | — | — | none passed validation |

## Reproduction

Run from repository root:

```text
python3 experiments/iterYIXI8_rank_space_calibration/freeze_predictions.py
python3 experiments/iterYIXI8_rank_space_calibration/phase_a_transforms.py
python3 experiments/iterYIXI8_rank_space_calibration/phase_b_weight_calibration.py
python3 experiments/iterYIXI8_rank_space_calibration/phase_c_shifted_robustness.py
python3 experiments/iterYIXI8_rank_space_calibration/diagnose_ties.py
```

The first command materializes validation predictions once. Phase A and later
commands verify and reuse their hashes without retraining base models.

## Conclusion

The hypothesis that stronger monotonic top-weighting would expose additional
complementarity is not supported. The three models benefit from being placed
in a common within-user rank space, but the best shape inside that space is
the existing linear percentile. More top-heavy curves consistently sacrifice
GAUC without improving nDCG@5 enough to compensate; the only flatter curve is
a metric trade that is fractionally negative overall.

**Final: REJECT. Keep YIXI7 as the current best; make no calibration change.**

# iterYIXI7 — ranking-objective alignment

## Final verdict: PROMOTE

Ranking-loss-specific tuning produces a confirmed new best ensemble:

```text
24% unchanged FM
42% tuned post-6f LightGBM
34% unchanged YIXI5 XGBoost
within-user average-tie percentile normalization

valid primary = 0.68415534
test primary  = 0.67437255
```

The promotion reference is the strongest post-6f ensemble, which remains the
YIXI5 24/40/36 percentile blend at **0.68237525 valid** and **0.66948622
test**. The new validation gain is **+0.00178009**. It clears the +0.001
criterion on every paired seed (mean +0.00209657, minimum +0.00178009).

The gain comes from LightGBM objective alignment, not XGBoost pair tuning:

| branch | standalone reference | selected | valid delta | decision |
|---|---:|---:|---:|---|
| LightGBM ranking objective | 0.67311722 | **0.67689133** | **+0.00377411** | confirmed |
| XGBoost pair generation | **0.66976142** | 0.66976142 | +0.00000000 | keep reference |
| final percentile ensemble | 0.68237525 | **0.68415534** | **+0.00178009** | **PROMOTE** |

The selected LightGBM change improves both parts of the official objective,
and the final ensemble does too. It is therefore not an nDCG-only trade:

| final valid metric | YIXI5 reference | YIXI7 | delta |
|---|---:|---:|---:|
| GAUC | 0.77264076 | **0.77445179** | **+0.00181103** |
| nDCG@5 | 0.59210974 | **0.59385890** | **+0.00174916** |
| primary | 0.68237525 | **0.68415534** | **+0.00178009** |

## File and condition separation

Following iter44's separate-runner structure, each condition has its own
entry point:

- `inspect_support.py`: installed-library support and effective-default check;
- `phase_a_lgb_objective.py`: LightGBM objective/truncation/sigmoid/norm only;
- `phase_b_xgb_pairs.py`: XGBoost pair method/count/normalization only;
- `blend.py`: eligible confirmed branches, percentile weight calibration,
  paired ensemble verification, and the sole final test stage;
- `diagnose_artifacts.py`: post-selection constant/random tie-floor and
  no-confound checks, with no role in model selection;
- `common.py`: fixed post-6f representations, fixed ordinary tree settings,
  fitting helpers, thresholds, and assertions.

Machine-readable results are in `support_results.json`,
`phase_a_results.json`, `phase_b_results.json`, `blend_results.json`, and
`artifact_results.json`.

No existing experiment, submission, ledger, summary, or shared source file
was modified. All new work is contained in
`experiments/iterYIXI7_ranking_objective_tuning/`.

## Harness-fidelity gates

Before new experiment code was written, the unmodified repository harness was
run:

```text
python3 make_submission.py /tmp/yixi6g_submission_check.csv
```

The repository's current `make_submission.py` now points to the later iter63
path rather than the historical iter44 output printed in the original
instruction text. It completed successfully and reproduced its current
published scores:

```text
LightGBM standalone: valid=0.67168 test=0.65353
FM standalone:       valid=0.63988 test=0.64187
iter63 blend:        valid=0.67606 test=0.65955
submit.py format/alignment check: PASSED (170588 rows)
```

Each new runner then performed the more relevant exact post-6f fidelity
assertion before its sweep:

| gate | published score | YIXI7 reproduction | tolerance |
|---|---:|---:|---:|
| post-6f LightGBM B1 standalone | 0.67311722 | 0.67311722 | `<1e-8` |
| post-6f XGBoost A1 standalone | 0.66976142 | 0.66976142 | `<1e-8` |
| current ensemble LightGBM | 0.67167872 | 0.67167872 | `<1e-8` |
| current ensemble XGBoost | 0.66755420 | 0.66755420 | `<1e-8` |
| unchanged FM five-seed ensemble | 0.63987792 | 0.63987792 | `<1e-8` |
| YIXI5 24/40/36 percentile blend | **0.68237525** | **0.68237525** | `<1e-8` |

No phase sweep accessed test predictions.

## Fixed representations and causal provenance

No feature was added, removed, or redesigned inside either objective sweep.
The exact post-6f standalone winners were used:

- LightGBM: 5-day user rate/activity replacement and iter63
  `decay_tab_rate_3`;
- XGBoost: 5-day user rate/activity replacement and iter63
  `decay_tab_rate_3`.

Both runners load YIXI6's already-verified unified native frame. Its
historical features were independently checked against strictly-earlier-date
direct sums, with maximum absolute error **7.11e-15**. Reusing this frame
holds the representation constant and introduces no new causal feature that
would require a new definition or leakage path.

For the final ensemble comparison, the unchanged YIXI5 component definitions
remain the valid promotion reference:

- LightGBM B0: 2.5-day user pair plus tab rate;
- XGBoost A0: 5-day user pair plus the older tab count;
- unchanged five-seed FM.

This distinction is intentional. YIXI6's B1/A1 standalones were stronger, but
using both in the final blend reduced validation and was rejected. Phase A/B
therefore tune the strongest standalone post-6f models, while final promotion
is measured against and reconstructed from the strongest actual ensemble.

## Installed ranking-parameter support

`inspect_support.py` used successful native ranking fits on a tiny grouped
dataset and inspected the resulting effective model configurations.

```text
Python    3.13.6
LightGBM  4.6.0
XGBoost   3.4.1
```

LightGBM accepted and retained all requested controls:

- `objective=lambdarank`;
- `objective=rank_xendcg`;
- `lambdarank_truncation_level`;
- `sigmoid`;
- `lambdarank_norm`.

The explicit LightGBM reference values 30 / 1.0 / true matched the installed
defaults. XGBoost accepted `lambdarank_pair_method`,
`lambdarank_num_pair_per_sample`, and `lambdarank_normalization`. Inspecting
the saved booster config revealed that this version's implicit reference is:

```text
lambdarank_pair_method           = topk
lambdarank_num_pair_per_sample   = 4294967295  (effectively unrestricted)
lambdarank_normalization         = true
```

The experiment compared candidates against those effective defaults rather
than assuming a default from another XGBoost release.

## Validation-only selection policy

- Official validation primary alone selected every axis winner.
- A change needed at least +0.0003 over the carried configuration to enter
  the next sequential phase.
- There was no Cartesian product. LightGBM's lambdarank controls were carried
  sequentially, while `rank_xendcg` was one independent objective branch.
- A standalone gain of at least +0.001 triggered paired five-seed
  confirmation. Confirmation required both mean and minimum paired delta to
  clear +0.001.
- Only confirmed standalone branches were eligible for the final ensemble.
- Percentile normalization was fixed. Weights used the established 0.10
  simplex followed by a local 0.02 refinement on validation.
- Final weights were frozen before paired ensemble confirmation.
- Test was evaluated once, after selection, diagnostics, confirmation, and
  the `PROMOTE` verdict had been written.

## Phase A — LightGBM ranking objective

All ordinary architecture and feature settings were held fixed:

```text
linear_tree=True, num_leaves=2
learning_rate=0.10, n_estimators=500
min_child_samples=200, reg_lambda=1
early_stopping_rounds=30
post-6f B1 features: 5-day user pair + decay_tab_rate_3
```

### A1. Truncation level

The installed implicit reference is truncation 30.

| truncation | best iteration | GAUC | nDCG@5 | primary | delta vs 30 |
|---:|---:|---:|---:|---:|---:|
| 30 reference | 47 | 0.76094317 | 0.58529127 | 0.67311722 | — |
| 5 | 3 | 0.72608042 | 0.57295555 | 0.64951801 | -0.02359921 |
| 10 | 50 | 0.76074058 | 0.58571184 | 0.67322624 | +0.00010902 |
| 20 | 50 | **0.76166028** | 0.58571696 | 0.67368865 | +0.00057143 |
| **50** | **49** | 0.76155031 | **0.58583456** | **0.67369246** | **+0.00057524** |

Truncation 50 narrowly wins and clears the +0.0003 carry gate. Values 20 and
50 form a near-flat positive plateau; the result is not a single isolated
grid spike. Truncation 5 is far too top-heavy and harms both metrics.

### A2. Sigmoid from truncation 50

| sigmoid | best iteration | GAUC | nDCG@5 | primary | delta vs carried 1.0 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 6 | 0.72055578 | 0.56806594 | 0.64431083 | -0.02938163 |
| 1.0 | 49 | 0.76155031 | 0.58583456 | 0.67369246 | — |
| **2.0** | **41** | **0.76465243** | **0.58913028** | **0.67689133** | **+0.00319886** |

`sigmoid=2.0` is the main discovery. Relative to the original post-6f
LightGBM reference, it participates in a total gain of +0.00377411:

```text
GAUC delta     = +0.00370926
nDCG@5 delta   = +0.00383902
primary delta  = +0.00377411
```

The objective is not merely becoming more top-heavy: whole-user GAUC and
top-five nDCG improve by similar amounts.

### A3. Lambdarank normalization

From truncation 50 / sigmoid 2.0:

| normalization | best iteration | GAUC | nDCG@5 | primary | carry? |
|---|---:|---:|---:|---:|---|
| **true** | **41** | **0.76465243** | **0.58913028** | **0.67689133** | keep |
| false | 110 | 0.68717664 | 0.54584283 | 0.61650974 | reject |

Disabling normalization damages both metrics severely. The installed default
`true` is retained.

### A4. Alternative `rank_xendcg` branch

`rank_xendcg` is supported but scores only **0.64573598**
(GAUC 0.72316867, nDCG@5 0.56830335). It is -0.02738124 versus the original
post-6f reference and -0.03115535 versus the carried lambdarank winner, so it
is rejected without further tuning.

### Phase A paired confirmation

The selected rank configuration is:

```text
objective=lambdarank
lambdarank_truncation_level=50
sigmoid=2.0
lambdarank_norm=true (unchanged default)
```

| seed | post-6f reference | selected candidate | paired delta |
|---:|---:|---:|---:|
| 0 | 0.67311722 | 0.67689133 | +0.00377411 |
| 1 | 0.67416626 | 0.67686027 | +0.00269401 |
| 2 | 0.67415625 | 0.67637551 | +0.00221926 |
| 3 | 0.67415625 | 0.67675328 | +0.00259703 |
| 4 | 0.67415625 | 0.67688465 | +0.00272840 |

Mean delta is **+0.00280256**, standard deviation 0.00051839, and minimum
delta **+0.00221926**. Every seed clears +0.001.

**Phase A: confirmed positive.**

## Phase B — XGBoost pair generation

Every ordinary tree setting and the post-6f A1 representation were fixed:

```text
rank:ndcg, max_depth=1
learning_rate=0.025, n_estimators=1000
early_stopping_rounds=120
min_child_weight=1, gamma=0, lambda=1, alpha=0
subsample=1, colsample_bytree=1
post-6f A1 features: 5-day user pair + decay_tab_rate_3
```

The installed implicit `topk` / unrestricted-pair / normalized reference
reproduced **0.66976142** exactly.

| change from carried reference | best iteration | GAUC | nDCG@5 | primary | delta |
|---|---:|---:|---:|---:|---:|
| implicit reference | 971 | **0.75475472** | **0.58476818** | **0.66976142** | — |
| pair method `mean` | 926 | 0.73846310 | 0.57750386 | 0.65798348 | -0.01177794 |
| `topk`, 5 pairs | 973 | 0.74340022 | 0.58020979 | 0.66180503 | -0.00795639 |
| `topk`, 10 pairs | 995 | 0.75110090 | 0.58207893 | 0.66658992 | -0.00317150 |
| `topk`, 20 pairs | 999 | 0.75414950 | 0.58430457 | 0.66922700 | -0.00053442 |
| normalization false | 831 | 0.66460013 | 0.54421705 | 0.60440862 | -0.06535280 |

No XGBoost change clears the +0.0003 preliminary gate. More pairs improve
smoothly toward the unrestricted reference, while `mean` and disabled
normalization hurt both GAUC and nDCG. There is no evidence of a trade that
could help the official primary or ensemble.

**Phase B: keep the implicit reference; no confirmed branch.** No five-seed
confirmation is warranted for a zero/negative selected delta.

## Final percentile ensemble

Only Phase A is eligible. The runner reconstructs the current YIXI5 system,
then replaces its LightGBM component with the confirmed post-6f B1
representation and selected ranking configuration. Because Phase B rejected
all changes, the final XGBoost remains the unchanged YIXI5 A0 ensemble
component. This avoids silently promoting YIXI6's final rejected A1 transfer.

At the old 24/40/36 weights, the new LightGBM already improves the ensemble:

| system | weights FM/LGB/XGB | valid | delta vs YIXI5 |
|---|---:|---:|---:|
| YIXI5 reference | 0.24 / 0.40 / 0.36 | 0.68237525 | — |
| tuned LightGBM, old weights | 0.24 / 0.40 / 0.36 | 0.68404955 | +0.00167429 |
| **tuned LightGBM, locally calibrated** | **0.24 / 0.42 / 0.34** | **0.68415534** | **+0.00178009** |

The local weight adjustment adds only +0.00010580 over the fixed old weights;
the result is driven by model improvement, not an isolated weight-search
point. Nearby weights are also strong: 0.24/0.40/0.36 scores 0.68404955 and
0.26/0.40/0.34 scores 0.68403143.

For a same-feature attribution, YIXI6's untuned B1 LightGBM-only substitution
scored 0.68219340 in the ensemble. With B1 held fixed and ranking parameters
tuned, the old-weight score becomes 0.68404955 (**+0.00185615**). Thus the
new ensemble gain is specifically attributable to ranking-objective tuning,
not merely reintroducing the previously rejected feature transfer.

### Final paired confirmation

Weights and normalization were frozen at 0.24/0.42/0.34 before these refits.
The unchanged full-row/full-column XGBoost component is deterministic across
`random_state`; the paired seeds vary the affected LightGBM while retaining
the same five-seed FM ensemble and unchanged XGBoost scores.

| seed | YIXI5 reference | YIXI7 candidate | paired delta |
|---:|---:|---:|---:|
| 0 | 0.68237525 | 0.68415534 | +0.00178009 |
| 1 | 0.68190324 | 0.68425941 | +0.00235617 |
| 2 | 0.68209529 | 0.68395257 | +0.00185728 |
| 3 | 0.68209529 | 0.68418711 | +0.00209183 |
| 4 | 0.68190324 | 0.68430072 | +0.00239748 |

Mean paired delta is **+0.00209657**, standard deviation 0.00025115, and
minimum **+0.00178009**. Every seed clears the promotion threshold.

## Accuracy and complementarity

The tuned LightGBM's pooled within-user-percentile correlation with the
unchanged XGBoost rises:

| tree pair | percentile-score correlation |
|---|---:|
| current YIXI5 LightGBM vs XGBoost | 0.78685857 |
| tuned post-6f LightGBM vs unchanged XGBoost | 0.83074524 |
| change | **+0.04388667** |

This means complementarity is reduced: the two trees rank more similarly.
Unlike YIXI6, however, the LightGBM accuracy gain is large enough to dominate
that diversity loss. The best blend shifts only two points of weight from
XGBoost to LightGBM, and both final GAUC and nDCG improve. The promotion is
therefore primarily an accuracy gain with a modest adverse diversity effect,
not a diversity gain.

## Section 3 artifact and confound checks

The unusually strong Phase A result received a post-selection diagnostic
pass. This runner used validation only and could not alter the frozen
candidate or verdict.

| check | reference/result | candidate/result | conclusion |
|---|---:|---:|---|
| mean per-user unique-score fraction | 0.98641379 | 0.98640278 | unchanged, not tie-driven |
| overall distinct LGB scores | 122,611 | 122,609 | unchanged |
| all-constant primary | 0.48367125 | — | expected trivial floor |
| seeded-random primary | 0.48288769 | — | expected trivial floor |
| validation score values finite | yes | yes | passed |
| exact post-6f component harness | 0.67311722 | 0.67311722 | passed |
| exact final ensemble harness | 0.68237525 | 0.68237525 | passed |

The feature list is exactly the same 11-column post-6f B1 representation in
reference and candidate. No feature or historical computation changed, all
ordinary tree parameters remained identical, and the reused causal metadata
passed. Both official metrics improve, the truncation result has a 20/50
plateau, nearby blend weights remain positive, and both standalone and final
gains hold across five seeds.

**Artifact/confound verdict: passed.** The gain is not inherited row order,
a heavily tied model, an accidentally added feature, a data-path change, or a
lucky seed.

## Frozen one-time test evaluation

`blend.py` first predicted test after validation selection, weight freezing,
paired confirmation, diagnostics, and the `PROMOTE` verdict were recorded.

| model/system | reference test | selected test | delta |
|---|---:|---:|---:|
| LightGBM, same B1 features (untuned → tuned) | 0.66269004 | **0.66578388** | **+0.00309384** |
| final percentile ensemble | 0.66948622 | **0.67437255** | **+0.00488633** |

Final test metric decomposition:

| test metric | YIXI5 reference | YIXI7 | delta |
|---|---:|---:|---:|
| GAUC | 0.74475724 | **0.74980152** | **+0.00504428** |
| nDCG@5 | 0.59421521 | **0.59894353** | **+0.00472832** |
| primary | 0.66948622 | **0.67437255** | **+0.00488633** |

These test results support the validation-selected choice but did not select
the configuration, weights, or verdict.

## Reproduction

Run from the repository root in this order:

```text
python3 experiments/iterYIXI7_ranking_objective_tuning/inspect_support.py
python3 experiments/iterYIXI7_ranking_objective_tuning/phase_a_lgb_objective.py
python3 experiments/iterYIXI7_ranking_objective_tuning/phase_b_xgb_pairs.py
python3 experiments/iterYIXI7_ranking_objective_tuning/blend.py
python3 experiments/iterYIXI7_ranking_objective_tuning/diagnose_artifacts.py
```

The first four commands are the selection/verification pipeline. The final
command is explicitly post-selection validation-only diagnosis.

## Promoted configuration

Promote the following system as the new validation-selected best:

```text
FM:
  unchanged iter38/YIXI5 five-seed component

LightGBM:
  post-6f B1 representation
  5-day user rate/activity replacement
  decay_tab_rate_3
  objective=lambdarank
  lambdarank_truncation_level=50
  sigmoid=2.0
  lambdarank_norm=true
  linear_tree=True
  num_leaves=2
  learning_rate=0.10
  n_estimators=500
  min_child_samples=200
  reg_lambda=1.0
  early_stopping_rounds=30

XGBoost:
  unchanged YIXI5 final-ensemble A0 component
  implicit topk/unrestricted-pair/normalized rank:ndcg objective

Blend:
  within-user average-tie percentile normalization
  FM=0.24, LightGBM=0.42, XGBoost=0.34
```

**Final: PROMOTE — valid 0.68415534, test 0.67437255.**

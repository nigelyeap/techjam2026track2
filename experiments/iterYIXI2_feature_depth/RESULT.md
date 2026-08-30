# iterYIXI2 — feature depth on the native LightGBM encoding

## Hypothesis

iter44's native LightGBM reused feature choices originally optimized for a
bucketed FM. This experiment tested whether a stump-based native GBM could use
additional continuous history detail that the FM could not exploit:

1. user decay-rate/activity half-lives at 1, 5, 7, and 14 days, both as
   replacements for 2.5 days and as parallel features;
2. user-tab positive-decay half-lives at 1, 5, 7, and 14 days, both as
   replacements for 3 days and in parallel;
3. 2.5-day causal author- and video-popularity decay (rate and activity);
4. three predeclared crosses useful to a `num_leaves=2` additive model:
   `decay_rate / duration`, `decay_rate * log1p(decay_activity)`, and a
   train-fitted user-activity-tier × tab categorical interaction.

## Harness-fidelity gates

Before any 6b code was created, the unmodified `make_submission.py` was run
end-to-end and reproduced the required references exactly:

```text
GBM standalone: valid=0.66135 test=0.64794
FM ensemble standalone: valid=0.63988 test=0.64187
iter44 blend: valid primary=0.66473 test primary=0.65197
submit.py format/alignment check: PASSED (170588 rows)
```

The new 6b runner then trained its own unmodified 11-column iter44 baseline
before evaluating any new feature. It reproduced the full validation metric:

```text
GAUC=0.74500167, nDCG@5=0.57770807, primary=0.66135490
best_iteration=332
```

This confirms both the environment and the new feature-selection harness are
faithful to iter44.

## Causal implementation and independent verification

`features.py` imports iter44's native encoder and iter27's existing causal
traversals. Expanded user and user-tab half-lives are computed using the same
two-phase date-grouped mechanism as iter27. Author/video features use a new
generalized keyed traversal whose only change is the state key
(`author_id`/`video_id` instead of `user_id`). For every date, all feature
values are read before that date's labels update state. Consequently, a row
can see prior dates only: same-date and future labels are excluded.

Before model fitting, an independent direct-sum implementation recomputed
three validation rows for each new history family by explicitly summing
`0.5 ** (gap_days / halflife)` over matching rows with strictly earlier dates.

| family | checked half-lives | maximum absolute error |
|---|---|---:|
| user decay rate/activity | 1, 5, 7, 14 | 1.78e-14 |
| user-tab positive decay | 1, 5, 7, 14 | 5.33e-15 |
| author decay rate/activity | 2.5 | 1.48e-12 |
| video decay rate/activity | 2.5 | 1.48e-12 |

Across those checks, 626 matching same-date rows were present and explicitly
excluded. Every comparison passed below the `1e-10` tolerance. Row labels and
user IDs were also asserted to match iter44 for train, validation, and test
before the feature arrays were joined. **No leakage or row misalignment was
detected.**

## Selection protocol

The model configuration was frozen to iter44's winner:

```text
LGBMRanker(objective='lambdarank', num_leaves=2,
           learning_rate=0.05, n_estimators=500,
           min_child_samples=200, reg_lambda=1.0,
           early_stopping_rounds=30)
```

Every initial feature was evaluated with seed 0 using official validation
primary only. Per the instructions, only a family gaining at least `+0.0003`
could enter a follow-up combination. A selected candidate had to improve by
at least `+0.001` before a five-seed confirmation was run. The confirmation
refit both baseline and candidate at each seed and used paired deltas.

Test scores were not computed until all feature selection, threshold gating,
causality/tie checks, and the paired five-seed validation check were complete.
The frozen selected candidate was evaluated on test once.

## Stage 1 — additional user half-lives

Each candidate adds both the smoothed rate and decayed activity at its stated
half-life.

| half-life | replacement valid | delta | parallel valid | delta |
|---:|---:|---:|---:|---:|
| 1 day | 0.65127 | -0.01008 | 0.66135 | +0.00000 |
| 5 days | 0.65711 | -0.00425 | 0.65711 | -0.00425 |
| 7 days | 0.65440 | -0.00695 | 0.65440 | -0.00695 |
| 14 days | 0.65151 | -0.00984 | 0.65151 | -0.00984 |

This is a clean rejection of wider user half-life depth. In train, the new
rates correlate with `decay_rate_2.5` at 0.981 (1d), 0.997 (5d), 0.996 (7d),
and 0.993 (14d). The 1-day parallel feature receives zero gain and exactly
reproduces baseline. At 5/7/14 days, the new rate displaces the stronger
2.5-day rate entirely and makes the model worse; all new activity features
receive zero gain. No candidate reached `+0.0003`, so the protocol correctly
forbade a multi-half-life follow-up union.

## Stage 2 — additional user-tab half-lives

| half-life | replacement valid | delta | parallel valid | delta |
|---:|---:|---:|---:|---:|
| 1 day | 0.66039 | -0.00096 | 0.66135 | +0.00000 |
| 5 days | 0.66053 | -0.00082 | 0.66053 | -0.00082 |
| 7 days | 0.66015 | -0.00121 | 0.66015 | -0.00121 |
| 14 days | 0.66056 | -0.00079 | 0.66056 | -0.00079 |

The established 3-day tab decay remains best. The parallel 1-day feature is
effectively ignored (0.17% of total gain) and reproduces baseline; longer
horizons take meaningful split gain from the 3-day signal but reduce the
official metric. Again, no result passed the `+0.0003` follow-up threshold,
so no tab-half-life union was run.

## Stage 3 — author/video popularity decay

Author/video `rate` uses the same Laplace form as the user rate,
`(decayed_pos + 0.5) / (decayed_total + 1)`. `activity` is the decayed total
impression count.

| added feature(s) | valid | delta |
|---|---:|---:|
| author rate | 0.64426 | -0.01710 |
| author activity | 0.66135 | +0.00000 |
| author rate + activity | 0.64426 | -0.01710 |
| video rate | 0.64377 | -0.01758 |
| video activity | 0.66135 | +0.00000 |
| video rate + activity | 0.64377 | -0.01758 |

The activity counts are ignored completely and reproduce baseline. The rate
features are not merely unused: author/video rate attracts 17.3%/17.8% of
total tree gain and sharply damages both GAUC and nDCG. The likely diagnosis
is objective mismatch: global author/video engagement is an attractive
population-level split but conflicts with the task's within-user
personalization. With only one split per tree, spending substantial capacity
on global popularity displaces stronger user-specific history. Neither author
nor video passed the follow-up threshold, so they were not combined.

## Stage 4 — pairwise cross-features

| added cross | GAUC | nDCG@5 | primary | delta |
|---|---:|---:|---:|---:|
| `decay_rate / (duration_ms + 1)` | 0.74464 | 0.57754 | 0.66109 | -0.00027 |
| `decay_rate * log1p(decay_activity)` | **0.74682** | **0.58276** | **0.66479** | **+0.00343** |
| train activity tier × tab | 0.74497 | 0.57777 | 0.66137 | +0.00001 |

Only `decay_rate * log1p(decay_activity)` passed the `+0.0003` gate. It
receives 14.8% of seed-0 feature gain and supplies a confidence-weighted
version of the user rate that a decision-stump ensemble cannot reconstruct
from separate rate/activity splits. The other two crosses stayed below the
noise threshold. Since no other family passed `+0.0003`, the instructions did
not permit a cross-family combination.

## Seed-0 selected candidate and tie check

Validation-only selection therefore froze the single added cross
`decay_rate_2.5 * log1p(decay_act_2.5)`:

```text
baseline:  GAUC=0.74500167 nDCG@5=0.57770807 primary=0.66135490
candidate: GAUC=0.74681538 nDCG@5=0.58276159 primary=0.66478848
delta: +0.00343359
```

Because this appeared unusually strong, the tie diagnostic was run before
test access:

```text
constant scores: primary=0.48367
random scores:   primary=0.48266
candidate mean within-user unique-score fraction=0.8852
candidate unique scores overall=19,771 / 124,909 rows
```

The trivial tied baseline remains at the random floor, and the candidate has
substantial score diversity. Seed 0's gain is not a stable-sort row-order
artifact. The candidate's columns were also recorded explicitly: the exact 11
iter44 columns plus only this one cross, ruling out a silently-added confound.

## Required paired five-seed confirmation — FAILED

The seed-0 gain exceeded `+0.001`, so both the baseline and the frozen
candidate were fit at seeds 0 through 4:

| seed | baseline valid | candidate valid | paired delta |
|---:|---:|---:|---:|
| 0 | 0.66135 | 0.66479 | **+0.00343** |
| 1 | 0.66092 | 0.65838 | **-0.00254** |
| 2 | 0.66109 | 0.65812 | **-0.00297** |
| 3 | 0.65633 | 0.66506 | **+0.00872** |
| 4 | 0.66075 | 0.65865 | **-0.00210** |
| mean | 0.66009 | 0.66100 | **+0.00091** |

Paired-delta standard deviation is 0.00455; the minimum delta is -0.00297.
Only 2/5 seeds improve, and the mean gain is below the required +0.001. The
baseline seed values exactly match iter44's previously published five-seed
log, including its weak seed 3, so this is not drift in the new runner. The
most plausible diagnosis is that the cross changes which nearly tied stump
splits win under LightGBM's seeded categorical split search: it rescues the
unusually weak baseline seed 3 but damages three ordinary seeds. Whatever the
exact tie-breaking path, the measured outcome is instability rather than a
systematic feature gain.

## Frozen one-time test result

After the five-seed failure had already fixed the verdict, seed 0's selected
candidate was checked on test once for the required record:

| model | split | GAUC | nDCG@5 | primary |
|---|---|---:|---:|---:|
| iter44 baseline | valid | 0.74500 | 0.57771 | 0.66135 |
| selected cross | valid | 0.74682 | 0.58276 | 0.66479 |
| iter44 baseline | test | 0.72021 | 0.57568 | 0.64794 |
| selected cross | test | 0.71618 | 0.58082 | 0.64850 |

Test primary improves by only **+0.00056**, below promotion scale. Its
nDCG@5 gain (+0.00514) is offset by a GAUC regression (-0.00403), reinforcing
the diagnosis that the cross produces an unstable top-of-list tradeoff rather
than a robust ranking improvement. Test was not used to select or rescue it.

## Final verdict: REJECT

No Section 6b feature meets the requirement of a `>=0.001` validation gain
that holds across five seeds.

- Wider user and tab half-lives are redundant or worse than 2.5d/3d.
- Author/video activity is ignored; author/video rate consumes capacity and
  strongly harms within-user ranking.
- Duration and tier×tab crosses are null results.
- The rate×log-activity cross is promising at seed 0 but fails paired
  confirmation, with three negative seeds and mean delta only +0.00091.
- Its one-time test delta is only +0.00056 and cannot override the
  validation-only decision.

Therefore **REJECT all tested 6b additions and retain iter44's native
LightGBM feature set unchanged**. The rate×log-activity interaction is worth
remembering only as a diagnosed unstable idea, not as a promotable result.

## Artifacts

- `features.py`: causal feature construction and independent brute-force
  checks
- `run_experiment.py`: threshold-gated validation sweep, paired seed check,
  and frozen final test stage
- `results.json`: exact per-candidate metrics, gain importances, causality
  records, seed results, and final test metrics

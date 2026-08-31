# iterYIXI10 — Leakage-safe intrinsic video metadata

## Verdict: PROMOTE

Adding only the native categorical `upload_type` field to the promoted YIXI9
LightGBM produces a confirmed new best system. The final ensemble retains the
existing FM and XGBoost components unchanged:

| system | FM / LGB / XGB | valid GAUC | valid nDCG@5 | valid primary | test GAUC | test nDCG@5 | test primary |
|---|---:|---:|---:|---:|---:|---:|---:|
| YIXI9 reference | 0.18 / 0.46 / 0.36 | 0.78442186 | 0.59734720 | 0.69088453 | 0.75533485 | 0.60150570 | 0.67842031 |
| **YIXI10 `upload_type`** | **0.10 / 0.52 / 0.38** | **0.79415059** | **0.60471827** | **0.69943440** | **0.76157826** | **0.60706699** | **0.68432260** |
| delta | — | **+0.00972873** | **+0.00737107** | **+0.00854987** | **+0.00624341** | **+0.00556129** | **+0.00590229** |

The branch and weights were chosen exclusively on validation. Test prediction
and evaluation occurred once, after the branch, weights, five-seed
confirmation, diagnostics, and `PROMOTE` verdict had been frozen.

## Harness fidelity

The authoritative instructions were reread completely. Before new experiment
code was written, the mandatory end-to-end harness passed:

```text
LightGBM standalone       valid 0.67168 / test 0.65353
FM ensemble standalone    valid 0.63988 / test 0.64187
iter63 blend              valid 0.67606 / test 0.65955
submission alignment      PASSED (170588 rows)
```

The new `harness.py` then reproduced the exact YIXI9 references at `1e-8`
tolerance without evaluating test:

| component/system | valid primary |
|---|---:|
| promoted YIXI9 LightGBM | 0.67822242 |
| final-ensemble XGBoost A0 | 0.66755420 |
| YIXI9 three-model percentile blend | 0.69088453 |

## Leakage and metadata audit

Only `video_features_basic_pure.csv` is read by this experiment. The file has
7,583 rows and 7,583 distinct `video_id` values. Its many-to-one join covers
all 1,436,609 impressions with zero unmatched rows.

`video_features_statistic_pure.csv` is never opened or joined. No aggregate
field such as `play_cnt`, `like_cnt`, `complete_play_cnt`, `show_cnt`, play
duration, comment count, follow count, or share count reaches a feature frame.
The allowed source hash and its exact columns are recorded in
`feature_results.json`.

All metadata values are static within video. Native categorical vocabularies
are derived from train only; future unseen values map to missing. The selected
`upload_type` has 14 training categories, no missing values, and no unseen
validation or test rows.

### Video age

`video_age_days` is computed independently for every impression as:

```text
parse(interaction_date) - parse(upload_dt)
```

The join contains no impression before upload and no missing upload dates.
Observed age ranges are:

| split | min | median | max | distinct days |
|---|---:|---:|---:|---:|
| train | 0 | 2 | 12 | 13 |
| valid | 11 | 14 | 19 | 9 |
| test | 18 | 23 | 29 | 12 |

Five row-level date-subtraction spot checks per split are recorded. Although
age is leakage-safe and available at impression time, both models assign it
zero gain under their fixed shallow architectures.

### Other metadata

- `video_type`, `upload_type`, `music_type`, and the complete `tag` string are
  passed as train-vocabulary native categoricals.
- `aspect_ratio = server_width / server_height` is raw continuous. All heights
  are positive and all 1,436,609 ratios are finite, ranging from 0.4225 to
  2.3392.
- `music_id` is kept separate. It has 7,160 training levels; 17 validation and
  13 test rows contain unseen IDs and are mapped to categorical missing.

## LightGBM independent tests

Each row below adds exactly one predeclared metadata group to the current
YIXI9 LightGBM. The model remains fixed at `linear_tree=True`, two leaves,
learning rate 0.10, rank-specific truncation 50 and sigmoid 2, with the
existing 5-day watch-depth feature.

| group | valid GAUC | valid nDCG@5 | valid primary | delta | group gain fraction | best iteration |
|---|---:|---:|---:|---:|---:|---:|
| reference | 0.76751000 | 0.58893478 | 0.67822242 | — | — | 40 |
| video age | 0.76751000 | 0.58893478 | 0.67822242 | 0 | 0% | 40 |
| video type | 0.76751000 | 0.58893478 | 0.67822242 | 0 | 0% | 40 |
| **upload type** | **0.77871114** | **0.59797174** | **0.68834144** | **+0.01011902** | **8.02%** | 16 |
| music type | 0.76751000 | 0.58893478 | 0.67822242 | 0 | 0% | 40 |
| tag | 0.76796377 | 0.59272444 | 0.68034410 | +0.00212169 | 11.54% | 16 |
| aspect ratio | 0.76751000 | 0.58893478 | 0.67822242 | 0 | 0% | 40 |
| music ID, high cardinality | 0.76890606 | 0.58985639 | 0.67938125 | +0.00115883 | 3.72% | 45 |

`tag` and `music_id` have preliminary positive evidence, but they are not
claimed as confirmed findings because validation selected `upload_type` and
their compositions failed the incremental carry rule:

| forward composition | valid primary | delta vs reference | incremental vs `upload_type` |
|---|---:|---:|---:|
| upload type + tag | 0.68034410 | +0.00212169 | -0.00799733 |
| upload type + music ID | 0.68842888 | +0.01020646 | +0.00008744 |

The latter numerical increase is below the project's `0.0003` noise gate, so
the simpler `upload_type` representation is retained.

### Standalone LightGBM confirmation

Paired candidate-minus-reference primary deltas are:

```text
seed 0  +0.01011902
seed 1  +0.00932813
seed 2  +0.00925070
seed 3  +0.00898629
seed 4  +0.01003724
mean    +0.00954428
```

On the single post-selection test evaluation, standalone LightGBM improves
from `0.66622883` to `0.67341733` (`+0.00718850`). Both test GAUC and nDCG@5
improve by approximately `+0.0072`.

## XGBoost independent tests

The final-ensemble XGBoost A0 representation and tuned YIXI5 hyperparameters
remain fixed for every comparison.

| group | valid GAUC | valid nDCG@5 | valid primary | delta | group gain fraction |
|---|---:|---:|---:|---:|---:|
| reference | 0.75199854 | 0.58310986 | 0.66755420 | — | — |
| video age | 0.75199854 | 0.58310986 | 0.66755420 | 0 | 0% |
| video type | 0.75199854 | 0.58310986 | 0.66755420 | 0 | 0% |
| upload type | 0.75068694 | 0.58252096 | 0.66660392 | -0.00095028 | 12.64% |
| music type | 0.75199854 | 0.58310986 | 0.66755420 | 0 | 0% |
| tag | 0.72187620 | 0.56783950 | 0.64485788 | -0.02269632 | 17.09% |
| aspect ratio | 0.75199854 | 0.58310986 | 0.66755420 | 0 | 0% |
| music ID, high cardinality | 0.75220525 | 0.58319265 | 0.66769898 | +0.00014478 | 5.94% |

No XGBoost group reaches the `+0.0003` preliminary threshold, so no XGBoost
composition, confirmation, ensemble replacement, or test evaluation is
allowed. This is another model-family-dependent feature result: the strongest
LightGBM metadata field is slightly harmful to XGBoost.

## Final ensemble selection

Only the confirmed LightGBM branch enters the current within-user-percentile
ensemble. At the unchanged YIXI9 weights it already scores `0.69862133`, a
`+0.00773680` gain. The predeclared coarse simplex and local 0.02 refinement
select `0.10 FM / 0.52 LGB / 0.38 XGB` at `0.69943440`.

The weight result is a broad plateau, not a grid spike. Thirteen three-model
points lie within `0.0003` of the winner:

```text
FM/LGB/XGB       valid primary      distance from best
0.10/0.52/0.38   0.69943440         0
0.12/0.54/0.34   0.69941843        -0.00001597
0.10/0.54/0.36   0.69940305        -0.00003135
0.08/0.56/0.36   0.69937325        -0.00006115
0.10/0.56/0.34   0.69937146        -0.00006294
```

### Paired final-ensemble confirmation

The branch and selected weights were frozen at seed 0, then compared against
the fixed YIXI9 reference across the remaining seeds:

| seed | reference valid | candidate valid | delta |
|---:|---:|---:|---:|
| 0 | 0.69088453 | 0.69943440 | +0.00854987 |
| 1 | 0.69095004 | 0.69839817 | +0.00744814 |
| 2 | 0.69096386 | 0.69833934 | +0.00737548 |
| 3 | 0.69081426 | 0.69810897 | +0.00729471 |
| 4 | 0.69083291 | 0.69962680 | +0.00879389 |

Mean delta is `+0.00789242`; the minimum is `+0.00729471`.

## Large-gain verification

### Tie and stable-sort artifact check

| scores | overall unique values | mean within-user unique fraction |
|---|---:|---:|
| reference LightGBM | 122,030 / 124,909 | 98.07% |
| candidate LightGBM | 120,557 / 124,909 | 97.51% |
| reference ensemble | 14,267 / 124,909 | 99.94% |
| candidate ensemble | 18,926 / 124,909 | 99.90% |

The all-constant baseline is `0.48367125` primary and uniform-random seed 0 is
`0.48265904`, so original row order is at the expected trivial floor. The
candidate is not heavily tied, and both final GAUC and nDCG@5 improve.

### Source and confound checks

- Reference and candidate LightGBM columns differ by exactly
  `meta_upload_type`; nothing is removed.
- Training rows, labels, all other features, and all LightGBM parameters are
  identical.
- FM and XGBoost predictions are unchanged.
- The only metadata source is the basic table. The statistic table is not read
  and prohibited aggregates are absent from all frames.
- `upload_type` is static per video, known at upload, complete in all splits,
  and uses only train-derived categories.
- The feature accounts for 8.02% of LightGBM split gain, confirming it is
  actively used.
- The gain survives before weight tuning (`+0.00773680` at old weights).
- Train and validation show consistent coarse engagement ordering: for
  example, `Web` is 0.3736/0.3449 long-view rate, `LongImport` 0.3476/0.3234,
  `ShortImport` 0.2927/0.2762, and `PictureSet` 0.0644/0.0718. This descriptive
  check was run only after selection was frozen.

## Interpretation

The low-cardinality upload mechanism exposes broad content-format differences
that the shallow LightGBM cannot efficiently recover from high-cardinality
`video_id` and `author_id`. Its train/validation category behavior is stable,
and it improves both whole-user GAUC and top-of-list nDCG on validation and
test. The same categorical split behavior does not transfer to XGBoost.

The promoted change is therefore deliberately narrow:

```text
LightGBM: add native categorical upload_type
XGBoost:  unchanged YIXI9 ensemble A0
FM:       unchanged five-seed ensemble
Blend:    within-user percentile, 0.10 FM / 0.52 LGB / 0.38 XGB
```

## Reproduction

Run from the repository root:

```bash
python3 experiments/iterYIXI10_video_metadata/inspect_features.py
python3 experiments/iterYIXI10_video_metadata/harness.py
python3 experiments/iterYIXI10_video_metadata/phase_lgb.py
python3 experiments/iterYIXI10_video_metadata/phase_xgb.py
python3 experiments/iterYIXI10_video_metadata/ensemble.py
python3 experiments/iterYIXI10_video_metadata/diagnose_artifacts.py
```

The runners are split by condition in the iter44 style. Machine-readable
outputs are `feature_results.json`, `harness_results.json`,
`phase_lgb_results.json`, `phase_xgb_results.json`, `ensemble_results.json`,
and `artifact_results.json`.

All code and outputs are confined to `experiments/iterYIXI10_video_metadata/`.
No prohibited shared file was modified.

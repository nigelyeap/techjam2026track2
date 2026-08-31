# iterYIXI3 — lightweight attention-pooling feature for native XGBoost

## Final verdict: REJECT

A frozen, scaled-dot-product attention pool over the user's last 20 or 40
interactions does **not** improve the strongest currently verified native
XGBoost component. Against YIXI4's confirmed 5-day-decay XGBoost reference:

| configuration | valid primary | delta vs h5 | test primary |
|---|---:|---:|---:|
| YIXI4 h5 XGBoost reference | **0.66649055** | — | **0.65197098** |
| h5 + attention, K=20 | 0.61597961 | -0.05051094 | not selected |
| h5 + attention, K=40 | 0.65841424 | -0.00807631 | 0.64911228 |

Neither attention window reached the required `+0.0003` preliminary gate.
The protocol therefore prohibited a multi-window union, five-seed
confirmation, or a three-model attention blend. The K=40 test number was
computed once only after validation had frozen it as the best attention
candidate; its test delta versus the published h5 reference is -0.00285870.

The project-level reference remains YIXI1's promoted 16% FM / 8% LightGBM /
76% XGBoost blend: **0.67006499 valid / 0.65645623 test**. Attention does not
supplant that score and does not qualify for a blend attempt.

## Hypothesis

Prior attention experiments tested either a bucketed FM feature or a full
end-to-end attention model. Section 6c posed a narrower question: can a small,
precomputed summary of the raw user sequence become a useful continuous split
candidate for a native tree ranker, even though attention did not succeed as a
standalone replacement architecture?

The experiment reused iter32's strongest lightweight definition rather than
building another full sequence model:

1. fit a small k=8 user×video FM on the training split only for eight epochs;
2. freeze its video embeddings;
3. for each row, retrieve only that user's interactions strictly earlier in
   `(time_ms, orig_idx)` order;
4. compute scaled dot products between the current video and the last K
   historical videos;
5. softmax the similarities and use them to pool the historical `long_view`
   labels into one scalar `attn_rate_K`.

K=20 and K=40 were predeclared, matching the range requested in Section 6c.
The feature was passed as a raw float to XGBoost; it was not bucketed and no
attention model was trained jointly with the tree ranker.

## Harness-fidelity gates

Before creating any 6c code, the required end-to-end command was run with its
output redirected outside the repository:

```text
python3 make_submission.py /tmp/yixi6c_submission_check.csv

GBM standalone: valid=0.66135 test=0.64794
FM ensemble standalone: valid=0.63988 test=0.64187
iter44 blend: valid primary=0.66473  test primary=0.65197
submit.py format/alignment check: PASSED (170588 rows)
```

The new runner then reconstructed YIXI4's exact h5 feature and fixed 6a
XGBoost model before adding attention. It reproduced the complete reference:

| metric | YIXI4 published | YIXI3 reproduced |
|---|---:|---:|
| GAUC | 0.7504962087 | 0.7504962087 |
| nDCG@5 | 0.5824848413 | 0.5824848413 |
| primary | **0.6664905548** | **0.6664905548** |
| best iteration | 340 | 340 |

Both the repository-wide and experiment-local harness gates passed exactly.

## Current-result-aware reference choice

The reference deliberately incorporates all relevant YIXI results:

- **YIXI1:** standalone XGBoost is rejected at 0.65863872, but its three-model
  blend is the current verified project best at 0.67006499 valid.
- **YIXI2:** the 6b families produced no five-seed-confirmed LightGBM gain;
  notably, replacing 2.5-day user decay with 5-day decay was negative under
  LightGBM.
- **YIXI4:** the same 5-day replacement was strongly positive under the fixed
  XGBoost ranker, reaching 0.66649055 valid with five identical seed deltas.

Consequently, testing attention against the old iter44 LightGBM score or the
old 6a standalone XGBoost score would have used a stale, easier baseline.
YIXI3 instead freezes YIXI4's h5 XGBoost component. Project promotion would
still require a separately verified blend improvement over YIXI1's 0.67006499.

## Fixed model and feature reference

Every comparison held the exact 6a XGBoost hyperparameters fixed:

```text
XGBRanker(
    objective='rank:ndcg', eval_metric='ndcg@5-',
    max_depth=1, n_estimators=500, learning_rate=0.05,
    min_child_weight=1.0, reg_lambda=1.0, reg_alpha=0.0,
    subsample=1.0, colsample_bytree=1.0,
    tree_method='hist', max_bin=256,
    enable_categorical=True, max_cat_to_onehot=4,
    early_stopping_rounds=30, random_state=seed
)
```

The 11-column h5 reference preserves iter44's five categoricals and all other
numeric features, replacing only `decay_rate_2.5`/`decay_act_2.5` with the
unchanged YIXI2 definitions `decay_rate_5`/`decay_act_5`. Attention was the
only added model input.

## Causality and leakage verification

### Reused 5-day decay

Three validation rows were independently recomputed by directly summing
`0.5 ** (gap_days / 5)` over matching-user rows from strictly earlier dates.
Maximum absolute error was **5.33e-15**. A matching same-date row was present
and explicitly excluded. The check passed below `1e-10`.

### Attention history

Five validation rows with histories ranging from 8 to 124 interactions were
independently reconstructed. For both K=20 and K=40, the verifier rebuilt the
prior-user row set using strict `(time_ms, orig_idx)` comparison, recalculated
the scaled-dot softmax, and pooled the prior labels. It also independently
recomputed the uniform-history control.

```text
maximum absolute error: 0.0
zero-history sentinel rows checked: 20
history semantics: strict prior (time_ms, orig_idx) only
```

The item embeddings are learned parameters fit once on the training split,
analogous to the main models' ID embeddings. They never use validation or test
labels and are frozen before feature construction. The label-derived pool
itself only reads rows causally preceding the current row. This is the same
explicit training/causality boundary documented and verified in iter32.

The frozen embedding table contained 7,538 training videos; mean L2 norm was
1.20367. Unseen videos use iter32's zero-vector fallback, reducing attention
to a uniform history mean rather than producing NaN or failing.

## Validation-only protocol

- Fit the exact h5 reference first and require reproduction within `1e-8`.
- Add `attn_rate_20` and `attn_rate_40` independently at seed 0.
- Permit a K20+K40 union only if both individual features gain at least
  `+0.0003` valid.
- Select only among attention candidates using official validation primary.
- Run five paired seeds only if the selected attention candidate gains at
  least `+0.0003`; promotion requires at least `+0.001` in the mean and every
  seed.
- Test a three-model blend only after the attention feature itself confirms.
- Access test once for the validation-selected attention candidate after all
  gates and diagnostics are frozen.

The uniform K-matched history rate was declared diagnostic-only and could not
replace the selected attention feature.

## Attention-window results

| candidate | GAUC | nDCG@5 | primary | delta | best iteration | added-feature gain |
|---|---:|---:|---:|---:|---:|---:|
| h5 reference | 0.75049621 | 0.58248484 | **0.66649055** | — | 340 | — |
| h5 + K20 attention | 0.67979234 | 0.55216688 | 0.61597961 | -0.05051094 | 74 | 14.92% |
| h5 + K40 attention | 0.73915529 | 0.57767326 | 0.65841424 | -0.00807631 | 358 | 14.99% |

Both components of the official metric regress. For the less harmful K40
candidate, GAUC falls by 0.01134092 and nDCG@5 by 0.00481158. This is not a
tradeoff hidden by the primary average.

K20 and K40 both miss the preliminary threshold by a wide margin. The union
was therefore forbidden, and K40 was frozen as the best attention candidate.

## Uniform-history diagnostic

To distinguish useful target conditioning from merely exposing a longer
label window, the same K=40 history was pooled with uniform weights:

| feature | valid primary | delta vs h5 | added-feature gain |
|---|---:|---:|---:|
| uniform last-40 label mean | 0.63294274 | -0.03354782 | 12.23% |
| scaled-dot attention last-40 pool | 0.65841424 | -0.00807631 | 14.99% |

Target-conditioned attention recovers 0.02547 primary relative to the naive
long-window mean, so its similarity weights are doing something meaningful.
Nevertheless, neither summary helps the strong h5 stump model. The uniform
control was diagnostic only and was never selection-eligible.

## Redundancy and capacity diagnosis

The training correlations explain why the feature is attractive but harmful:

| attention feature | with uniform K rate | with h5 decay rate | with `lastk_rate` | mean absolute change from uniform |
|---|---:|---:|---:|---:|
| K20 | 0.9165 | 0.7050 | 0.7416 | 0.0619 |
| K40 | 0.9151 | 0.7360 | 0.7160 | 0.0602 |

Coverage is 97.70%, identical for both windows. Attention is neither absent
nor numerically degenerate: K40 takes 14.99% of total split gain, second only
to `tab`, while the reference h5 rate's share falls from 11.02% to 10.35%.
The feature therefore diverts substantial one-split-per-tree capacity toward
a highly correlated estimate of recent user success. Its extra target
conditioning is insufficient to compensate for displaced h5/tab/item signal.

K20 is noisier and more local, attracts essentially the same gain share, and
causes early stopping at only 74 rounds. K40's smoothing makes it less
destructive but not useful. This aligns with iter34 and iter40: attention over
the same recent label sequence overlaps the already strong decay and momentum
summaries instead of composing additively with them.

## Tie-artifact check

The selected K40 candidate was checked before test access:

```text
all-constant valid primary: 0.48367125
random valid primary:       0.48265904
K40 unique scores:          11,288 / 124,909 rows
mean within-user unique fraction: 0.8639
```

The model is far above the stable-sort floor and retains substantial
within-user score diversity. Its regression is a modeling result, not a tie
or row-order artifact.

## Why no five-seed confirmation or blend file was run

Section 3 says a single-run candidate must gain at least `+0.0003` before it
is examined again. The best attention candidate instead loses 0.00808.
Running four more deterministic XGBoost fits or searching blend weights after
that result would violate the stated threshold discipline.

This is also why the iter44-style condition boundary does not require a
`blend.py` here. `run_experiment.py` tests only attention feature utility. A
three-model attention blend would be a separate condition, but it was never
eligible to be created or executed. YIXI1's promoted blend remains the
project-level reference rather than being silently replaced by a weaker
standalone comparison.

## Frozen one-time test result

After the validation decision and REJECT verdict were fixed, K40 was evaluated
on test once:

| model | split | GAUC | nDCG@5 | primary |
|---|---|---:|---:|---:|
| YIXI4 h5 XGBoost | valid | 0.75049621 | 0.58248484 | 0.66649055 |
| K40 attention XGBoost | valid | 0.73915529 | 0.57767326 | 0.65841424 |
| YIXI4 h5 XGBoost | test | 0.71995062 | 0.58399141 | 0.65197098 |
| K40 attention XGBoost | test | 0.71552199 | 0.58270258 | 0.64911228 |

The selected attention candidate loses 0.00442863 GAUC, 0.00128883 nDCG@5,
and **0.00285870 primary** on test. Test agrees with, but did not determine,
the validation-only rejection.

## Conclusion

**REJECT lightweight attention pooling for the current native XGBoost
component.** The negative finding is clean:

- the exact strongest current component was reproduced first;
- both h5 decay and attention histories passed independent causal checks;
- both requested attention windows were tested independently;
- the feature received substantial model gain rather than being ignored;
- its signal is highly redundant with existing causal rate/momentum features;
- both official metric components regress;
- the validation failure is far below the preliminary threshold;
- the one frozen test check confirms the same direction.

The Section 6c null therefore extends the earlier sequence-model diagnosis to
the strongest current native tree component: preprocessing attention into a
continuous scalar does not unlock information that the h5 decay, momentum,
tab, and item features cannot already use more effectively.

## Artifacts

- `features.py`: imported frozen-attention construction, local cache,
  uniform control, alignment checks, and independent causal verification
- `run_experiment.py`: fixed h5-XGBoost feature sweep, threshold gates,
  diagnostics, and frozen one-time test
- `results.json`: exact metrics, feature gain, correlations, causality
  records, provenance, selection decisions, and final test result
- `.attention_cache_v1.pkl`: generated feature cache local to this experiment
  and excluded by the repository's experiment-cache ignore rule

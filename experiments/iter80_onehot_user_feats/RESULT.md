# iter80 — anonymized `onehot_feat0..17` user categoricals, GBM-native

## Provenance

The one slice of the project's own data deliberately excluded by every prior
iteration: iter15's own comment reads *"onehot_feat* columns are
anonymized/undocumented -- left unused, out of scope"*, and iter68's later
retest of `user_features_pure.csv` on the GBM-native representation (which
flipped several other directions from REJECT to ACCEPT) still only touched
the 6 documented demographic fields, not these 18. Genuinely untested by any
iteration to date (confirmed via `grep -rn onehot_feat experiments/`).

Inspection first (`pandas.nunique`): these are not literal one-hot vectors
despite the name — each of the 18 columns is itself a discrete anonymized
category code, cardinality from 2 (`onehot_feat0`) to 1471 (`onehot_feat3`);
6 of the 18 (indices 12-17) carry ~2.6% nulls (714/27,285 users). Treated as
native LightGBM categoricals (NaN → `'UNK'`), joined by `user_id`, exactly
matching iter68's `user` block's join pattern.

## Implementation

`experiments/iter80_onehot_user_feats/run.py`: reuses iter63's own
`prepare()`/base cat-num columns/hyperparameters unchanged (`num_leaves=2,
learning_rate=0.10, n_estimators=500, min_child_samples=200, reg_lambda=1.0,
linear_tree=True`), adds all 18 `onehot_feat*` columns as one block (not
swept individually — kept to a single cheap ablation). Harness-fidelity
check reproduced iter63's exact `rate_only` baseline (valid=0.67168,
test=0.65353) before trusting the new number.

## Result (seed 0)

| variant | valid | test | Δvalid | Δtest |
|---|---|---|---|---|
| baseline (`rate_only`) | 0.67168 | 0.65353 | +0.00000 | +0.00000 |
| +onehot (18 fields) | 0.67168 | 0.65353 | +0.00000 | +0.00000 |

Exact-zero delta.

## Diagnosis

Same mechanism already established for iter68's `user` demographic block and
iter75's categorical additions: at `num_leaves=2` there is exactly one split
in the whole tree, already won by `tab`/the decay-rate features; a
*categorical* column (unlike a numeric one) only contributes to
`linear_tree`'s prediction if it is selected as that split variable, since
the leaf-linear regression only regresses on numeric inputs. None of these
18 anonymized categoricals — despite some having real cardinality (up to
1471 distinct values, more than the demographic fields iter68 tested) — ever
wins that one split against the already-dominant decay-rate features, so
they contribute exactly nothing, regardless of whatever real signal they
may or may not encode about the user.

## Verdict: REJECT (clean no-op, no 5-seed confirmation needed)

iter63 remains the current best. This closes off the very last untested
static-side-info resource in the dataset — every column of every provided
CSV (`user_features_pure.csv` including the previously-skipped
`onehot_feat*` block, `video_features_statistic_pure.csv`,
`video_features_basic_pure.csv`, plus every log column) has now been tested
at least once on the GBM-native representation. Combined with iter68/75's
findings, this strongly suggests the model's bottleneck at `num_leaves=2` is
structural (only one split can ever be chosen, so no *categorical* addition
can help no matter its content) rather than a missing-data problem — a
capacity/architecture question, not a feature-search question. Any future
categorical-side-info attempt would need either a higher `num_leaves` (see
iter74/76's finding that `num_leaves=3` overfits under this project's
current regularization) or a fundamentally different way to expose
categoricals to `linear_tree` (e.g. as one-hot numeric dummy columns instead
of native categoricals) to have any chance of mattering.

# iter67 — Multi-task learning via GBM-side stacking (auxiliary engagement predictions) — independent re-verification

## Provenance

Reported by teammate "Xuxia" on a separate clone (`XUXIA_SUMMARY.md`
section 6c; see `experiments/LEDGER.md`'s "Parallel track" section).
Independently re-implemented and re-checked on this clone per direct user
instruction. Distinct from the project's earlier multi-task REJECTs
(iter31, iter36), which shared the FM's own embedding table/loss — this
approach has no shared gradient path at all: auxiliary predictions are
just additional GBM input columns.

## Hypothesis

The dataset carries several other engagement signals besides `long_view`
(`is_like`, `is_follow`, `is_comment`, `is_forward`). If these correlate
with `long_view` in a way the main feature set doesn't already capture,
predictions of these auxiliary labels — fed back as new GBM columns —
could add signal without needing joint multi-task training.

## Implementation

- **Aux label recovery**: `aux_labels.py` re-parses the two raw log CSVs
  in the exact same file-then-file, row-append order as
  `iter63_decay_tab_rate/data_ext.py`'s `_load_raw_time` (whose `orig_idx`
  field is precisely each row's position in that same unfiltered read
  order), recovering `is_like`/`is_follow`/`is_comment`/`is_forward` per
  `orig_idx`.
- **Alignment verification**: rather than a spot sample, the reconstructed
  `long_view` (parsed by the same loop) was compared against the
  already-trusted `y[split]` array for **every row in every split**:
  1,141,112 train + 124,909 valid + 170,588 test rows, **0 mismatches** —
  an exhaustive proof the `orig_idx` alignment is exactly correct, not
  just a spot-checked one.
- **Auxiliary classifiers**: 4 independent `LGBMClassifier` models
  (`n_estimators=100, num_leaves=31, learning_rate=0.1`) on the same
  native feature set as the main GBM (`iter63_decay_tab_rate`'s
  `rate_only` variant). Each trained via 5-fold OOF on train (row-level
  `KFold(shuffle=True, random_state=0)`, no leakage — a row's aux
  prediction always comes from a fold that excluded it) and a separate
  full-train fit applied to valid/test.
- **Main GBM**: the 4 auxiliary probability columns
  (`aux_is_like`/`aux_is_follow`/`aux_is_comment`/`aux_is_forward`) were
  appended to the native feature set, and the main `long_view` GBM
  retrained with iter63's exact hyperparameters
  (`num_leaves=2, learning_rate=0.10, n_estimators=500,
  min_child_samples=200, reg_lambda=1.0, linear_tree=True`), seed=0.

Train-split prevalence of the 4 auxiliary labels (all rare): `is_like`
1.87%, `is_follow` 0.10%, `is_comment` 0.26%, `is_forward` 0.10%.

## Result (single seed 0, per protocol — a >0.001 valid gain would need a 5-seed confirm; none was observed)

| | valid | test |
|---|---|---|
| baseline (`rate_only`, no aux columns) | 0.67168 | 0.65353 |
| with 4 aux columns | 0.67168 | 0.65353 |
| delta | **+0.00000** | **+0.00000** |

An **exact tie to 5 decimal places** — the retrained model is bit-for-bit
identical in its evaluated metric. Feature-importance inspection confirms
why: across all 48 trees actually built (`linear_tree=True` with
`num_leaves=2` builds very shallow trees), **0 of the 4 auxiliary columns
were ever chosen as a split feature.** This is an even cleaner null result
than Xuxia's own report (their single-seed run found a marginal
+0.00024 valid with `aux_is_like` used as a split exactly once, below
their 0.0003 look-threshold) — the difference is consistent with ordinary
OOF-fold/random-seed variation in the auxiliary classifiers' output
distribution, not a disagreement about the underlying finding.

## Diagnosis

Both this run and Xuxia's independently arrive at the same mechanism: the
engagement labels being predicted (`is_like`, `is_follow`, `is_comment`,
`is_forward`) are each under 2% prevalent and, per the auxiliary
classifiers' own OOF predictions, barely distinguishable from their base
rates across most rows (e.g. `is_follow` OOF mean_pred=0.0020 vs. actual
rate=0.0010) — these signals carry very little row-level information to
begin with, so a `num_leaves=2` linear-leaf GBM (already highly
regularized by construction) has no incentive to ever split on them ahead
of the existing, much more informative recency/decay features. This points
to the engagement signals themselves being low-information for
`long_view` prediction under this feature representation, not an
artifact of the stacking mechanism or of iter31/36's earlier
shared-embedding conflict (which structurally cannot apply here).

## Verdict: **REJECT** (independently confirmed, with an even cleaner exact-zero result)

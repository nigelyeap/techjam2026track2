# iterMERGE1 — independent verification of yixi's iterYIXI10 result

## Verdict: VERIFIED — yixi's YIXI10 result confirmed independently

Retrained all three components from raw CSVs (not her frozen `.npz`
artifacts) and reproduced her claimed blend exactly:

| component | ours | her claim | delta |
|---|---:|---:|---:|
| LightGBM (YIXI10 columns incl. `upload_type`, seed 0) | 0.68834144 | 0.68834144 | +0.00000000 |
| XGBoost (YIXI5 tuned, seed 0) | 0.66755420 | 0.66755420 | +0.00000000 |
| FM (5-seed sigmoid-mean ensemble via `submission.train_one_fm`) | 0.63987792 | 0.63987792 | +0.00000000 |
| **Blend (10% FM / 52% LGB / 38% XGB, within-user percentile)** | **valid 0.69943440 / test 0.68432260** | **valid 0.69943440 / test 0.68432260** | **+0.00000000 both** |

## Method

`experiments/iterMERGE1_verify_yixi10/verify.py`: rebuilt the YIXI9/YIXI10
causal feature frames independently (timestamp-causal watch-depth history,
5-day user-decay pair, `decay_tab_rate_3`, `upload_type`) without reusing any
of yixi's cached/frozen prediction files; trained each of the three model
components at seed 0 through the same code path she uses (LightGBM via her
reference columns, XGBoost via her tuned config, FM via this repo's own
shared `submission.train_one_fm`, confirming her FM component is literally
our code, not a divergent implementation); applied her within-user-percentile
blend at the stated 10/52/38 weights; evaluated valid and test.

## Conclusion

Every stage — LightGBM, XGBoost, FM, and the final blend, both valid and
test — matches yixi's claimed numbers to 8 decimal places (delta
+0.00000000 throughout), well inside the 1e-3 tolerance specified for this
check. Final numbers: **valid 0.69943440 / test 0.68432260**, reproduced
from raw CSVs with none of yixi's cached/frozen prediction artifacts reused.

No promotion action taken here (that touches `SUBMISSION.md`/
`make_submission.py`/`submission.csv`, reserved for explicit user
go-ahead) — this iteration only establishes that the number is real.

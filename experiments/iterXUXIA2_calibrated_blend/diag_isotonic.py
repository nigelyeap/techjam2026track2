"""Diagnose why isotonic-calibrated blending collapsed so badly (-0.036
valid vs. current blend, seed 0): checks whether isotonic regression is
destroying GBM's fine-grained ranking distinctions by pooling them into a
small number of tied plateaus (the tie-artifact-awareness pattern from
Section 3, applied to an unusually BAD result instead of a suspiciously
good one)."""
import os, sys
import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_gbm, get_fm_ensemble, minmax  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate  # noqa: E402

if __name__ == '__main__':
    gbm = get_gbm(seed=0, verbose=False)
    fm = get_fm_ensemble(verbose=False)

    gbm_va_n = minmax(gbm['gbm_va_raw'])
    print(f"raw GBM (minmax) standalone valid: {evaluate(gbm['uva'], gbm['yva'], gbm_va_n)['primary']:.5f}")
    print(f"  unique score values: {len(np.unique(gbm_va_n))} / {len(gbm_va_n)} rows")

    iso_gbm = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso_gbm.fit(gbm['gbm_tr_raw'], gbm['ytr'])
    calib_gbm_va = iso_gbm.predict(gbm['gbm_va_raw'])
    print(f"isotonic-calibrated GBM standalone valid: {evaluate(gbm['uva'], gbm['yva'], calib_gbm_va)['primary']:.5f}")
    print(f"  unique score values: {len(np.unique(calib_gbm_va))} / {len(calib_gbm_va)} rows")
    print(f"  isotonic regressor's own number of pooled plateaus (X thresholds): {len(iso_gbm.X_thresholds_)}")

    print()
    fm_va = fm['fm_va_ens']
    print(f"raw FM ensemble standalone valid: {evaluate(fm['uva'], fm['yva'], fm_va)['primary']:.5f}")
    print(f"  unique score values: {len(np.unique(fm_va))} / {len(fm_va)} rows")
    iso_fm = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso_fm.fit(fm['fm_tr_ens'], fm['ytr'])
    calib_fm_va = iso_fm.predict(fm_va)
    print(f"isotonic-calibrated FM standalone valid: {evaluate(fm['uva'], fm['yva'], calib_fm_va)['primary']:.5f}")
    print(f"  unique score values: {len(np.unique(calib_fm_va))} / {len(calib_fm_va)} rows")
    print(f"  isotonic regressor's own number of pooled plateaus (X thresholds): {len(iso_fm.X_thresholds_)}")

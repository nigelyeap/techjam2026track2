"""Harness-fidelity check for this experiment's own code path (common.py).

Reproduces iter63's exact standalone and global-alpha-blend numbers using
this folder's own get_gbm/get_fm_ensemble helpers, before any segmentation
logic is trusted. Expected (experiments/iter63_decay_tab_rate/blend_results.json):
  GBM standalone: valid=0.67168 test=0.65353
  FM ensemble standalone: valid=0.63988 test=0.64187
  global-alpha blend (alpha=0.14): valid=0.67606 test=0.65955
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_gbm, get_fm_ensemble, minmax  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate  # noqa: E402

if __name__ == '__main__':
    gbm = get_gbm(seed=0, verbose=True)
    print(f"GBM standalone: valid={gbm['va_metrics']['primary']:.5f} test={gbm['te_metrics']['primary']:.5f}")

    fm = get_fm_ensemble(verbose=True)
    assert np.array_equal(fm['yva'], gbm['yva']), "FM/GBM valid label order mismatch"
    assert np.array_equal(fm['yte'], gbm['yte']), "FM/GBM test label order mismatch"
    fm_va_m = evaluate(fm['uva'], fm['yva'], fm['fm_va_ens'])
    fm_te_m = evaluate(fm['ute'], fm['yte'], fm['fm_te_ens'])
    print(f"FM ensemble standalone: valid={fm_va_m['primary']:.5f} test={fm_te_m['primary']:.5f}")

    gbm_va_n, gbm_te_n = minmax(gbm['gbm_va_raw']), minmax(gbm['gbm_te_raw'])
    alpha = 0.14
    blend_va = alpha * fm['fm_va_ens'] + (1 - alpha) * gbm_va_n
    blend_te = alpha * fm['fm_te_ens'] + (1 - alpha) * gbm_te_n
    va_m = evaluate(gbm['uva'], gbm['yva'], blend_va)
    te_m = evaluate(gbm['ute'], gbm['yte'], blend_te)
    print(f"global-alpha blend (alpha=0.14): valid={va_m['primary']:.5f} test={te_m['primary']:.5f}")

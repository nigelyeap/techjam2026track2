"""iter66: independent re-verification of Xuxia's rank-based/calibrated
blending finding (see experiments/LEDGER.md). Compares the current
linear+minmax blend (alpha=0.14) against (a) Borda rank fusion, (b)
reciprocal-rank fusion (RRF), both computed PER USER (each user's own
candidate set is the natural "query" for rank fusion here), across all 5
GBM seeds from iter65's scores_5seed.npz. Isotonic-regression calibration
(fit on train) is checked separately in run_isotonic.py since it needs
train-split scores not saved by iter65's generator.
"""
import os, sys
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
_ITER65_DIR = os.path.join(_REPO_ROOT, 'experiments', 'iter65_segment_blend')
sys.path.insert(0, _REPO_ROOT)
from evaluate import evaluate  # noqa: E402

ALPHA_GLOBAL = 0.14
RRF_K = 60  # standard RRF constant


def minmax(x):
    x = np.asarray(x, dtype=np.float64); lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def per_user_ranks(u, scores):
    """Rank within each user's group, descending (rank 1 = highest score)."""
    df = pd.DataFrame({'u': u, 's': scores})
    return df.groupby('u')['s'].rank(ascending=False, method='average').to_numpy()


def borda_fuse(u, s_gbm, s_fm):
    r_gbm = per_user_ranks(u, s_gbm)
    r_fm = per_user_ranks(u, s_fm)
    return -(r_gbm + r_fm)  # higher = better (lower combined rank)


def rrf_fuse(u, s_gbm, s_fm, k=RRF_K):
    r_gbm = per_user_ranks(u, s_gbm)
    r_fm = per_user_ranks(u, s_fm)
    return 1.0 / (k + r_gbm) + 1.0 / (k + r_fm)


if __name__ == '__main__':
    d = np.load(os.path.join(_ITER65_DIR, 'scores_5seed.npz'), allow_pickle=True)
    y_va, y_te, u_va, u_te = d['y_va'], d['y_te'], d['u_va'], d['u_te']
    fm_va, fm_te = d['fm_va_ens'], d['fm_te_ens']

    borda_va_d, borda_te_d, rrf_va_d, rrf_te_d = [], [], [], []
    for seed in range(5):
        gbm_va_norm = minmax(d['gbm_va_raw'][seed])
        gbm_te_norm = minmax(d['gbm_te_raw'][seed])
        global_va = ALPHA_GLOBAL * fm_va + (1 - ALPHA_GLOBAL) * gbm_va_norm
        global_te = ALPHA_GLOBAL * fm_te + (1 - ALPHA_GLOBAL) * gbm_te_norm
        gva_m = evaluate(u_va, y_va, global_va)['primary']
        gte_m = evaluate(u_te, y_te, global_te)['primary']

        borda_va = borda_fuse(u_va, gbm_va_norm, fm_va)
        borda_te = borda_fuse(u_te, gbm_te_norm, fm_te)
        bva_m = evaluate(u_va, y_va, borda_va)['primary']
        bte_m = evaluate(u_te, y_te, borda_te)['primary']

        rrf_va = rrf_fuse(u_va, gbm_va_norm, fm_va)
        rrf_te = rrf_fuse(u_te, gbm_te_norm, fm_te)
        rva_m = evaluate(u_va, y_va, rrf_va)['primary']
        rte_m = evaluate(u_te, y_te, rrf_te)['primary']

        print(f"seed={seed}  global valid={gva_m:.5f} test={gte_m:.5f}  |  "
              f"borda valid={bva_m:.5f} ({bva_m-gva_m:+.5f}) test={bte_m:.5f} ({bte_m-gte_m:+.5f})  |  "
              f"rrf valid={rva_m:.5f} ({rva_m-gva_m:+.5f}) test={rte_m:.5f} ({rte_m-gte_m:+.5f})", flush=True)
        borda_va_d.append(bva_m - gva_m); borda_te_d.append(bte_m - gte_m)
        rrf_va_d.append(rva_m - gva_m); rrf_te_d.append(rte_m - gte_m)

    borda_va_d = np.array(borda_va_d); rrf_va_d = np.array(rrf_va_d)
    print(f"\nborda: mean delta valid={borda_va_d.mean():+.5f} wins={int((borda_va_d>0).sum())}/5 "
          f"mean delta test={np.mean(borda_te_d):+.5f}")
    print(f"rrf:   mean delta valid={rrf_va_d.mean():+.5f} wins={int((rrf_va_d>0).sum())}/5 "
          f"mean delta test={np.mean(rrf_te_d):+.5f}")

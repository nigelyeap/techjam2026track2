"""6b: rank-based / calibrated blending vs. the current linear+minmax blend.

Two alternatives, compared against the current `alpha*FM + (1-alpha)*
minmax(GBM)` blend on valid:

1. Rank-average (Borda-style) fusion: within each user's group, convert
   each model's raw scores to normalized ranks (average-rank for ties,
   scaled to (0,1] with 1=best), then blend `beta*rank_FM + (1-beta)*
   rank_GBM`. Also tries weighted reciprocal-rank fusion (RRF) as a second
   rank-based variant.
2. Isotonic-regression calibration: fit an isotonic regressor mapping each
   model's raw TRAIN scores to calibrated P(long_view), then blend the
   calibrated probabilities the same way the current blend mixes
   min-max-normalized scores.

Repeated across 5 GBM seeds, FM ensemble held fixed, same convention as
iterXUXIA1.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_gbm, get_fm_ensemble, minmax  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate  # noqa: E402

ALPHA_GRID = np.round(np.arange(0.0, 1.01, 0.02), 4)
GBM_SEEDS = (0, 1, 2, 3, 4)
RRF_K_GRID = (10, 30, 60, 100)


def _descending_rank(scores, user_ids):
    """Per-user average rank, descending (rank 1 = best score). Vectorized
    via pandas groupby (C-level), not a per-user Python loop -- the naive
    loop version was the bottleneck that made the first attempt at this
    script too slow to finish (killed after >6min CPU on seed 0 alone)."""
    df = pd.DataFrame({'u': user_ids, 's': scores})
    return df.groupby('u')['s'].rank(method='average', ascending=False).to_numpy()


def normalized_rank(scores, user_ids):
    """Per-user rank normalized to (0,1], 1=best (ascending average rank / group size)."""
    df = pd.DataFrame({'u': user_ids, 's': scores})
    ranks = df.groupby('u')['s'].rank(method='average', ascending=True)
    counts = df.groupby('u')['s'].transform('count')
    return (ranks / counts).to_numpy()


def rrf_component(desc_rank, k):
    """1/(k+rank) given a precomputed descending rank array (rank 1=best)."""
    return 1.0 / (k + desc_rank)


def sweep_beta(fm_x, gbm_x, y, u):
    best_beta, best_valid = -1, -1
    for b in ALPHA_GRID:
        m = evaluate(u, y, b * fm_x + (1 - b) * gbm_x)
        if m['primary'] > best_valid:
            best_valid, best_beta = m['primary'], b
    return float(best_beta), float(best_valid)


def run_for_seed(gbm_seed, fm, verbose=True):
    gbm = get_gbm(seed=gbm_seed, verbose=False)
    assert np.array_equal(gbm['yva'], fm['yva']) and np.array_equal(gbm['yte'], fm['yte'])
    yva, uva = gbm['yva'], gbm['uva']
    yte, ute = gbm['yte'], gbm['ute']
    ytr, utr = gbm['ytr'], gbm['utr']

    # --- baseline: current linear + minmax blend ---
    gbm_va_n, gbm_te_n = minmax(gbm['gbm_va_raw']), minmax(gbm['gbm_te_raw'])
    fm_va, fm_te = fm['fm_va_ens'], fm['fm_te_ens']
    lin_beta, lin_valid = sweep_beta(fm_va, gbm_va_n, yva, uva)
    lin_test = evaluate(ute, yte, lin_beta * fm_te + (1 - lin_beta) * gbm_te_n)['primary']

    # --- 1a. rank-average (Borda) fusion ---
    rank_fm_va = normalized_rank(fm_va, uva)
    rank_gbm_va = normalized_rank(gbm_va_n, uva)
    rank_fm_te = normalized_rank(fm_te, ute)
    rank_gbm_te = normalized_rank(gbm_te_n, ute)
    borda_beta, borda_valid = sweep_beta(rank_fm_va, rank_gbm_va, yva, uva)
    borda_test = evaluate(ute, yte, borda_beta * rank_fm_te + (1 - borda_beta) * rank_gbm_te)['primary']

    # --- 1b. reciprocal-rank fusion, best K ---
    desc_fm_va, desc_gbm_va = _descending_rank(fm_va, uva), _descending_rank(gbm_va_n, uva)
    desc_fm_te, desc_gbm_te = _descending_rank(fm_te, ute), _descending_rank(gbm_te_n, ute)
    best_rrf = {'k': None, 'beta': None, 'valid': -1}
    for k in RRF_K_GRID:
        rrf_fm_va = rrf_component(desc_fm_va, k)
        rrf_gbm_va = rrf_component(desc_gbm_va, k)
        b, v = sweep_beta(rrf_fm_va, rrf_gbm_va, yva, uva)
        if v > best_rrf['valid']:
            best_rrf = {'k': k, 'beta': b, 'valid': v}
    rrf_fm_te = rrf_component(desc_fm_te, best_rrf['k'])
    rrf_gbm_te = rrf_component(desc_gbm_te, best_rrf['k'])
    rrf_test = evaluate(ute, yte, best_rrf['beta'] * rrf_fm_te + (1 - best_rrf['beta']) * rrf_gbm_te)['primary']

    # --- 2. isotonic-regression calibration (fit on TRAIN only) ---
    # Isotonic regression only needs monotonic ordering of X, not a
    # particular pre-scaling -- fit directly on raw scores; the output
    # (calibrated P(long_view), y_min=0/y_max=1) is already on a common
    # scale across both models, unlike the raw scores themselves.
    iso_gbm = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso_gbm.fit(gbm['gbm_tr_raw'], ytr)
    calib_gbm_va = iso_gbm.predict(gbm['gbm_va_raw'])
    calib_gbm_te = iso_gbm.predict(gbm['gbm_te_raw'])

    iso_fm = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso_fm.fit(fm['fm_tr_ens'], ytr)
    calib_fm_va = iso_fm.predict(fm_va)
    calib_fm_te = iso_fm.predict(fm_te)

    iso_beta, iso_valid = sweep_beta(calib_fm_va, calib_gbm_va, yva, uva)
    iso_test = evaluate(ute, yte, iso_beta * calib_fm_te + (1 - iso_beta) * calib_gbm_te)['primary']

    result = {
        'gbm_seed': gbm_seed,
        'linear_minmax': {'beta': lin_beta, 'valid': lin_valid, 'test': lin_test},
        'rank_borda': {'beta': borda_beta, 'valid': borda_valid, 'test': borda_test},
        'rank_rrf': {'k': best_rrf['k'], 'beta': best_rrf['beta'], 'valid': best_rrf['valid'], 'test': rrf_test},
        'isotonic': {'beta': iso_beta, 'valid': iso_valid, 'test': iso_test},
    }
    if verbose:
        print(f"\n--- GBM seed {gbm_seed} ---")
        print(f"  linear+minmax (current): beta={lin_beta:.2f} valid={lin_valid:.5f} test={lin_test:.5f}")
        print(f"  rank-borda:              beta={borda_beta:.2f} valid={borda_valid:.5f} (delta={borda_valid-lin_valid:+.5f}) test={borda_test:.5f}")
        print(f"  rank-rrf (k={best_rrf['k']}):        beta={best_rrf['beta']:.2f} valid={best_rrf['valid']:.5f} (delta={best_rrf['valid']-lin_valid:+.5f}) test={rrf_test:.5f}")
        print(f"  isotonic-calibrated:     beta={iso_beta:.2f} valid={iso_valid:.5f} (delta={iso_valid-lin_valid:+.5f}) test={iso_test:.5f}")
    return result


if __name__ == '__main__':
    fm = get_fm_ensemble(verbose=True)
    all_results = []
    for seed in GBM_SEEDS:
        all_results.append(run_for_seed(seed, fm, verbose=True))

    print("\n=== 5-seed summary ===")
    for method in ('rank_borda', 'rank_rrf', 'isotonic'):
        deltas = [r[method]['valid'] - r['linear_minmax']['valid'] for r in all_results]
        print(f"{method} valid deltas per seed: {[f'{d:+.5f}' for d in deltas]}")
        print(f"  mean={np.mean(deltas):+.5f}  min={np.min(deltas):+.5f}  seeds>=+0.001: {sum(d >= 0.001 for d in deltas)}/5")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibrated_blend_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nwrote calibrated_blend_results.json")

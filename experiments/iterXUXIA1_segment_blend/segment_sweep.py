"""6a: per-segment blend alpha, instead of one global alpha=0.14.

Segments the valid set two ways -- by `tab` (a handful of discrete values)
and by tertiles of the decayed-activity feature (`decay_act_2.5`, the
denominator of the decayed per-tab rate: total decayed prior exposure
count, already used elsewhere in this project as an activity-tier proxy)
-- sweeps alpha independently within each segment on valid, applies the
learned per-segment alpha to the corresponding rows (test rows get the
alpha chosen for their own segment, using tertile edges fit on valid
only), and evaluates the resulting per-row-blended scores with the
project's standard full-set evaluate() (same user groupings as every
other number in this project -- segmentation only changes which alpha a
row gets, not how the metric is computed).

Repeated across 5 GBM seeds (0-4) with the FM ensemble held fixed (same
convention as every other GBM-only resweep in this project, e.g.
iter55-iter63's blend.py: "iter38's unchanged ensemble") to guard against
the extra free parameters (one alpha per segment instead of one global
alpha) fitting valid-set noise rather than real signal.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_gbm, get_fm_ensemble, minmax  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate  # noqa: E402

ALPHA_GRID = np.round(np.arange(0.0, 0.41, 0.01), 4)
GBM_SEEDS = (0, 1, 2, 3, 4)
GLOBAL_ALPHA_BASELINE_VALID = 0.67606  # iter63 blend, current project best (this project's HEAD)


def sweep_alpha_on_mask(fm_scores, gbm_scores, y, u, mask):
    """Best alpha (from ALPHA_GRID) restricted to rows where mask is True."""
    best_alpha, best_score = ALPHA_GRID[0], -1.0
    ym, um = y[mask], u[mask]
    fmm, gbmm = fm_scores[mask], gbm_scores[mask]
    for a in ALPHA_GRID:
        s = a * fmm + (1 - a) * gbmm
        m = evaluate(um, ym, s)
        if m['primary'] > best_score:
            best_score, best_alpha = m['primary'], a
    return float(best_alpha), float(best_score)


def apply_segment_alphas(fm_scores, gbm_scores, seg_ids, alpha_by_seg, default_alpha):
    alphas = np.array([alpha_by_seg.get(s, default_alpha) for s in seg_ids])
    return alphas * fm_scores + (1 - alphas) * gbm_scores


def run_for_seed(gbm_seed, fm, verbose=True):
    gbm = get_gbm(seed=gbm_seed, verbose=False)
    assert np.array_equal(gbm['yva'], fm['yva']) and np.array_equal(gbm['yte'], fm['yte']), \
        "label order mismatch between GBM and FM caches"
    gbm_va_n, gbm_te_n = minmax(gbm['gbm_va_raw']), minmax(gbm['gbm_te_raw'])
    fm_va, fm_te = fm['fm_va_ens'], fm['fm_te_ens']
    yva, uva = gbm['yva'], gbm['uva']
    yte, ute = gbm['yte'], gbm['ute']

    # global alpha (reference, fine grid)
    best_g_alpha, best_g_valid = -1, -1
    for a in ALPHA_GRID:
        m = evaluate(uva, yva, a * fm_va + (1 - a) * gbm_va_n)
        if m['primary'] > best_g_valid:
            best_g_valid, best_g_alpha = m['primary'], a
    global_te = evaluate(ute, yte, best_g_alpha * fm_te + (1 - best_g_alpha) * gbm_te_n)['primary']

    results = {'gbm_seed': gbm_seed, 'global': {'alpha': float(best_g_alpha), 'valid': float(best_g_valid), 'test': float(global_te)}}

    # --- segment by tab ---
    tabs = sorted(set(gbm['tab_va'].tolist()))
    tab_alpha, tab_sizes = {}, {}
    for t in tabs:
        mask_va = gbm['tab_va'] == t
        a, _ = sweep_alpha_on_mask(fm_va, gbm_va_n, yva, uva, mask_va)
        tab_alpha[t] = a
        tab_sizes[t] = {'valid_rows': int(mask_va.sum()), 'valid_users': int(len(set(uva[mask_va].tolist()))),
                         'test_rows': int((gbm['tab_te'] == t).sum())}
    blend_va_tab = apply_segment_alphas(fm_va, gbm_va_n, gbm['tab_va'], tab_alpha, best_g_alpha)
    blend_te_tab = apply_segment_alphas(fm_te, gbm_te_n, gbm['tab_te'], tab_alpha, best_g_alpha)
    tab_valid_m = evaluate(uva, yva, blend_va_tab)['primary']
    tab_test_m = evaluate(ute, yte, blend_te_tab)['primary']
    results['tab_segment'] = {'alphas': tab_alpha, 'sizes': tab_sizes,
                               'valid': float(tab_valid_m), 'test': float(tab_test_m)}

    # --- segment by activity tertile (edges fit on valid only) ---
    edges = np.quantile(gbm['act_va'], [1 / 3, 2 / 3])
    tier_va = np.digitize(gbm['act_va'], edges)
    tier_te = np.digitize(gbm['act_te'], edges)
    tier_alpha, tier_sizes = {}, {}
    for tier in (0, 1, 2):
        mask_va = tier_va == tier
        a, _ = sweep_alpha_on_mask(fm_va, gbm_va_n, yva, uva, mask_va)
        tier_alpha[tier] = a
        tier_sizes[tier] = {'valid_rows': int(mask_va.sum()), 'valid_users': int(len(set(uva[mask_va].tolist()))),
                             'test_rows': int((tier_te == tier).sum())}
    blend_va_tier = apply_segment_alphas(fm_va, gbm_va_n, tier_va, tier_alpha, best_g_alpha)
    blend_te_tier = apply_segment_alphas(fm_te, gbm_te_n, tier_te, tier_alpha, best_g_alpha)
    tier_valid_m = evaluate(uva, yva, blend_va_tier)['primary']
    tier_test_m = evaluate(ute, yte, blend_te_tier)['primary']
    results['tier_segment'] = {'alphas': {int(k): v for k, v in tier_alpha.items()}, 'sizes': tier_sizes,
                                'edges': edges.tolist(),
                                'valid': float(tier_valid_m), 'test': float(tier_test_m)}

    if verbose:
        print(f"\n--- GBM seed {gbm_seed} ---")
        print(f"  global alpha={best_g_alpha:.2f}: valid={best_g_valid:.5f} test={global_te:.5f}")
        print(f"  tab-segment alphas={tab_alpha}")
        print(f"    valid={tab_valid_m:.5f} (delta={tab_valid_m - best_g_valid:+.5f})  test={tab_test_m:.5f} (delta={tab_test_m - global_te:+.5f})")
        print(f"  tier-segment alphas={tier_alpha}  tertile edges={edges}")
        print(f"    valid={tier_valid_m:.5f} (delta={tier_valid_m - best_g_valid:+.5f})  test={tier_test_m:.5f} (delta={tier_test_m - global_te:+.5f})")
    return results


if __name__ == '__main__':
    fm = get_fm_ensemble(verbose=True)
    all_results = []
    for seed in GBM_SEEDS:
        r = run_for_seed(seed, fm, verbose=True)
        all_results.append(r)

    print("\n=== 5-seed summary ===")
    tab_deltas = [r['tab_segment']['valid'] - r['global']['valid'] for r in all_results]
    tier_deltas = [r['tier_segment']['valid'] - r['global']['valid'] for r in all_results]
    print(f"tab-segment valid deltas per seed: {[f'{d:+.5f}' for d in tab_deltas]}")
    print(f"  mean={np.mean(tab_deltas):+.5f}  min={np.min(tab_deltas):+.5f}  seeds>=+0.001: {sum(d >= 0.001 for d in tab_deltas)}/5")
    print(f"tier-segment valid deltas per seed: {[f'{d:+.5f}' for d in tier_deltas]}")
    print(f"  mean={np.mean(tier_deltas):+.5f}  min={np.min(tier_deltas):+.5f}  seeds>=+0.001: {sum(d >= 0.001 for d in tier_deltas)}/5")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'segment_sweep_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nwrote segment_sweep_results.json")

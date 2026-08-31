"""Pure-numpy blend check: loads iter63_scores.npz + sasrec_scores_seed*.npz
(no torch, no lightgbm -- avoids the segfault) and sweeps beta blending the
SASRec score on top of the current iter63 blend (alpha=0.14 GBM/FM).
"""
import os, sys, glob, json, re
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
from evaluate import evaluate  # noqa: E402

ALPHA_BLEND = 0.14


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


if __name__ == '__main__':
    d63 = np.load(os.path.join(_THIS_DIR, 'iter63_scores.npz'), allow_pickle=True)
    yva, yte = d63['y_va'], d63['y_te']
    uva, ute = d63['u_va'], d63['u_te']

    gbm_va_norm, gbm_te_norm = minmax(d63['gbm_va_raw']), minmax(d63['gbm_te_raw'])
    iter63_va = ALPHA_BLEND * d63['fm_va_ens'] + (1 - ALPHA_BLEND) * gbm_va_norm
    iter63_te = ALPHA_BLEND * d63['fm_te_ens'] + (1 - ALPHA_BLEND) * gbm_te_norm
    iter63_va_m = evaluate(uva, yva, iter63_va)
    iter63_te_m = evaluate(ute, yte, iter63_te)
    print(f"iter63 blend (current best): valid={iter63_va_m['primary']:.5f} test={iter63_te_m['primary']:.5f}")

    iter63_va_norm, iter63_te_norm = minmax(iter63_va), minmax(iter63_te)

    files = sorted(glob.glob(os.path.join(_THIS_DIR, 'sasrec_scores_seed*.npz')))
    all_out = {'iter63_blend': {'valid': iter63_va_m['primary'], 'test': iter63_te_m['primary']}, 'per_seed': {}}
    for fpath in files:
        seed = re.search(r'seed(\d+)', fpath).group(1)
        d = np.load(fpath, allow_pickle=True)
        assert np.array_equal(d['y_va'], yva), f"label mismatch seed {seed} valid"
        assert np.array_equal(d['y_te'], yte), f"label mismatch seed {seed} test"
        sas_va_m = evaluate(uva, yva, d['va_scores'])
        sas_te_m = evaluate(ute, yte, d['te_scores'])
        print(f"\nseed {seed}: SASRec standalone valid={sas_va_m['primary']:.5f} test={sas_te_m['primary']:.5f}")

        sas_va_norm, sas_te_norm = minmax(d['va_scores']), minmax(d['te_scores'])
        seed_results = []
        for beta in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
            cva = (1 - beta) * iter63_va_norm + beta * sas_va_norm
            cte = (1 - beta) * iter63_te_norm + beta * sas_te_norm
            vm = evaluate(uva, yva, cva)
            tm = evaluate(ute, yte, cte)
            seed_results.append({'beta': beta, 'valid': vm['primary'], 'test': tm['primary']})
            print(f"  beta={beta:.2f}  valid={vm['primary']:.5f}  test={tm['primary']:.5f}")
        best = max(seed_results, key=lambda r: r['valid'])
        print(f"  best beta on valid: {best['beta']:.2f} -> valid={best['valid']:.5f} (delta {best['valid']-iter63_va_m['primary']:+.5f})")
        all_out['per_seed'][seed] = {'standalone': {'valid': sas_va_m['primary'], 'test': sas_te_m['primary']},
                                       'beta_sweep': seed_results, 'best': best}

    def _floatify(o):
        if isinstance(o, dict):
            return {k: _floatify(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_floatify(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    with open(os.path.join(_THIS_DIR, 'blend_check2_results.json'), 'w') as f:
        json.dump(_floatify(all_out), f, indent=2)
    print("\nDONE")

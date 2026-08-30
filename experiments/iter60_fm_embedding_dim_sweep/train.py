"""iter60: FM embedding dimension (k) resweep, single seed.

k=16 has been the FM embedding dimension since iter38 and was never
resystematically resweept in this session's records. This is the FM-side
analogue of the GBM's num_leaves capacity knob -- a natural next lever now
that the GBM hyperparameter space (iter53/57/58/59) is exhausted. Single
seed=0 standalone metric first; only worth a 5-seed confirmation + blend
re-run if a real (>0.0003 valid) gain shows up.
"""
import os, sys, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'iter27_triple_fusion'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'iter47_stacking_meta'))
from evaluate import evaluate  # noqa: E402
from data_ext import load_ext, encode_ext, HALFLIVES, TAB_HALFLIVES  # noqa: E402
import stack as iter47  # noqa: E402

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')

if __name__ == '__main__':
    splits = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc, dim = encode_ext(splits, feature_set=FEATURES, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                           alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    K_GRID = [8, 12, 16, 24, 32, 48, 64]
    results = []
    for k in K_GRID:
        m = iter47.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed=0, k=k)
        va_scores = iter47.sigmoid(m.predict(Xva))
        te_scores = iter47.sigmoid(m.predict(Xte))
        va = evaluate(uva, yva, va_scores)
        te = evaluate(ute, yte, te_scores)
        print(f"k={k:3d}  valid={va['primary']:.5f}  test={te['primary']:.5f}", flush=True)
        results.append((k, va['primary'], te['primary']))

    best = max(results, key=lambda r: r[1])
    print(f"\nbest: k={best[0]}  valid={best[1]:.5f}  test={best[2]:.5f}")
    print(f"iter38 baseline (k=16): see results above")

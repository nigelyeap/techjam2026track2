"""iter61: FM learning_rate resweep, single seed.

The GBM's own learning_rate turned out to be the one hyperparameter left
stale from an earlier structural change (iter55, a real gain). The FM's
lr=0.001 (Adam) has been unchanged since iter38 and was never
resystematically resweept. Single seed=0 standalone metric first; only
worth a 5-seed confirmation + blend re-run if a real (>0.0003 valid) gain
shows up.
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

    LR_GRID = [0.0002, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005]
    results = []
    for lr in LR_GRID:
        m = iter47.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed=0, lr=lr)
        va_scores = iter47.sigmoid(m.predict(Xva))
        te_scores = iter47.sigmoid(m.predict(Xte))
        va = evaluate(uva, yva, va_scores)
        te = evaluate(ute, yte, te_scores)
        print(f"lr={lr:.4f}  valid={va['primary']:.5f}  test={te['primary']:.5f}", flush=True)
        results.append((lr, va['primary'], te['primary']))

    best = max(results, key=lambda r: r[1])
    print(f"\nbest: lr={best[0]:.4f}  valid={best[1]:.5f}  test={best[2]:.5f}")
    print(f"iter38 baseline (lr=0.001): see results above")

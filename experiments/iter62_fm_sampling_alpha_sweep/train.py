"""iter62: FM negative-sampling alpha resweep, single seed.

sampling_alpha=0.75 controls how strongly user-level decayed-positive-count
weighting shapes the BPR negative-sampling distribution. Unchanged since
iter38, never resystematically resweept. Structurally different from
lr/k (this is a data-sampling knob, not a model-capacity or optimizer
knob) so worth checking independently. Single seed=0 standalone metric
first; only worth a 5-seed confirmation + blend re-run if a real
(>0.0003 valid) gain shows up.
"""
import os, sys
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

    ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    results = []
    for sa in ALPHA_GRID:
        m = iter47.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed=0, sampling_alpha=sa)
        va_scores = iter47.sigmoid(m.predict(Xva))
        te_scores = iter47.sigmoid(m.predict(Xte))
        va = evaluate(uva, yva, va_scores)
        te = evaluate(ute, yte, te_scores)
        print(f"sampling_alpha={sa:.2f}  valid={va['primary']:.5f}  test={te['primary']:.5f}", flush=True)
        results.append((sa, va['primary'], te['primary']))

    best = max(results, key=lambda r: r[1])
    print(f"\nbest: sampling_alpha={best[0]:.2f}  valid={best[1]:.5f}  test={best[2]:.5f}")
    print(f"iter38 baseline (sampling_alpha=0.75): see results above")

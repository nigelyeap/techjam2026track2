"""Shared cache helpers for 6a (per-segment blend alpha).

Trains iter63's rate_only GBM (varying seed) and iter38's unchanged FM
5-seed ensemble (fixed, seeds 0-4 always -- same convention as every prior
GBM-only resweep in this project, e.g. iter55-iter63's blend.py scripts),
caching both to local .pkl files so repeated segment-alpha experiments
don't re-pay training cost. Also exposes each valid/test row's `tab` and
`decay_act_2.5` (decayed-activity) column for segmentation.
"""
import os, sys, importlib.util, pickle
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'iter27_triple_fusion'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'iter47_stacking_meta'))
from evaluate import evaluate  # noqa: E402
from data_ext import load_ext, encode_ext, HALFLIVES, TAB_HALFLIVES  # noqa: E402
import stack as iter47  # noqa: E402


def _load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t63 = _load_module(os.path.join(_REPO_ROOT, 'experiments', 'iter63_decay_tab_rate', 'train.py'), 'iterXUXIA1_t63_train')

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
FM_SEEDS = (0, 1, 2, 3, 4)


def minmax(x):
    return iter47.minmax(x)


def get_fm_ensemble(verbose=True):
    cache_path = os.path.join(_THIS_DIR, '.cache_fm_ensemble.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    splits = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc, dim = encode_ext(splits, feature_set=FEATURES, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                           alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    fm_va_scores, fm_te_scores = [], []
    for seed in FM_SEEDS:
        if verbose:
            print(f"  training FM seed {seed}...", flush=True)
        m = iter47.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed)
        fm_va_scores.append(iter47.sigmoid(m.predict(Xva)))
        fm_te_scores.append(iter47.sigmoid(m.predict(Xte)))
    fm_va_ens = np.mean(np.stack(fm_va_scores), axis=0)
    fm_te_ens = np.mean(np.stack(fm_te_scores), axis=0)
    result = {
        'fm_va_ens': fm_va_ens, 'fm_te_ens': fm_te_ens,
        'yva': np.asarray(yva), 'yte': np.asarray(yte),
        'uva': np.asarray(uva), 'ute': np.asarray(ute),
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    return result


def get_gbm(seed, verbose=True):
    cache_path = os.path.join(_THIS_DIR, f'.cache_gbm_seed{seed}.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    model, va, te, cache = t63.run(DATA_DIR, 'rate_only', seed=seed, verbose=verbose)
    dfs, y, u = cache
    gbm_va_raw = model.predict(dfs['valid'])
    gbm_te_raw = model.predict(dfs['test'])
    result = {
        'gbm_va_raw': gbm_va_raw, 'gbm_te_raw': gbm_te_raw,
        'va_metrics': va, 'te_metrics': te,
        'tab_va': dfs['valid']['tab'].astype(str).to_numpy(),
        'tab_te': dfs['test']['tab'].astype(str).to_numpy(),
        'act_va': dfs['valid']['decay_act_2.5'].to_numpy(dtype=np.float64),
        'act_te': dfs['test']['decay_act_2.5'].to_numpy(dtype=np.float64),
        'yva': np.asarray(y['valid']), 'yte': np.asarray(y['test']),
        'uva': np.asarray(u['valid']), 'ute': np.asarray(u['test']),
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    return result

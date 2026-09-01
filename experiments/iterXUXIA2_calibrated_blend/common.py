"""Shared cache helpers for 6b (rank-based / calibrated blending).

Extends iterXUXIA1's pattern to also cache train-split predictions (needed
to fit isotonic calibration without leakage: calibration must be fit on
train, never on valid/test).
"""
import os, sys, pickle
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


t63 = _load_module(os.path.join(_REPO_ROOT, 'experiments', 'iter63_decay_tab_rate', 'train.py'), 'iterXUXIA2_t63_train')

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
FM_SEEDS = (0, 1, 2, 3, 4)


def minmax(x):
    return iter47.minmax(x)


def get_fm_ensemble(verbose=True):
    cache_path = os.path.join(_THIS_DIR, '.cache_fm_ensemble_tr.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    splits = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc, dim = encode_ext(splits, feature_set=FEATURES, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                           alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    fm_tr_scores, fm_va_scores, fm_te_scores = [], [], []
    for seed in FM_SEEDS:
        if verbose:
            print(f"  training FM seed {seed}...", flush=True)
        m = iter47.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed)
        fm_tr_scores.append(iter47.sigmoid(m.predict(Xtr)))
        fm_va_scores.append(iter47.sigmoid(m.predict(Xva)))
        fm_te_scores.append(iter47.sigmoid(m.predict(Xte)))
    result = {
        'fm_tr_ens': np.mean(np.stack(fm_tr_scores), axis=0),
        'fm_va_ens': np.mean(np.stack(fm_va_scores), axis=0),
        'fm_te_ens': np.mean(np.stack(fm_te_scores), axis=0),
        'ytr': np.asarray(ytr), 'yva': np.asarray(yva), 'yte': np.asarray(yte),
        'utr': np.asarray(utr), 'uva': np.asarray(uva), 'ute': np.asarray(ute),
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    return result


def get_gbm(seed, verbose=True):
    cache_path = os.path.join(_THIS_DIR, f'.cache_gbm_seed{seed}_tr.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    model, va, te, cache = t63.run(DATA_DIR, 'rate_only', seed=seed, verbose=verbose)
    dfs, y, u = cache
    result = {
        'gbm_tr_raw': model.predict(dfs['train']),
        'gbm_va_raw': model.predict(dfs['valid']),
        'gbm_te_raw': model.predict(dfs['test']),
        'va_metrics': va, 'te_metrics': te,
        'tab_va': dfs['valid']['tab'].astype(str).to_numpy(),
        'tab_te': dfs['test']['tab'].astype(str).to_numpy(),
        'ytr': np.asarray(y['train']), 'yva': np.asarray(y['valid']), 'yte': np.asarray(y['test']),
        'utr': np.asarray(u['train']), 'uva': np.asarray(u['valid']), 'ute': np.asarray(u['test']),
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    return result

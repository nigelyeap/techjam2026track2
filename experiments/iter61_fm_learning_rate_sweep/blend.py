"""iter61 (part 4): blend the iter55 GBM with the FM ensemble retrained at
lr=0.0005 (this iteration's finding) instead of iter38's lr=0.001.
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

_THIS_MOD_DIR = _THIS_DIR


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gbm51 = _load_module(os.path.join(_THIS_DIR, '..', 'iter51_linear_tree', 'train.py'), 'iter61_gbm51_train')

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
SEEDS = (0, 1, 2, 3, 4)


def main():
    print("=== GBM (linear_tree=True, learning_rate=0.10, iter55, seed=0) ===", flush=True)
    gbm_model, gbm_va, gbm_te, cache = _gbm51.run(DATA_DIR, linear_tree=True, learning_rate=0.10, seed=0, verbose=False)
    dfs, y, u = cache
    gbm_va_raw = gbm_model.predict(dfs['valid'])
    gbm_te_raw = gbm_model.predict(dfs['test'])
    print(f"  GBM standalone: valid={gbm_va['primary']:.5f} test={gbm_te['primary']:.5f}", flush=True)

    print("\n=== FM 5-seed ensemble RETRAINED at lr=0.0005 (iter61 finding) ===", flush=True)
    splits = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc, dim = encode_ext(splits, feature_set=FEATURES, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                           alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    assert np.array_equal(np.asarray(yva), y['valid']), "FM/GBM valid label order mismatch"
    assert np.array_equal(np.asarray(yte), y['test']), "FM/GBM test label order mismatch"
    fm_va_scores, fm_te_scores = [], []
    for seed in SEEDS:
        m = iter47.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed, lr=0.0005)
        fm_va_scores.append(iter47.sigmoid(m.predict(Xva)))
        fm_te_scores.append(iter47.sigmoid(m.predict(Xte)))
    fm_va_ens = np.mean(np.stack(fm_va_scores), axis=0)
    fm_te_ens = np.mean(np.stack(fm_te_scores), axis=0)
    fm_va_m = evaluate(uva, yva, fm_va_ens)
    fm_te_m = evaluate(ute, yte, fm_te_ens)
    print(f"  FM ensemble standalone (lr=0.0005): valid={fm_va_m['primary']:.5f} test={fm_te_m['primary']:.5f}", flush=True)
    print(f"  [compare] FM ensemble (lr=0.001, iter55 blend): valid=0.63988 test=0.64187")

    gbm_va_n, gbm_te_n = iter47.minmax(gbm_va_raw), iter47.minmax(gbm_te_raw)

    print("\n=== alpha sweep (FM weight) on valid, direct metric ===", flush=True)
    best = {'alpha': None, 'valid': -1}
    for alpha in np.arange(0.0, 0.41, 0.02):
        va_pred = alpha * fm_va_ens + (1 - alpha) * gbm_va_n
        va_m = evaluate(uva, yva, va_pred)
        if va_m['primary'] > best['valid']:
            best = {'alpha': float(alpha), 'valid': float(va_m['primary'])}
    alpha = best['alpha']
    te_pred = alpha * fm_te_ens + (1 - alpha) * gbm_te_n
    te_m = evaluate(ute, yte, te_pred)
    print(f"  best alpha={alpha:.2f} -> valid={best['valid']:.5f} test={te_m['primary']:.5f}", flush=True)
    print(f"\n  [compare] iter55 blend (FM lr=0.001, alpha=0.10): valid=0.67451 test=0.65832")
    print(f"  [compare] iter61 blend (FM lr=0.0005, alpha={alpha:.2f}): valid={best['valid']:.5f} test={te_m['primary']:.5f}")

    import json
    with open(os.path.join(_THIS_MOD_DIR, 'blend_results.json'), 'w') as f:
        json.dump({
            'gbm_standalone': {'valid': float(gbm_va['primary']), 'test': float(gbm_te['primary'])},
            'fm_standalone_lr0005': {'valid': float(fm_va_m['primary']), 'test': float(fm_te_m['primary'])},
            'best_blend': {'alpha': alpha, 'valid': best['valid'], 'test': float(te_m['primary'])},
            'iter55_blend_for_comparison': {'alpha': 0.10, 'valid': 0.67451, 'test': 0.65832},
        }, f, indent=2)
    print("\nwrote blend_results.json")


if __name__ == '__main__':
    main()

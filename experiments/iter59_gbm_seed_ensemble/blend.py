"""iter59: GBM 5-seed PREDICTION-AVERAGED ensemble, blended with the FM
5-seed ensemble.

Every prior blend (iter44, iter51, iter55) used a SINGLE GBM seed (seed=0)
on the GBM side, even though the FM side has always been a genuine 5-seed
prediction-averaged ensemble. iter55's own 5-seed confirmation run showed
GBM seed-to-seed valid std of 0.00021 -- real but small noise. This checks
whether averaging the GBM's raw prediction scores across 5 seeds (the same
ensembling treatment already given to the FM side) reduces that noise and
produces a better blend than using a single arbitrary GBM seed.
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
import stack as iter47  # noqa: E402  (reuse train_one_fm, sigmoid, minmax)

_THIS_MOD_DIR = _THIS_DIR


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gbm51 = _load_module(os.path.join(_THIS_DIR, '..', 'iter51_linear_tree', 'train.py'), 'iter59_gbm51_train')

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
SEEDS = (0, 1, 2, 3, 4)


def main():
    print("=== GBM 5-seed prediction-averaged ensemble (linear_tree=True, lr=0.10) ===", flush=True)
    cache = None
    gbm_va_scores, gbm_te_scores = [], []
    for seed in SEEDS:
        model, va, te, cache = _gbm51.run(DATA_DIR, linear_tree=True, learning_rate=0.10,
                                           seed=seed, _cache=cache, verbose=False)
        dfs, y, u = cache
        va_raw = model.predict(dfs['valid'])
        te_raw = model.predict(dfs['test'])
        gbm_va_scores.append(iter47.minmax(va_raw))
        gbm_te_scores.append(iter47.minmax(te_raw))
        print(f"  seed={seed} valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)

    dfs, y, u = cache
    gbm_va_ens = np.mean(np.stack(gbm_va_scores), axis=0)
    gbm_te_ens = np.mean(np.stack(gbm_te_scores), axis=0)
    gbm_va_ens_m = evaluate(u['valid'], y['valid'], gbm_va_ens)
    gbm_te_ens_m = evaluate(u['test'], y['test'], gbm_te_ens)
    print(f"\n  GBM 5-seed ensemble standalone: valid={gbm_va_ens_m['primary']:.5f} test={gbm_te_ens_m['primary']:.5f}", flush=True)
    print(f"  [compare] GBM single seed=0 (iter55): valid=0.67052 test=0.65277")

    print("\n=== FM 5-seed ensemble (iter38 exact config, unchanged) ===", flush=True)
    splits = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc, dim = encode_ext(splits, feature_set=FEATURES, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                           alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    assert np.array_equal(np.asarray(yva), y['valid']), "FM/GBM valid label order mismatch"
    assert np.array_equal(np.asarray(yte), y['test']), "FM/GBM test label order mismatch"
    fm_va_scores, fm_te_scores = [], []
    for seed in SEEDS:
        m = iter47.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed)
        fm_va_scores.append(iter47.sigmoid(m.predict(Xva)))
        fm_te_scores.append(iter47.sigmoid(m.predict(Xte)))
    fm_va_ens = np.mean(np.stack(fm_va_scores), axis=0)
    fm_te_ens = np.mean(np.stack(fm_te_scores), axis=0)
    fm_va_m = evaluate(uva, yva, fm_va_ens)
    fm_te_m = evaluate(ute, yte, fm_te_ens)
    print(f"  FM ensemble standalone: valid={fm_va_m['primary']:.5f} test={fm_te_m['primary']:.5f}", flush=True)

    print("\n=== alpha sweep (FM weight) on valid, direct metric ===", flush=True)
    best = {'alpha': None, 'valid': -1}
    for alpha in np.arange(0.0, 0.41, 0.02):
        va_pred = alpha * fm_va_ens + (1 - alpha) * gbm_va_ens
        va_m = evaluate(uva, yva, va_pred)
        if va_m['primary'] > best['valid']:
            best = {'alpha': float(alpha), 'valid': float(va_m['primary'])}
    alpha = best['alpha']
    te_pred = alpha * fm_te_ens + (1 - alpha) * gbm_te_ens
    te_m = evaluate(ute, yte, te_pred)
    print(f"  best alpha={alpha:.2f} -> valid={best['valid']:.5f} test={te_m['primary']:.5f}", flush=True)
    print(f"\n  [compare] iter55 blend (single GBM seed, alpha=0.10): valid=0.67451 test=0.65832")
    print(f"  [compare] iter59 blend (5-seed GBM ensemble, alpha={alpha:.2f}): valid={best['valid']:.5f} test={te_m['primary']:.5f}")

    import json
    with open(os.path.join(_THIS_MOD_DIR, 'blend_results.json'), 'w') as f:
        json.dump({
            'gbm_5seed_ensemble': {'valid': float(gbm_va_ens_m['primary']), 'test': float(gbm_te_ens_m['primary'])},
            'fm_standalone': {'valid': float(fm_va_m['primary']), 'test': float(fm_te_m['primary'])},
            'best_blend': {'alpha': alpha, 'valid': best['valid'], 'test': float(te_m['primary'])},
            'iter55_blend_for_comparison': {'alpha': 0.10, 'valid': 0.67451, 'test': 0.65832},
        }, f, indent=2)
    print("\nwrote blend_results.json")


if __name__ == '__main__':
    main()

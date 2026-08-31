"""Trains iter63's GBM (rate_only) + iter38's 5-seed FM ensemble and saves
raw score arrays to disk. Split into its own process (no torch import) after
discovering that importing torch before LightGBM's native training in the
same process reliably segfaults (exit code 139) in this environment.
"""
import os, sys, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
from evaluate import evaluate  # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')

if __name__ == '__main__':
    print("=== iter63 GBM rate_only (seed 0) ===", flush=True)
    iter63_train = _load_module(os.path.join(_THIS_DIR, '..', 'iter63_decay_tab_rate', 'train.py'),
                                 'iter63_train_for_gen')
    gbm_model, gbm_va_m, gbm_te_m, cache_gbm = iter63_train.run(
        DATA_DIR, 'rate_only', linear_tree=True, num_leaves=2, learning_rate=0.10,
        n_estimators=500, min_child_samples=200, reg_lambda=1.0, seed=0, verbose=False)
    dfs_gbm, y_gbm, u_gbm = cache_gbm
    gbm_va_raw = gbm_model.predict(dfs_gbm['valid'])
    gbm_te_raw = gbm_model.predict(dfs_gbm['test'])
    print(f"  GBM standalone valid={gbm_va_m['primary']:.5f} test={gbm_te_m['primary']:.5f}", flush=True)

    print("=== iter38 FM 5-seed ensemble ===", flush=True)
    ms = _load_module(os.path.join(_REPO_ROOT, 'make_submission.py'), 'make_submission_for_gen')
    splits = ms.load_ext(DATA_DIR, halflives=ms.HALFLIVES, tab_halflives=ms.TAB_HALFLIVES)
    enc, dim = ms.encode_ext(splits, feature_set=ms.FEATURES, halflives=ms.HALFLIVES,
                              tab_halflives=ms.TAB_HALFLIVES, alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    assert np.array_equal(np.asarray(yva), y_gbm['valid'])
    assert np.array_equal(np.asarray(yte), y_gbm['test'])
    fm_va_scores, fm_te_scores = [], []
    for seed in (0, 1, 2, 3, 4):
        m = ms.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed)
        fm_va_scores.append(sigmoid(m.predict(Xva)))
        fm_te_scores.append(sigmoid(m.predict(Xte)))
    fm_va_ens = np.mean(np.stack(fm_va_scores), axis=0)
    fm_te_ens = np.mean(np.stack(fm_te_scores), axis=0)
    fm_va_m = evaluate(uva, yva, fm_va_ens)
    fm_te_m = evaluate(ute, yte, fm_te_ens)
    print(f"  FM standalone valid={fm_va_m['primary']:.5f} test={fm_te_m['primary']:.5f}", flush=True)

    np.savez(os.path.join(_THIS_DIR, 'iter63_scores.npz'),
              gbm_va_raw=gbm_va_raw, gbm_te_raw=gbm_te_raw,
              fm_va_ens=fm_va_ens, fm_te_ens=fm_te_ens,
              y_va=np.asarray(yva), y_te=np.asarray(yte),
              u_va=np.asarray(uva, dtype=object), u_te=np.asarray(ute, dtype=object))
    print("SAVED iter63_scores.npz", flush=True)

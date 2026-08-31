"""Quick blend-level check: does iter64's SASRec history encoder add ANY
incremental signal on top of the current best (iter63) blend, before
investing in 5-seed confirmation? Trains (a) iter63's GBM rate_only variant,
(b) iter38's unchanged 5-seed FM ensemble, (c) this module's SASRec (seed
0), forms the current iter63 blend (alpha=0.14) from (a)+(b), then sweeps a
second weight `beta` blending in the SASRec score on top of that.
"""
import os, sys, json, importlib.util, time
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
from evaluate import evaluate  # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
ALPHA_BLEND = 0.14  # iter63's confirmed GBM/FM blend weight, unchanged

if __name__ == '__main__':
    t0 = time.time()
    torch.set_num_threads(4)

    print("=== (a) iter63 GBM rate_only (seed 0) ===", flush=True)
    iter63_train = _load_module(os.path.join(_THIS_DIR, '..', 'iter63_decay_tab_rate', 'train.py'),
                                 'iter63_train_for_blendcheck')
    gbm_model, gbm_va_m, gbm_te_m, cache_gbm = iter63_train.run(
        DATA_DIR, 'rate_only', linear_tree=True, num_leaves=2, learning_rate=0.10,
        n_estimators=500, min_child_samples=200, reg_lambda=1.0, seed=0, verbose=False)
    dfs_gbm, y_gbm, u_gbm = cache_gbm
    gbm_va_raw = gbm_model.predict(dfs_gbm['valid'])
    gbm_te_raw = gbm_model.predict(dfs_gbm['test'])
    print(f"  GBM standalone valid={gbm_va_m['primary']:.5f} test={gbm_te_m['primary']:.5f} ({time.time()-t0:.0f}s)", flush=True)

    print("=== (b) iter38 FM 5-seed ensemble ===", flush=True)
    ms = _load_module(os.path.join(_REPO_ROOT, 'make_submission.py'), 'make_submission_for_blendcheck')
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
    print(f"  FM standalone valid={fm_va_m['primary']:.5f} test={fm_te_m['primary']:.5f} ({time.time()-t0:.0f}s)", flush=True)

    gbm_va_norm, gbm_te_norm = minmax(gbm_va_raw), minmax(gbm_te_raw)
    iter63_va = ALPHA_BLEND * fm_va_ens + (1 - ALPHA_BLEND) * gbm_va_norm
    iter63_te = ALPHA_BLEND * fm_te_ens + (1 - ALPHA_BLEND) * gbm_te_norm
    iter63_va_m = evaluate(uva, yva, iter63_va)
    iter63_te_m = evaluate(ute, yte, iter63_te)
    print(f"  iter63 blend (current best) valid={iter63_va_m['primary']:.5f} test={iter63_te_m['primary']:.5f}", flush=True)

    print("=== (c) iter64 SASRec (seed 0) ===", flush=True)
    tr = _load_module(os.path.join(_THIS_DIR, 'train.py'), 'iter64_train_for_blendcheck')
    sas_model, sas_va_m, sas_te_m, sas_cache = tr.run(DATA_DIR, epochs=25, patience=4, bs=4096, seed=0, verbose=False)
    sas_va_scores, sas_te_scores, y_va_sas, u_va_sas, y_te_sas, u_te_sas = sas_cache
    assert np.array_equal(np.asarray(y_va_sas), y_gbm['valid'])
    assert np.array_equal(np.asarray(y_te_sas), y_gbm['test'])
    print(f"  SASRec standalone valid={sas_va_m['primary']:.5f} test={sas_te_m['primary']:.5f} ({time.time()-t0:.0f}s)", flush=True)

    sas_va_norm, sas_te_norm = minmax(sas_va_scores), minmax(sas_te_scores)

    print("\n=== sweeping beta: (1-beta)*iter63_blend + beta*SASRec ===", flush=True)
    iter63_va_norm, iter63_te_norm = minmax(iter63_va), minmax(iter63_te)
    results = []
    for beta in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
        cva = (1 - beta) * iter63_va_norm + beta * sas_va_norm
        cte = (1 - beta) * iter63_te_norm + beta * sas_te_norm
        vm = evaluate(uva, yva, cva)
        tm = evaluate(ute, yte, cte)
        results.append({'beta': beta, 'valid': vm['primary'], 'test': tm['primary']})
        print(f"  beta={beta:.2f}  valid={vm['primary']:.5f}  test={tm['primary']:.5f}", flush=True)

    out = {
        'gbm_standalone': gbm_va_m, 'gbm_standalone_test': gbm_te_m,
        'fm_standalone': fm_va_m, 'fm_standalone_test': fm_te_m,
        'iter63_blend': {'valid': iter63_va_m['primary'], 'test': iter63_te_m['primary']},
        'sasrec_standalone': {'valid': sas_va_m['primary'], 'test': sas_te_m['primary']},
        'beta_sweep': results,
    }
    with open(os.path.join(_THIS_DIR, 'blend_check_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nDONE ({time.time()-t0:.0f}s total)", flush=True)

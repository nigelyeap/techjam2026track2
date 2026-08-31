"""Shared score generator for iter65 (segment blend alpha) and iter66
(rank/calibrated blend) -- independent re-verification of findings reported
by a teammate's parallel Claude instance (see experiments/LEDGER.md, the
"Parallel track -- teammate (Xuxia)" section). Trains iter63's winning GBM
(rate_only variant) at 5 seeds + the unchanged iter38 FM 5-seed ensemble,
saving raw scores plus per-row `tab` and decayed-activity metadata needed
for segment blending.
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


DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
SEEDS = (0, 1, 2, 3, 4)


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


if __name__ == '__main__':
    t = _load_module(os.path.join(_REPO_ROOT, 'experiments', 'iter63_decay_tab_rate', 'train.py'),
                      'iter63_train_for_65')
    print("=== preparing iter63 rate_only features (cached) ===", flush=True)
    dfs, y, u = t.prepare(DATA_DIR, 'rate_only')
    splits = t._de.load_ext(DATA_DIR)
    origidx = {name: np.array([x[t._de.IDX['orig_idx']] for x in splits[name]]) for name in ('train', 'valid', 'test')}

    print("=== training 5-seed GBM ===", flush=True)
    gbm_va_raw, gbm_te_raw = [], []
    for seed in SEEDS:
        model, va_m, te_m, _ = t.run(DATA_DIR, 'rate_only', seed=seed, _cache=(dfs, y, u), verbose=True)
        gbm_va_raw.append(model.predict(dfs['valid']))
        gbm_te_raw.append(model.predict(dfs['test']))
    gbm_va_raw = np.stack(gbm_va_raw)  # (5, N_valid)
    gbm_te_raw = np.stack(gbm_te_raw)

    print("\n=== training FM 5-seed ensemble (iter38 exact config, unchanged) ===", flush=True)
    ms = _load_module(os.path.join(_REPO_ROOT, 'make_submission.py'), 'make_submission_for_65')
    splits_fm = ms.load_ext(DATA_DIR, halflives=ms.HALFLIVES, tab_halflives=ms.TAB_HALFLIVES)
    enc, dim = ms.encode_ext(splits_fm, feature_set=ms.FEATURES, halflives=ms.HALFLIVES,
                              tab_halflives=ms.TAB_HALFLIVES, alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    assert np.array_equal(np.asarray(yva), y['valid']), "FM/GBM valid label order mismatch"
    assert np.array_equal(np.asarray(yte), y['test']), "FM/GBM test label order mismatch"

    fm_va_scores, fm_te_scores = [], []
    for seed in SEEDS:
        print(f"  training FM seed {seed}...", flush=True)
        m = ms.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits_fm['train'], dim, seed)
        fm_va_scores.append(sigmoid(m.predict(Xva)))
        fm_te_scores.append(sigmoid(m.predict(Xte)))
    fm_va_ens = np.mean(np.stack(fm_va_scores), axis=0)
    fm_te_ens = np.mean(np.stack(fm_te_scores), axis=0)
    print(f"  FM ensemble standalone: valid={evaluate(uva, yva, fm_va_ens)['primary']:.5f} "
          f"test={evaluate(ute, yte, fm_te_ens)['primary']:.5f}", flush=True)

    print("\n=== sanity: seed-0 blend should reproduce iter63 (valid=0.67606 test=0.65955) ===", flush=True)
    def minmax(x):
        x = np.asarray(x, dtype=np.float64); lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)
    ALPHA_BLEND = 0.14
    blend_va = ALPHA_BLEND * fm_va_ens + (1 - ALPHA_BLEND) * minmax(gbm_va_raw[0])
    blend_te = ALPHA_BLEND * fm_te_ens + (1 - ALPHA_BLEND) * minmax(gbm_te_raw[0])
    print(f"  reproduced: valid={evaluate(uva, yva, blend_va)['primary']:.5f} "
          f"test={evaluate(ute, yte, blend_te)['primary']:.5f}", flush=True)

    out_path = os.path.join(_THIS_DIR, 'scores_5seed.npz')
    np.savez(out_path,
             gbm_va_raw=gbm_va_raw, gbm_te_raw=gbm_te_raw,
             fm_va_ens=fm_va_ens, fm_te_ens=fm_te_ens,
             y_va=np.asarray(y['valid']), y_te=np.asarray(y['test']),
             u_va=np.asarray(u['valid'], dtype=object), u_te=np.asarray(u['test'], dtype=object),
             tab_va=dfs['valid']['tab'].astype(str).to_numpy(), tab_te=dfs['test']['tab'].astype(str).to_numpy(),
             activity_va=dfs['valid']['decay_act_2.5'].to_numpy(), activity_te=dfs['test']['decay_act_2.5'].to_numpy(),
             origidx_va=origidx['valid'], origidx_te=origidx['test'], origidx_tr=origidx['train'])
    print(f"SAVED {out_path}", flush=True)

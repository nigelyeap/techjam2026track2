"""iter66 part 2: independent re-verification of Xuxia's isotonic-
calibration finding -- fit IsotonicRegression on (GBM raw score, label) on
TRAIN only, apply to valid/test, use calibrated score in place of the
minmax-normalized score in the alpha=0.14 blend. Also checks GBM standalone
under calibration (their reported collapse: 0.67168 -> 0.54189).
"""
import os, sys, importlib.util
import numpy as np
from sklearn.isotonic import IsotonicRegression

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
_ITER65_DIR = os.path.join(_REPO_ROOT, 'experiments', 'iter65_segment_blend')
sys.path.insert(0, _REPO_ROOT)
from evaluate import evaluate  # noqa: E402

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
ALPHA_GLOBAL = 0.14


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def minmax(x):
    x = np.asarray(x, dtype=np.float64); lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


if __name__ == '__main__':
    t = _load_module(os.path.join(_REPO_ROOT, 'experiments', 'iter63_decay_tab_rate', 'train.py'),
                      'iter63_train_for_66_iso')
    dfs, y, u = t.prepare(DATA_DIR, 'rate_only')
    model, va_m, te_m, _ = t.run(DATA_DIR, 'rate_only', seed=0, _cache=(dfs, y, u), verbose=True)
    gbm_tr_raw = model.predict(dfs['train'])
    gbm_va_raw = model.predict(dfs['valid'])
    gbm_te_raw = model.predict(dfs['test'])
    print(f"GBM standalone (raw, minmax): valid={va_m['primary']:.5f} test={te_m['primary']:.5f}")
    print(f"unique raw score count: train={len(np.unique(gbm_tr_raw))} valid={len(np.unique(gbm_va_raw))} test={len(np.unique(gbm_te_raw))}")

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(gbm_tr_raw, y['train'])
    gbm_va_iso = iso.predict(gbm_va_raw)
    gbm_te_iso = iso.predict(gbm_te_raw)
    print(f"isotonic-calibrated unique levels: valid={len(np.unique(gbm_va_iso))} test={len(np.unique(gbm_te_iso))}")

    gbm_standalone_iso_va = evaluate(u['valid'], y['valid'], gbm_va_iso)['primary']
    gbm_standalone_iso_te = evaluate(u['test'], y['test'], gbm_te_iso)['primary']
    print(f"GBM standalone (isotonic-calibrated): valid={gbm_standalone_iso_va:.5f} test={gbm_standalone_iso_te:.5f}")

    d = np.load(os.path.join(_ITER65_DIR, 'scores_5seed.npz'), allow_pickle=True)
    fm_va, fm_te = d['fm_va_ens'], d['fm_te_ens']
    y_va, y_te, u_va, u_te = d['y_va'], d['y_te'], d['u_va'], d['u_te']

    global_va = ALPHA_GLOBAL * fm_va + (1 - ALPHA_GLOBAL) * minmax(gbm_va_raw)
    global_te = ALPHA_GLOBAL * fm_te + (1 - ALPHA_GLOBAL) * minmax(gbm_te_raw)
    gva_m = evaluate(u_va, y_va, global_va)['primary']
    gte_m = evaluate(u_te, y_te, global_te)['primary']

    iso_blend_va = ALPHA_GLOBAL * fm_va + (1 - ALPHA_GLOBAL) * gbm_va_iso
    iso_blend_te = ALPHA_GLOBAL * fm_te + (1 - ALPHA_GLOBAL) * gbm_te_iso
    iva_m = evaluate(u_va, y_va, iso_blend_va)['primary']
    ite_m = evaluate(u_te, y_te, iso_blend_te)['primary']

    print(f"\ncurrent blend (minmax):    valid={gva_m:.5f} test={gte_m:.5f}")
    print(f"isotonic-calibrated blend: valid={iva_m:.5f} test={ite_m:.5f}  "
          f"delta valid={iva_m-gva_m:+.5f} test={ite_m-gte_m:+.5f}")

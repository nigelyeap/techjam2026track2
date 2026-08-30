"""iter47: replace the fixed alpha=0.1 linear blend (iter44) with a learned
logistic-regression stack over the base models' scores.

iter44's blend picks one scalar (alpha) by a grid sweep on valid. This
generalizes that to a small (3-4 parameter) logistic regression fit on
valid: intercept + one coefficient per base model's min-max-normalized
score, sigmoid-mapped to predict `long_view` directly, then used as the
final ranking score. This is fit with plain gradient descent (no new
dependency) rather than a library, since the parameter count is tiny.

Fitting on valid and then reporting valid primary is exactly the same
selection procedure iter44's alpha sweep already used (search over a
handful of parameters, score on valid, pick the best) -- not "peeking" at
anything test doesn't already tolerate under this project's convention.
Test is still checked exactly once, at the end, never used to pick
between configs.
"""
import os, sys, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'iter27_triple_fusion'))
from evaluate import evaluate  # noqa: E402
from data_ext import load_ext, encode_ext, compute_final_decayed_pos, HALFLIVES, TAB_HALFLIVES  # noqa: E402
from train import build_pos_neg_index, sample_pairs, bpr_step  # noqa: E402
from baseline import FM  # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gbm_train = _load_module(os.path.join(_REPO_ROOT, 'experiments', 'iter44_gbm_native_features', 'train.py'), 'iter47_gbm_train')
_cb_train = _load_module(os.path.join(_REPO_ROOT, 'experiments', 'iter45_catboost_native', 'train.py'), 'iter47_cb_train')

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
SEEDS = (0, 1, 2, 3, 4)


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits_train, dim, seed,
                  k=16, lr=0.001, epochs=40, bs=8192, patience=4,
                  sampling_alpha=0.75, decay_halflife=3):
    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs)))
    decayed_pos_dict = compute_final_decayed_pos(splits_train, halflife=decay_halflife)
    decayed_arr = np.array([decayed_pos_dict.get(u, 0.0) for u in eligible], dtype=np.float64)
    weights = decayed_arr ** sampling_alpha
    user_cumw = np.cumsum(weights); user_totalw = user_cumw[-1]
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            Xpos_rows, Xneg_rows = sample_pairs(rng, n_users, bs, pos_flat, pos_start, pos_len,
                                                 neg_flat, neg_start, neg_len,
                                                 user_cumw=user_cumw, user_totalw=user_totalw)
            bpr_step(m, Xtr[Xpos_rows], Xtr[Xneg_rows])
        va = evaluate(uva, yva, m.predict(Xva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    return m


def fit_logistic_gd(X, y, l2=1e-3, lr=0.5, epochs=2000):
    """X: (n, d) already-normalized features. y: (n,) in {0,1}. Plain full-batch GD."""
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        grad_z = (p - y) / n
        gw = X.T @ grad_z + l2 * w
        gb = grad_z.sum()
        w -= lr * gw
        b -= lr * gb
    return w, b


def main():
    print("=== base model 1/3: GBM (LightGBM, iter44 native) ===", flush=True)
    dfs_gbm, y_gbm, u_gbm = _gbm_train.prepare(DATA_DIR)
    gbm_model, gbm_va, gbm_te, _ = _gbm_train.run(DATA_DIR, num_leaves=2, learning_rate=0.05,
                                                   n_estimators=500, min_child_samples=200,
                                                   reg_lambda=1.0, _cache=(dfs_gbm, y_gbm, u_gbm))
    gbm_va_raw = gbm_model.predict(dfs_gbm['valid'])
    gbm_te_raw = gbm_model.predict(dfs_gbm['test'])
    print(f"  GBM standalone: valid={gbm_va['primary']:.5f} test={gbm_te['primary']:.5f}", flush=True)

    print("\n=== base model 2/3: CatBoost (iter45 native) ===", flush=True)
    cb_model, cb_va, cb_te, _ = _cb_train.run(DATA_DIR, depth=2, iterations=500, learning_rate=0.1,
                                               l2_leaf_reg=3.0, loss_function='YetiRank',
                                               _cache=(dfs_gbm, y_gbm, u_gbm))
    Xva_cb = _cb_train._to_catboost_frame(dfs_gbm['valid'])
    Xte_cb = _cb_train._to_catboost_frame(dfs_gbm['test'])
    cb_va_raw = cb_model.predict(Xva_cb)
    cb_te_raw = cb_model.predict(Xte_cb)
    print(f"  CatBoost standalone: valid={cb_va['primary']:.5f} test={cb_te['primary']:.5f}", flush=True)

    print("\n=== base model 3/3: FM 5-seed ensemble (iter38 exact config) ===", flush=True)
    splits = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc, dim = encode_ext(splits, feature_set=FEATURES, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                           alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    assert np.array_equal(np.asarray(yva), y_gbm['valid']), "FM/GBM valid label order mismatch"
    assert np.array_equal(np.asarray(yte), y_gbm['test']), "FM/GBM test label order mismatch"
    fm_va_scores, fm_te_scores = [], []
    for seed in SEEDS:
        m = train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed)
        fm_va_scores.append(sigmoid(m.predict(Xva)))
        fm_te_scores.append(sigmoid(m.predict(Xte)))
    fm_va_ens = np.mean(np.stack(fm_va_scores), axis=0)
    fm_te_ens = np.mean(np.stack(fm_te_scores), axis=0)
    fm_va_metrics = evaluate(uva, yva, fm_va_ens)
    fm_te_metrics = evaluate(ute, yte, fm_te_ens)
    print(f"  FM ensemble standalone: valid={fm_va_metrics['primary']:.5f} test={fm_te_metrics['primary']:.5f}", flush=True)

    # baseline: iter44's fixed alpha=0.1 (FM+GBM only, no CatBoost)
    gbm_va_n, gbm_te_n = minmax(gbm_va_raw), minmax(gbm_te_raw)
    base_va = 0.1 * fm_va_ens + 0.9 * gbm_va_n
    base_te = 0.1 * fm_te_ens + 0.9 * gbm_te_n
    base_va_m = evaluate(uva, yva, base_va)
    base_te_m = evaluate(ute, yte, base_te)
    print(f"\n[baseline] iter44 fixed-alpha blend (FM+GBM, alpha=0.1): "
          f"valid={base_va_m['primary']:.5f} test={base_te_m['primary']:.5f}", flush=True)

    cb_va_n, cb_te_n = minmax(cb_va_raw), minmax(cb_te_raw)

    y_va_arr = np.asarray(yva, dtype=np.float64)
    y_te_arr = np.asarray(yte, dtype=np.float64)

    results = {}
    configs = {
        '2way_FM_GBM': (np.stack([fm_va_ens, gbm_va_n], axis=1), np.stack([fm_te_ens, gbm_te_n], axis=1)),
        '3way_FM_GBM_CB': (np.stack([fm_va_ens, gbm_va_n, cb_va_n], axis=1),
                            np.stack([fm_te_ens, gbm_te_n, cb_te_n], axis=1)),
    }
    for name, (Xva_stack, Xte_stack) in configs.items():
        w, b = fit_logistic_gd(Xva_stack, y_va_arr)
        va_pred = sigmoid(Xva_stack @ w + b)
        te_pred = sigmoid(Xte_stack @ w + b)
        va_m = evaluate(uva, yva, va_pred)
        te_m = evaluate(ute, yte, te_pred)
        print(f"[stack:{name}] w={w} b={b:.4f} -> valid={va_m['primary']:.5f} test={te_m['primary']:.5f}", flush=True)
        results[name] = {'w': w.tolist(), 'b': float(b), 'valid': float(va_m['primary']), 'test': float(te_m['primary'])}

    # Direct grid search on the actual primary metric (not a BCE proxy loss) --
    # tests whether the logistic stack's underperformance vs. the fixed-alpha
    # baseline is specifically because BCE is the wrong objective for a
    # group-wise ranking metric, or whether 3 base models just don't have
    # more to offer than 2 regardless of how the weights are chosen.
    print("\n=== direct grid search on primary metric (3-way weights) ===", flush=True)
    best = {'valid': -1}
    grid_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    for w_fm in grid_vals:
        for w_cb in [0.0, 0.05, 0.1, 0.15, 0.2]:
            w_gbm = 1.0 - w_fm - w_cb
            if w_gbm < 0.3:
                continue
            va_pred = w_fm * fm_va_ens + w_gbm * gbm_va_n + w_cb * cb_va_n
            va_m = evaluate(uva, yva, va_pred)
            if va_m['primary'] > best['valid']:
                best = {'w_fm': w_fm, 'w_gbm': w_gbm, 'w_cb': w_cb, 'valid': float(va_m['primary'])}
    te_pred = best['w_fm'] * fm_te_ens + best['w_gbm'] * gbm_te_n + best['w_cb'] * cb_te_n
    te_m = evaluate(ute, yte, te_pred)
    best['test'] = float(te_m['primary'])
    print(f"  best direct-grid weights: {best}", flush=True)
    results['direct_grid_3way'] = best

    import json
    with open(os.path.join(_THIS_DIR, 'results.json'), 'w') as f:
        json.dump({
            'baseline_fixed_alpha': {'valid': float(base_va_m['primary']), 'test': float(base_te_m['primary'])},
            'gbm_standalone': {'valid': float(gbm_va['primary']), 'test': float(gbm_te['primary'])},
            'catboost_standalone': {'valid': float(cb_va['primary']), 'test': float(cb_te['primary'])},
            'fm_standalone': {'valid': float(fm_va_metrics['primary']), 'test': float(fm_te_metrics['primary'])},
            'stacks': results,
        }, f, indent=2)
    print("\nwrote results.json")


if __name__ == '__main__':
    main()

"""iter44c: score-level blend of the native-feature LightGBM ranker with the
current-best FM+BPR 5-seed ensemble (iter38).

Unlike iter41's blend (LightGBM trained on FM's own bucketed features,
which failed -- monotonic-in-FM-weight, no real diversity), this LightGBM
sees a structurally different representation of the same underlying signal
(raw floats vs FM's embedding buckets), so its errors have a better chance
of being genuinely complementary rather than a strict subset of FM's.
"""
import os, sys, json, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402

_ITER27_DIR = os.path.join(_THIS_DIR, '..', 'iter27_triple_fusion')


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# load iter44's own train.py by explicit path BEFORE iter27_dir goes on
# sys.path -- both directories have modules that could collide.
gbm_train = _load_module(os.path.join(_THIS_DIR, 'train.py'), 'iter44_train')

sys.path.insert(0, _ITER27_DIR)  # need iter27's data_ext/train/baseline for the FM retrain

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
ITER24_FEATS = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def main():
    print("=== training native-feature LightGBM ranker (num_leaves=2, the sweep2 winner) ===", flush=True)
    model, va_gbm, te_gbm, (dfs, y, u) = gbm_train.run(
        DATA_DIR, verbose=True, num_leaves=2, learning_rate=0.05,
        n_estimators=500, min_child_samples=200, reg_lambda=1.0)
    # gbm_train.run predicts on dfs['valid']/dfs['test'], which prepare() never
    # reordered -- these scores are already in encode_ext's original row order.
    gbm_va_scores = model.predict(dfs['valid'])
    gbm_te_scores = model.predict(dfs['test'])
    uva, yva = u['valid'], y['valid']
    ute, yte = u['test'], y['test']

    print("\n=== retraining FM 5-seed ensemble (iter38 exact method) ===", flush=True)
    from data_ext import load_ext, encode_ext, HALFLIVES, TAB_HALFLIVES, compute_final_decayed_pos
    from train import build_pos_neg_index, sample_pairs, bpr_step  # iter27's train.py
    from baseline import FM  # repo root (already on sys.path)

    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc_fm, dim_fm = encode_ext(SPLITS_CACHE, feature_set=ITER24_FEATS, halflives=HALFLIVES,
                                 tab_halflives=TAB_HALFLIVES, alpha=0.5, n_buckets=20)
    Xtr_fm, ytr_fm, utr_fm = enc_fm['train']
    Xva_fm, yva_fm, uva_fm = enc_fm['valid']
    Xte_fm, yte_fm, ute_fm = enc_fm['test']

    assert np.array_equal(np.asarray(yva_fm), np.asarray(yva)), "valid label order mismatch between FM and GBM pipelines"
    assert np.array_equal(np.asarray(yte_fm), np.asarray(yte)), "test label order mismatch between FM and GBM pipelines"

    def train_one_fm(seed, k=16, lr=0.001, epochs=40, bs=8192, patience=4, sampling_alpha=0.75, decay_halflife=3):
        eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = build_pos_neg_index(ytr_fm, utr_fm)
        n_users = len(eligible)
        steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs)))
        decayed_pos_dict = compute_final_decayed_pos(SPLITS_CACHE['train'], halflife=decay_halflife)
        decayed_arr = np.array([decayed_pos_dict.get(u, 0.0) for u in eligible], dtype=np.float64)
        weights = decayed_arr ** sampling_alpha
        user_cumw = np.cumsum(weights); user_totalw = user_cumw[-1]
        m = FM(dim_fm, k=k, lr=lr, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            for _ in range(steps_per_epoch):
                Xpos_rows, Xneg_rows = sample_pairs(rng, n_users, bs, pos_flat, pos_start, pos_len,
                                                     neg_flat, neg_start, neg_len,
                                                     user_cumw=user_cumw, user_totalw=user_totalw)
                bpr_step(m, Xtr_fm[Xpos_rows], Xtr_fm[Xneg_rows])
            v = evaluate(uva_fm, yva_fm, m.predict(Xva_fm))
            if v['primary'] > best + 1e-5:
                best, bad = v['primary'], 0
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            else:
                bad += 1
                if bad >= patience:
                    break
        m.V, m.W, m.b = best_state
        return m

    fm_va_scores, fm_te_scores = [], []
    for seed in range(5):
        m = train_one_fm(seed)
        fm_va_scores.append(m.predict(Xva_fm))
        fm_te_scores.append(m.predict(Xte_fm))
        print(f"  fm seed {seed} done", flush=True)

    fm_va_ens = np.mean(np.stack([sigmoid(s) for s in fm_va_scores]), axis=0)
    fm_te_ens = np.mean(np.stack([sigmoid(s) for s in fm_te_scores]), axis=0)
    r_fm_va = evaluate(uva_fm, yva_fm, fm_va_ens)
    r_fm_te = evaluate(ute_fm, yte_fm, fm_te_ens)
    print(f"\nFM ensemble (sanity check vs iter38): valid={r_fm_va['primary']:.5f} test={r_fm_te['primary']:.5f}")

    def minmax(x):
        x = np.asarray(x, dtype=np.float64)
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    gbm_va_norm = minmax(gbm_va_scores)
    gbm_te_norm = minmax(gbm_te_scores)

    print("\n=== blend sweep (alpha = weight on FM ensemble) ===")
    best_alpha, best_valid = None, -1
    sweep_results = []
    for alpha in np.arange(0.0, 1.01, 0.1):
        blend_va = alpha * fm_va_ens + (1 - alpha) * gbm_va_norm
        r = evaluate(uva, yva, blend_va)
        sweep_results.append((round(alpha, 2), float(r['primary'])))
        print(f"  alpha={alpha:.1f}  valid_primary={r['primary']:.5f}")
        if r['primary'] > best_valid:
            best_valid, best_alpha = r['primary'], alpha

    blend_te = best_alpha * fm_te_ens + (1 - best_alpha) * gbm_te_norm
    r_te_blend = evaluate(ute, yte, blend_te)
    print(f"\nbest alpha (on valid) = {best_alpha:.1f}  valid={best_valid:.5f}  test={r_te_blend['primary']:.5f}")
    print(f"reference: FM-only ensemble valid={r_fm_va['primary']:.5f} test={r_fm_te['primary']:.5f}")
    print(f"reference: native-feature LightGBM-only valid={va_gbm['primary']:.5f} test={te_gbm['primary']:.5f}")

    out = {
        'gbm_only': {'valid': float(va_gbm['primary']), 'test': float(te_gbm['primary'])},
        'fm_ensemble_only': {'valid': float(r_fm_va['primary']), 'test': float(r_fm_te['primary'])},
        'sweep': sweep_results,
        'best_alpha': float(best_alpha),
        'blend_at_best_alpha': {'valid': float(best_valid), 'test': float(r_te_blend['primary'])},
    }
    with open(os.path.join(_THIS_DIR, 'blend_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nwrote blend_results.json")


if __name__ == '__main__':
    main()

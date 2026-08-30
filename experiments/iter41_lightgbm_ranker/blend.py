"""iter41b: score-level blend of the LightGBM lambdarank ranker (different
model family: trees, NDCG-native objective) with the current-best FM+BPR
5-seed ensemble (iter38). Different model families tend to make different
mistakes, unlike stacking more FM variants (which kept saturating in
iter34/40) -- this tests whether that diversity is real here.

Reuses iter38's driver.py `train_one` verbatim (via importlib) to retrain
the exact same 5-seed FM ensemble, and iter41's train.py `run()` for the
LightGBM ranker -- both evaluated in encode_ext's ORIGINAL row order so
per-row scores align for blending.
"""
import os, sys, json, importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402

_ITER27_DIR = os.path.join(_THIS_DIR, '..', 'iter27_triple_fusion')
_ITER38_DIR = os.path.join(_THIS_DIR, '..', 'iter38_seed_ensemble')


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# load iter41's own train.py by explicit path BEFORE iter27_dir goes on
# sys.path -- both directories have a train.py, and a plain `import train`
# would resolve to whichever is earlier on sys.path.
lgb_train = _load_module(os.path.join(_THIS_DIR, 'train.py'), 'iter41_train')

sys.path.insert(0, _ITER27_DIR)  # iter38's driver.py does `from data_ext import ...` / `from train import ...`

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
ITER24_FEATS = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def main():
    print("=== training LightGBM ranker ===", flush=True)
    model, va_lgb, te_lgb, _cache, (lgb_va_scores, lgb_te_scores) = lgb_train.run(DATA_DIR, verbose=True)
    data, enc = _cache
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']

    print("\n=== retraining FM 5-seed ensemble (iter38 exact method) ===", flush=True)
    from data_ext import load_ext, encode_ext, HALFLIVES, TAB_HALFLIVES
    driver38 = _load_module(os.path.join(_ITER38_DIR, 'driver.py').replace('driver.py', 'driver.py'), 'iter38_driver_unused')
    # driver.py runs its whole pipeline at import time under `if __name__`, so
    # instead pull just the pieces we need (train_one, FM) via direct imports.
    from train import build_pos_neg_index, sample_pairs, bpr_step  # iter27's train.py
    from baseline import FM  # repo root (already on sys.path via lgb_train's insert)

    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc_fm, dim_fm = encode_ext(SPLITS_CACHE, feature_set=ITER24_FEATS, halflives=HALFLIVES,
                                 tab_halflives=TAB_HALFLIVES, alpha=0.5, n_buckets=20)
    Xtr_fm, ytr_fm, utr_fm = enc_fm['train']
    Xva_fm, yva_fm, uva_fm = enc_fm['valid']
    Xte_fm, yte_fm, ute_fm = enc_fm['test']

    assert np.array_equal(np.asarray(yva_fm), np.asarray(yva)), "valid label order mismatch between FM and LightGBM pipelines"
    assert np.array_equal(np.asarray(yte_fm), np.asarray(yte)), "test label order mismatch between FM and LightGBM pipelines"

    from data_ext import compute_final_decayed_pos

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

    # normalize LightGBM scores to [0,1] (min-max) for blending with FM's sigmoid scores
    def minmax(x):
        x = np.asarray(x, dtype=np.float64)
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    lgb_va_norm = minmax(lgb_va_scores)
    lgb_te_norm = minmax(lgb_te_scores)

    print("\n=== blend sweep (alpha = weight on FM ensemble) ===")
    best_alpha, best_valid = None, -1
    sweep_results = []
    for alpha in np.arange(0.0, 1.01, 0.1):
        blend_va = alpha * fm_va_ens + (1 - alpha) * lgb_va_norm
        r = evaluate(uva, yva, blend_va)
        sweep_results.append((round(alpha, 2), float(r['primary'])))
        print(f"  alpha={alpha:.1f}  valid_primary={r['primary']:.5f}")
        if r['primary'] > best_valid:
            best_valid, best_alpha = r['primary'], alpha

    blend_te = best_alpha * fm_te_ens + (1 - best_alpha) * lgb_te_norm
    r_te_blend = evaluate(ute, yte, blend_te)
    print(f"\nbest alpha (on valid) = {best_alpha:.1f}  valid={best_valid:.5f}  test={r_te_blend['primary']:.5f}")
    print(f"reference: FM-only ensemble valid={r_fm_va['primary']:.5f} test={r_fm_te['primary']:.5f}")
    print(f"reference: LightGBM-only valid={va_lgb['primary']:.5f} test={te_lgb['primary']:.5f}")

    out = {
        'lgb_only': {'valid': float(va_lgb['primary']), 'test': float(te_lgb['primary'])},
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

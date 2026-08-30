"""iter44f: does a 5-seed GBM ensemble (num_leaves=2, mirroring FM's own
5-seed ensembling in iter38) add anything over the single-seed GBM used in
blend.py, either standalone or in the FM+GBM blend? Single-seed variance
was already tight (std ~0.0002 valid across 3 seeds), so this checks
whether averaging meaningfully helps or is just extra compute -- and
refines the alpha sweep near blend.py's optimum (0.1) with a finer grid.
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


gbm_train = _load_module(os.path.join(_THIS_DIR, 'train.py'), 'iter44_train')
sys.path.insert(0, _ITER27_DIR)

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
ITER24_FEATS = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def main():
    print("=== training 5-seed GBM ensemble (num_leaves=2) ===", flush=True)
    data = gbm_train.prepare(DATA_DIR)
    dfs, y, u = data
    uva, yva = u['valid'], y['valid']
    ute, yte = u['test'], y['test']

    gbm_va_list, gbm_te_list = [], []
    for seed in range(5):
        model, va, te, _ = gbm_train.run(DATA_DIR, seed=seed, verbose=False, _cache=data,
                                          num_leaves=2, learning_rate=0.05, n_estimators=500,
                                          min_child_samples=200, reg_lambda=1.0)
        gbm_va_list.append(model.predict(dfs['valid']))
        gbm_te_list.append(model.predict(dfs['test']))
        print(f"  gbm seed {seed}: valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)

    # ensemble via mean of min-max normalized scores per seed (each seed's raw
    # LightGBM scores aren't on a common scale by construction)
    gbm_va_ens = np.mean(np.stack([minmax(s) for s in gbm_va_list]), axis=0)
    gbm_te_ens = np.mean(np.stack([minmax(s) for s in gbm_te_list]), axis=0)
    r_gbm_va = evaluate(uva, yva, gbm_va_ens)
    r_gbm_te = evaluate(ute, yte, gbm_te_ens)
    print(f"\nGBM 5-seed ensemble: valid={r_gbm_va['primary']:.5f} test={r_gbm_te['primary']:.5f}")
    print(f"reference (single-seed, seed=0): valid=0.66135 test=0.64794")

    print("\n=== retraining FM 5-seed ensemble (iter38 exact method) ===", flush=True)
    from data_ext import load_ext, encode_ext, HALFLIVES, TAB_HALFLIVES, compute_final_decayed_pos
    from train import build_pos_neg_index, sample_pairs, bpr_step
    from baseline import FM

    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc_fm, dim_fm = encode_ext(SPLITS_CACHE, feature_set=ITER24_FEATS, halflives=HALFLIVES,
                                 tab_halflives=TAB_HALFLIVES, alpha=0.5, n_buckets=20)
    Xtr_fm, ytr_fm, utr_fm = enc_fm['train']
    Xva_fm, yva_fm, uva_fm = enc_fm['valid']
    Xte_fm, yte_fm, ute_fm = enc_fm['test']

    assert np.array_equal(np.asarray(yva_fm), np.asarray(yva))
    assert np.array_equal(np.asarray(yte_fm), np.asarray(yte))

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

    print("\n=== fine alpha sweep: FM-ensemble vs GBM-5-seed-ensemble ===")
    best_alpha, best_valid = None, -1
    sweep_results = []
    for alpha in np.arange(0.0, 0.41, 0.02):
        blend_va = alpha * fm_va_ens + (1 - alpha) * gbm_va_ens
        r = evaluate(uva, yva, blend_va)
        sweep_results.append((round(float(alpha), 3), float(r['primary'])))
        print(f"  alpha={alpha:.2f}  valid_primary={r['primary']:.5f}")
        if r['primary'] > best_valid:
            best_valid, best_alpha = r['primary'], alpha

    blend_te = best_alpha * fm_te_ens + (1 - best_alpha) * gbm_te_ens
    r_te_blend = evaluate(ute, yte, blend_te)
    print(f"\nbest alpha (on valid) = {best_alpha:.2f}  valid={best_valid:.5f}  test={r_te_blend['primary']:.5f}")
    print(f"reference: FM-ensemble-only valid={r_fm_va['primary']:.5f} test={r_fm_te['primary']:.5f}")
    print(f"reference: GBM-5-seed-ensemble-only valid={r_gbm_va['primary']:.5f} test={r_gbm_te['primary']:.5f}")
    print(f"reference: single-seed blend (blend.py) valid=0.66473 test=0.65197")

    out = {
        'gbm_5seed_ens': {'valid': float(r_gbm_va['primary']), 'test': float(r_gbm_te['primary'])},
        'fm_ensemble_only': {'valid': float(r_fm_va['primary']), 'test': float(r_fm_te['primary'])},
        'fine_sweep': sweep_results,
        'best_alpha': float(best_alpha),
        'blend_at_best_alpha': {'valid': float(best_valid), 'test': float(r_te_blend['primary'])},
    }
    with open(os.path.join(_THIS_DIR, 'blend2_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nwrote blend2_results.json")


if __name__ == '__main__':
    main()

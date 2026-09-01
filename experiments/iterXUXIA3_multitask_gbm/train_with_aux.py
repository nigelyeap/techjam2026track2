"""6c: reopen multi-task learning under the GBM-native representation
(approach 1: auxiliary features via stacking, not auxiliary losses).

iter31/iter36 rejected multi-task learning under FM's shared-embedding
architecture (diagnosed: shared absolute score conflated a rank-invariant
BPR loss with base-rate-calibrated pointwise losses). This tests whether
that REJECT generalizes to the GBM-native ranker, which has no shared
embedding table -- via leakage-free out-of-fold predictions of
is_like/is_follow/is_comment/is_forward fed in as 4 new input features
(see aux_features.py for the full no-leakage argument and alignment
spot-check).

Uses t63.run()'s own `_cache=(dfs, y, u)` parameter to inject the
aux-augmented dataframe directly into iter63's unmodified training code
path -- same LGBMRanker call, same hyperparameters, only the input
DataFrame's column set differs between baseline and aux-augmented runs.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aux_features import build_oof_features, AUX_TASKS, t63, DATA_DIR  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'iter47_stacking_meta'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'iter27_triple_fusion'))
from evaluate import evaluate  # noqa: E402
from data_ext import load_ext, encode_ext, HALFLIVES, TAB_HALFLIVES  # noqa: E402
import stack as iter47  # noqa: E402

GBM_SEEDS = (0, 1, 2, 3, 4)
FM_SEEDS = (0, 1, 2, 3, 4)
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')


def build_dfs_variants():
    r = build_oof_features(use_cache=True)
    dfs_base, y, u = r['dfs'], r['y'], r['u']
    dfs_aux = {name: dfs_base[name].copy() for name in ('train', 'valid', 'test')}
    for name in ('train', 'valid', 'test'):
        for task in AUX_TASKS:
            dfs_aux[name][f'aux_{task}'] = r['oof'][name][task]
    return dfs_base, dfs_aux, y, u


def train_variant(dfs, y, u, seed, verbose=False):
    model, va, te, _ = t63.run(DATA_DIR, 'rate_only', seed=seed, verbose=verbose, _cache=(dfs, y, u))
    return model, va, te


def get_fm_ensemble_cached(verbose=True):
    import pickle
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache_fm_ensemble.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    splits = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc, dim = encode_ext(splits, feature_set=FEATURES, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                           alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    fm_va_scores, fm_te_scores = [], []
    for seed in FM_SEEDS:
        if verbose:
            print(f"  training FM seed {seed}...", flush=True)
        m = iter47.train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed)
        fm_va_scores.append(iter47.sigmoid(m.predict(Xva)))
        fm_te_scores.append(iter47.sigmoid(m.predict(Xte)))
    result = {
        'fm_va_ens': np.mean(np.stack(fm_va_scores), axis=0), 'fm_te_ens': np.mean(np.stack(fm_te_scores), axis=0),
        'yva': np.asarray(yva), 'yte': np.asarray(yte), 'uva': np.asarray(uva), 'ute': np.asarray(ute),
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    return result


if __name__ == '__main__':
    dfs_base, dfs_aux, y, u = build_dfs_variants()

    print("=== harness-fidelity check (baseline, no aux features, seed=0) ===")
    model_b0, va_b0, te_b0 = train_variant(dfs_base, y, u, seed=0, verbose=True)
    print(f"baseline standalone: valid={va_b0['primary']:.5f} test={te_b0['primary']:.5f}  "
          f"[expect ~0.67168/0.65353, matching iter63 rate_only]")

    print("\n=== seed=0 single-run comparison: baseline vs. +aux features ===")
    model_a0, va_a0, te_a0 = train_variant(dfs_aux, y, u, seed=0, verbose=True)
    delta0 = va_a0['primary'] - va_b0['primary']
    print(f"baseline: valid={va_b0['primary']:.5f} test={te_b0['primary']:.5f}")
    print(f"+aux:     valid={va_a0['primary']:.5f} test={te_a0['primary']:.5f}  (delta valid={delta0:+.5f})")

    print("\n=== feature importance (aux-augmented model, seed=0) ===")
    imp = dict(zip(model_a0.feature_name_, model_a0.feature_importances_))
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        flag = "  <-- aux" if k.startswith('aux_') else ""
        print(f"  {k:20s} {v:6d}{flag}")

    results = {'seed0': {'baseline': {'valid': float(va_b0['primary']), 'test': float(te_b0['primary'])},
                          'aux': {'valid': float(va_a0['primary']), 'test': float(te_a0['primary'])},
                          'delta_valid': float(delta0)},
               'feature_importance': {k: int(v) for k, v in imp.items()}}

    if delta0 >= 0.0003:
        print(f"\nseed=0 delta {delta0:+.5f} >= 0.0003 promotion-look threshold -- running full 5-seed confirmation + blend check")
        seed_results = []
        for seed in GBM_SEEDS:
            mb, vb, tb = train_variant(dfs_base, y, u, seed=seed, verbose=False)
            ma, va, ta = train_variant(dfs_aux, y, u, seed=seed, verbose=False)
            d = va['primary'] - vb['primary']
            print(f"  seed {seed}: baseline valid={vb['primary']:.5f}  +aux valid={va['primary']:.5f}  delta={d:+.5f}")
            seed_results.append({'seed': seed, 'baseline_valid': float(vb['primary']), 'aux_valid': float(va['primary']),
                                  'baseline_test': float(tb['primary']), 'aux_test': float(ta['primary']), 'delta': float(d)})
        deltas = [r['delta'] for r in seed_results]
        print(f"\n5-seed deltas: {[f'{d:+.5f}' for d in deltas]}")
        print(f"mean={np.mean(deltas):+.5f}  min={np.min(deltas):+.5f}  seeds>=+0.001: {sum(d >= 0.001 for d in deltas)}/5")
        results['seed_confirmation'] = seed_results

        if np.mean(deltas) >= 0.001 and min(deltas) > 0:
            print("\n=== checking blend impact with FM ensemble (seed=0 aux model) ===")
            fm = get_fm_ensemble_cached(verbose=True)
            gbm_va_n = iter47.minmax(model_a0.predict(dfs_aux['valid']))
            gbm_te_n = iter47.minmax(model_a0.predict(dfs_aux['test']))
            best = {'alpha': None, 'valid': -1}
            for a in np.arange(0.0, 0.31, 0.01):
                m = evaluate(fm['uva'], fm['yva'], a * fm['fm_va_ens'] + (1 - a) * gbm_va_n)
                if m['primary'] > best['valid']:
                    best = {'alpha': float(a), 'valid': float(m['primary'])}
            te_m = evaluate(fm['ute'], fm['yte'], best['alpha'] * fm['fm_te_ens'] + (1 - best['alpha']) * gbm_te_n)['primary']
            print(f"aux-augmented blend: alpha={best['alpha']:.2f} valid={best['valid']:.5f} test={te_m:.5f}")
            print(f"[compare] current best (iter63) blend: valid=0.67606 test=0.65955")
            results['blend_check'] = {'alpha': best['alpha'], 'valid': best['valid'], 'test': float(te_m)}
    else:
        print(f"\nseed=0 delta {delta0:+.5f} below the 0.0003 promotion-look threshold -- not proceeding to 5-seed confirmation")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aux_gbm_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote aux_gbm_results.json")

"""iter42: does adding a decayed 'other engagement' (like/follow/comment/
forward) rate feature improve on iter27/iter38's proven feature set?
Harness-fidelity check (iter24 feature set, seed 0, must match iter27's
published 0.6389/0.6399) then a single-seed test of iter24_feats +
decay_engage_2.5, then 5-seed confirm if the single-seed gain looks real.
"""
import os, sys, json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, _THIS_DIR)
from data_ext import load_ext, encode_ext, HALFLIVES, TAB_HALFLIVES, ENGAGE_HALFLIVES, compute_final_decayed_pos
from train import build_pos_neg_index, sample_pairs, bpr_step
from baseline import FM
from evaluate import evaluate

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
ITER24_FEATS = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
ITER42_FEATS = ITER24_FEATS + ('decay_engage_2.5',)


def train_one(Xtr, ytr, utr, Xva, yva, uva, Xte, yte, ute, dim, seed, splits_cache,
              k=16, lr=0.001, epochs=40, bs=8192, patience=4,
              sampling_alpha=0.75, decay_halflife=3):
    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs)))
    decayed_pos_dict = compute_final_decayed_pos(splits_cache['train'], halflife=decay_halflife)
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
        v = evaluate(uva, yva, m.predict(Xva))
        if v['primary'] > best + 1e-5:
            best, bad = v['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    va = evaluate(uva, yva, m.predict(Xva))
    te = evaluate(ute, yte, m.predict(Xte))
    return va, te


def run_config(feature_set, seeds, splits_cache, tag):
    enc, dim = encode_ext(splits_cache, feature_set=feature_set, halflives=HALFLIVES,
                           tab_halflives=TAB_HALFLIVES, alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    results = []
    for seed in seeds:
        va, te = train_one(Xtr, ytr, utr, Xva, yva, uva, Xte, yte, ute, dim, seed, splits_cache)
        print(f"  [{tag}] seed={seed} valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)
        results.append({'seed': seed, 'valid': float(va['primary']), 'test': float(te['primary'])})
    return results


def main():
    print("loading data...", flush=True)
    splits_cache = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)

    print("\n=== harness-fidelity check: iter24 feature set, seed 0 (expect ~0.63894/0.63989) ===", flush=True)
    fid = run_config(ITER24_FEATS, [0], splits_cache, 'fidelity')
    assert abs(fid[0]['valid'] - 0.6389358639717102) < 0.003, f"fidelity check FAILED: {fid}"
    print("fidelity check PASSED", flush=True)

    print("\n=== iter42: iter24_feats + decay_engage_2.5, seed 0 ===", flush=True)
    single = run_config(ITER42_FEATS, [0], splits_cache, 'iter42-seed0')

    out = {'fidelity': fid, 'iter42_seed0': single}
    delta = single[0]['valid'] - fid[0]['valid']
    print(f"\nseed-0 delta (iter42 - iter24): valid {delta:+.5f}")

    if delta > 0.0003:  # worth a full 5-seed confirm
        print("\n=== promising -- running 5-seed confirm on both configs ===", flush=True)
        base5 = run_config(ITER24_FEATS, range(5), splits_cache, 'base-5seed')
        new5 = run_config(ITER42_FEATS, range(5), splits_cache, 'iter42-5seed')
        out['base_5seed'] = base5
        out['iter42_5seed'] = new5
        base_mean = np.mean([r['valid'] for r in base5])
        new_mean = np.mean([r['valid'] for r in new5])
        print(f"\nbase 5-seed valid mean={base_mean:.5f}  iter42 5-seed valid mean={new_mean:.5f}  "
              f"delta={new_mean - base_mean:+.5f}")
    else:
        print("\nseed-0 delta too small to bother with 5-seed confirm -- treating as REJECT")

    with open(os.path.join(_THIS_DIR, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nwrote results.json")


if __name__ == '__main__':
    main()

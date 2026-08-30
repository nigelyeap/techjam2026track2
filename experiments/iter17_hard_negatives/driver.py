"""Sweep driver for iter17. Loads data / computes causal features / encodes
ONCE (these are identical across all modes and seeds -- feature_set is fixed
to iter9's activity,tab,rate), then loops negative_mode x seed, writing each
result to results.json incrementally so partial progress survives even if
interrupted. Run as a plain foreground command.
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate
from baseline import FM, sigmoid
from data_ext import load_ext, encode_ext, BASE_FIELDS
from train import (build_pos_neg_index, build_negative_pools, sample_pairs,
                    sample_negatives, bpr_step, ITEM_COLS)

DATA_DIR = '../../KuaiRand-Pure/data'
FEATURE_SET = ('activity', 'tab', 'rate')
K, LR, EPOCHS, BS, PATIENCE = 16, 0.001, 40, 8192, 4
MIN_TAB_POOL = 20
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')


def load_existing():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save(results):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)


def run_one(Xtr, ytr, utr, Xva, yva, uva, Xte, yte, ute,
            eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
            user_cumw, user_totalw, pools, negative_mode, seed, dim):
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / BS)))

    m = FM(dim, k=K, lr=LR, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    fb_counter = [0, 0]
    t_start = time.time()
    for ep in range(1, EPOCHS + 1):
        for _ in range(steps_per_epoch):
            pos_rows, neg_rows_uniform = sample_pairs(
                rng, n_users, BS, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                user_cumw=user_cumw, user_totalw=user_totalw)
            Xpos = Xtr[pos_rows]
            Xneg = sample_negatives(negative_mode, rng, pos_rows, Xtr, pools, neg_rows_uniform, fb_counter)
            bpr_step(m, Xpos, Xneg)
        va = evaluate(uva, yva, m.predict(Xva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.V, m.W, m.b = best_state
    elapsed = time.time() - t_start
    fallback_rate = (fb_counter[0] / fb_counter[1]) if fb_counter[1] > 0 else None
    va_final = evaluate(uva, yva, m.predict(Xva))
    te_final = evaluate(ute, yte, m.predict(Xte))
    return {
        'negative_mode': negative_mode, 'seed': seed, 'epochs_run': ep,
        'elapsed_s': round(elapsed, 1),
        'fallback_rate': (float(fallback_rate) if fallback_rate is not None else None),
        'valid_GAUC': float(va_final['GAUC']), 'valid_nDCG5': float(va_final['nDCG@5']),
        'valid_primary': float(va_final['primary']),
        'test_GAUC': float(te_final['GAUC']), 'test_nDCG5': float(te_final['nDCG@5']),
        'test_primary': float(te_final['primary']),
    }


def main(modes, seeds):
    print(f"loading {DATA_DIR} + computing causal features (shared across all configs)...")
    t0 = time.time()
    splits = load_ext(DATA_DIR)
    enc, dim = encode_ext(splits, feature_set=FEATURE_SET)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    print(f"  done in {time.time()-t0:.1f}s, dim={dim}")

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = build_pos_neg_index(ytr, utr)
    user_cumw = np.cumsum(pos_len.astype(np.float64))
    user_totalw = user_cumw[-1]
    pools = build_negative_pools(Xtr, min_tab_pool=MIN_TAB_POOL)

    # report tab pool sizes once for RESULT.md
    print("  tab candidate-pool sizes (unique videos per tab):",
          {t: len(v) for t, v in sorted(pools['tab_uniq_rep'].items())})

    results = load_existing()
    done = {(r['negative_mode'], r['seed']) for r in results}

    for mode in modes:
        for seed in seeds:
            if (mode, seed) in done:
                print(f"skip {mode} seed={seed} (already done)")
                continue
            print(f"running {mode} seed={seed} ...")
            t0 = time.time()
            r = run_one(Xtr, ytr, utr, Xva, yva, uva, Xte, yte, ute,
                        eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                        user_cumw, user_totalw, pools, mode, seed, dim)
            print(f"  -> valid primary {r['valid_primary']:.4f} test primary {r['test_primary']:.4f} "
                  f"epochs {r['epochs_run']} fallback {r['fallback_rate']} ({time.time()-t0:.1f}s)")
            results.append(r)
            save(results)

    print("\nsweep complete. results.json written.")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='uniform,same_tab,pop_weighted,same_tab_pop_weighted')
    ap.add_argument('--seeds', default='0,1,2')
    a = ap.parse_args()
    main(a.modes.split(','), [int(s) for s in a.seeds.split(',')])

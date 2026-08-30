"""iter14 driver: capacity (k) + bucketing resolution (n_buckets) sweep on
iter9's exact feature set (activity+tab+rate), activity-weighted BPR loss.

Writes incremental results to results.json after every single run so partial
progress survives interruption. Caches load_ext() (expensive: sort+traverse
~1.4M rows) and encode_ext() (depends only on feature_set/n_buckets, not on
k/seed) across configs to avoid redundant work.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..'))

from baseline import FM, sigmoid                      # noqa: E402
from evaluate import evaluate                          # noqa: E402
from data_ext import load_ext, encode_ext, BASE_FIELDS  # noqa: E402
from train import build_pos_neg_index, sample_pairs, bpr_step  # noqa: E402

DATA_DIR = os.path.join(HERE, '..', '..', 'KuaiRand-Pure', 'data')
FEATURE_SET = ('activity', 'tab', 'rate')
RESULTS_PATH = os.path.join(HERE, 'results.json')

_encode_cache = {}


def get_encoded(splits, n_buckets):
    if n_buckets not in _encode_cache:
        print(f"  [encode] n_buckets={n_buckets} ...", flush=True)
        t0 = time.time()
        enc, dim = encode_ext(splits, feature_set=FEATURE_SET, n_buckets=n_buckets)
        print(f"  [encode] done in {time.time()-t0:.1f}s, dim={dim}", flush=True)
        _encode_cache[n_buckets] = (enc, dim)
    return _encode_cache[n_buckets]


def run_one(enc, dim, k, lr, seed, epochs=40, bs=8192, patience=4, steps_mult=1):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs))) * steps_mult
    user_cumw = np.cumsum(pos_len.astype(np.float64))
    user_totalw = user_cumw[-1]

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            Xpos_rows, Xneg_rows = sample_pairs(
                rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
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
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test': evaluate(ute, yte, m.predict(Xte))}


def clean(d):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_result(entry):
    results = load_results()
    results.append(entry)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)


def already_done(tag, k, n_buckets, seed):
    for r in load_results():
        if r['tag'] == tag and r['k'] == k and r['n_buckets'] == n_buckets and r['seed'] == seed:
            return True
    return False


def main():
    print(f"loading {DATA_DIR} ...", flush=True)
    t0 = time.time()
    splits = load_ext(DATA_DIR)
    print(f"load_ext done in {time.time()-t0:.1f}s", flush=True)

    configs = []
    # Axis A: k sweep, n_buckets=10 fixed, seeds 0,1,2
    for k in (16, 24, 32):
        for seed in (0, 1, 2):
            configs.append(('axisA_k', k, 10, seed))
    # Axis B: n_buckets sweep, k=16 fixed, seeds 0,1,2
    for nb in (5, 10, 20):
        for seed in (0, 1, 2):
            configs.append(('axisB_nbuckets', 16, nb, seed))

    for tag, k, nb, seed in configs:
        if already_done(tag, k, nb, seed):
            print(f"SKIP (done) tag={tag} k={k} n_buckets={nb} seed={seed}", flush=True)
            continue
        enc, dim = get_encoded(splits, nb)
        print(f"RUN tag={tag} k={k} n_buckets={nb} seed={seed} dim={dim} ...", flush=True)
        t0 = time.time()
        res = run_one(enc, dim, k=k, lr=0.001, seed=seed)
        dt = time.time() - t0
        entry = {'tag': tag, 'k': k, 'n_buckets': nb, 'seed': seed, 'dim': dim,
                  'valid': clean(res['valid']), 'test': clean(res['test']), 'time_s': dt}
        save_result(entry)
        print(f"  -> valid primary {res['valid']['primary']:.5f} | test primary "
              f"{res['test']['primary']:.5f} | {dt:.1f}s", flush=True)

    print("\nAll configs done.", flush=True)


if __name__ == '__main__':
    main()

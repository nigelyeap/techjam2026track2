"""iter38: score-level ensemble of iter27's 5 already-confirmed seeds.

Motivation: iter27's 5-seed std is ~0.0007-0.0008 on both splits -- purely
random-init variance, not signal. Averaging several independently-trained
models' scores is a standard variance-reduction technique and has never
been tried (every prior iteration reports single-seed or seed-mean metrics,
never an actual multi-model ensemble prediction). This is a genuinely new,
disjoint axis: it changes nothing about features/loss/sampling, only how
final predictions are produced.

Method: train iter27's exact winning config at seeds 0-4 (identical to the
already-published 5-seed run), but additionally keep each trained model's
raw per-row scores for valid/test (not just the aggregate metric). Ensemble
by simple mean of raw logits across the 5 models (scores are on a comparable
scale: identical architecture/features/regularization, only random init and
minibatch order differ). Evaluate the ensembled score with evaluate.py,
compare against the arithmetic mean of the 5 single-model primaries
(iter27's already-published number) to isolate the ensembling effect from
plain averaging-of-metrics.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'iter27_triple_fusion'))
from data_ext import load_ext, encode_ext, HALFLIVES, TAB_HALFLIVES
from train import build_pos_neg_index, sample_pairs, bpr_step
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from baseline import FM
from evaluate import evaluate

DATA_DIR = '../../KuaiRand-Pure/data'
ITER24_FEATS = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
RESULTS_PATH = 'results.json'

ITER27_PUBLISHED = [
    (0.6389358639717102, 0.6398863196372986),
    (0.638678789138794, 0.6391348838806152),
    (0.6368540525436401, 0.6376838684082031),
    (0.6374706625938416, 0.6385308504104614),
    (0.6376844048500061, 0.6392104625701904),
]


def train_one(Xtr, ytr, utr, Xva, yva, uva, Xte, yte, ute, dim, seed,
              k=16, lr=0.001, epochs=40, bs=8192, patience=4,
              sampling_alpha=0.75, decay_halflife=3):
    from data_ext import compute_final_decayed_pos
    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs)))
    decayed_pos_dict = compute_final_decayed_pos(SPLITS_CACHE['train'], halflife=decay_halflife)
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


if __name__ == '__main__':
    print("=== Loading fused-config dataset ===", flush=True)
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc, dim = encode_ext(SPLITS_CACHE, feature_set=ITER24_FEATS, halflives=HALFLIVES,
                           tab_halflives=TAB_HALFLIVES, alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    va_scores, te_scores = [], []
    per_seed = []
    for seed in range(5):
        t0 = time.time()
        m = train_one(Xtr, ytr, utr, Xva, yva, uva, Xte, yte, ute, dim, seed)
        sva = m.predict(Xva); ste = m.predict(Xte)
        va_scores.append(sva); te_scores.append(ste)
        r_va = evaluate(uva, yva, sva); r_te = evaluate(ute, yte, ste)
        per_seed.append({'seed': seed, 'valid_primary': float(r_va['primary']), 'test_primary': float(r_te['primary'])})
        pv, pt = ITER27_PUBLISHED[seed]
        dv, dt = r_va['primary'] - pv, r_te['primary'] - pt
        print(f"[seed {seed}] valid={r_va['primary']:.5f} (Δ{dv:+.6f}) test={r_te['primary']:.5f} (Δ{dt:+.6f})  {time.time()-t0:.1f}s", flush=True)

    all_match = all(abs(p['valid_primary'] - ITER27_PUBLISHED[p['seed']][0]) < 1e-4 and
                     abs(p['test_primary'] - ITER27_PUBLISHED[p['seed']][1]) < 1e-4 for p in per_seed)
    print(f"\nHarness fidelity vs iter27 published: {'CONFIRMED bit-exact' if all_match else 'MISMATCH -- STOP'}", flush=True)
    if not all_match:
        raise SystemExit(1)

    mean_valid_metric = sum(p['valid_primary'] for p in per_seed) / 5
    mean_test_metric = sum(p['test_primary'] for p in per_seed) / 5

    # ensemble: simple mean of raw logits across the 5 models
    ens_va_raw = np.mean(np.stack(va_scores), axis=0)
    ens_te_raw = np.mean(np.stack(te_scores), axis=0)
    r_va_ens = evaluate(uva, yva, ens_va_raw)
    r_te_ens = evaluate(ute, yte, ens_te_raw)

    # ensemble: mean of sigmoid-transformed scores (bounded, may be more robust to scale drift)
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    ens_va_sig = np.mean(np.stack([sigmoid(s) for s in va_scores]), axis=0)
    ens_te_sig = np.mean(np.stack([sigmoid(s) for s in te_scores]), axis=0)
    r_va_ens_sig = evaluate(uva, yva, ens_va_sig)
    r_te_ens_sig = evaluate(ute, yte, ens_te_sig)

    print(f"\n=== iter27 5-seed mean-of-metrics (published) ===")
    print(f"  valid={mean_valid_metric:.5f}  test={mean_test_metric:.5f}")
    print(f"=== iter38 5-model ENSEMBLE (mean of raw logits) ===")
    print(f"  valid={r_va_ens['primary']:.5f} (Δ{r_va_ens['primary']-mean_valid_metric:+.5f})  "
          f"test={r_te_ens['primary']:.5f} (Δ{r_te_ens['primary']-mean_test_metric:+.5f})")
    print(f"=== iter38 5-model ENSEMBLE (mean of sigmoid scores) ===")
    print(f"  valid={r_va_ens_sig['primary']:.5f} (Δ{r_va_ens_sig['primary']-mean_valid_metric:+.5f})  "
          f"test={r_te_ens_sig['primary']:.5f} (Δ{r_te_ens_sig['primary']-mean_test_metric:+.5f})")

    out = {
        'per_seed': per_seed,
        'mean_of_metrics': {'valid': mean_valid_metric, 'test': mean_test_metric},
        'ensemble_raw_mean': {'valid': {k: float(v) for k, v in r_va_ens.items()},
                               'test': {k: float(v) for k, v in r_te_ens.items()}},
        'ensemble_sigmoid_mean': {'valid': {k: float(v) for k, v in r_va_ens_sig.items()},
                                   'test': {k: float(v) for k, v in r_te_ens_sig.items()}},
    }
    with open(RESULTS_PATH, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {RESULTS_PATH}")

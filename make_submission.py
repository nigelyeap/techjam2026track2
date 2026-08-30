"""Generates the final submission CSV using iter55's blend (current best, see
experiments/LEDGER.md's "Best-known candidate"): a score-level blend
(alpha=0.10, 90% weight on the GBM) of

  (a) a single LightGBM LGBMRanker (num_leaves=2, lr=0.10, n_estimators=500,
      min_child_samples=200, reg_lambda=1.0, linear_tree=True) trained on a
      GBM-native (un-bucketed) encoding of iter27's causal features -- see
      experiments/iter55_learning_rate_sweep/train.py, which reuses iter51's
      run() with learning_rate=0.10 -- confirmed in
      experiments/iter55_learning_rate_sweep/RESULT.md: 5-seed mean
      valid 0.67011/test 0.65230 standalone (learning_rate=0.10 beats
      iter51's lr=0.05 by +0.00085 valid).

  (b) iter38's unchanged 5-model ensemble (seeds 0-4) of FM + activity-
      weighted BPR with iter24's recency-decay/momentum features -- see
      experiments/iter38_seed_ensemble/driver.py -- valid 0.63988/test
      0.64187 standalone.

The blend itself is confirmed in
experiments/iter55_learning_rate_sweep/blend_results.json: valid 0.67451/
test 0.65832 (alpha=0.10), the current project best -- superseding iter51's
blend (valid 0.67297/test 0.65643). Scores the official test split in file
order, matching submit.py's expected format.

Note: unlike earlier iterations (numpy only), this final model additionally
requires pandas and lightgbm (both pip-installable; see README.md).

Usage: python3 make_submission.py [output_path]  (default: submission.csv)
"""
import os, sys, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', 'iter27_triple_fusion'))
import numpy as np
from data import load
from baseline import FM
from submit import write_submission, read_submission
from evaluate import evaluate
from data_ext import load_ext, encode_ext, compute_final_decayed_pos, HALFLIVES, TAB_HALFLIVES
from train import build_pos_neg_index, sample_pairs, bpr_step

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_ITER51_DIR = os.path.join(_REPO_ROOT, 'experiments', 'iter51_linear_tree')

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
SEEDS = (0, 1, 2, 3, 4)
ALPHA_BLEND = 0.10  # weight on the FM ensemble; 1-ALPHA_BLEND on the GBM (iter55/blend.py's confirmed optimum)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


if __name__ == '__main__':
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'submission.csv'

    print("=== training native-feature LightGBM ranker (num_leaves=2, linear_tree=True, lr=0.10, iter55's winner) ===", flush=True)
    gbm_train = _load_module(os.path.join(_ITER51_DIR, 'train.py'), 'iter51_train_final')
    dfs_gbm, y_gbm, u_gbm = gbm_train.gbm44.prepare(DATA_DIR)
    gbm_model, gbm_va_metrics, gbm_te_metrics, _ = gbm_train.run(
        DATA_DIR, verbose=True, linear_tree=True, num_leaves=2, learning_rate=0.10,
        n_estimators=500, min_child_samples=200, reg_lambda=1.0, seed=0, _cache=(dfs_gbm, y_gbm, u_gbm))
    print(f"  GBM standalone: valid={gbm_va_metrics['primary']:.5f} test={gbm_te_metrics['primary']:.5f}", flush=True)
    gbm_va_raw = gbm_model.predict(dfs_gbm['valid'])
    gbm_te_raw = gbm_model.predict(dfs_gbm['test'])

    print("\n=== training FM 5-seed ensemble (iter38 exact config) ===", flush=True)
    splits = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc, dim = encode_ext(splits, feature_set=FEATURES, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                           alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    assert np.array_equal(np.asarray(yva), y_gbm['valid']), "FM/GBM valid label order mismatch"
    assert np.array_equal(np.asarray(yte), y_gbm['test']), "FM/GBM test label order mismatch"

    fm_va_scores, fm_te_scores = [], []
    for seed in SEEDS:
        print(f"  training FM seed {seed}...", flush=True)
        m = train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed)
        fm_va_scores.append(sigmoid(m.predict(Xva)))
        fm_te_scores.append(sigmoid(m.predict(Xte)))

    fm_va_ens = np.mean(np.stack(fm_va_scores), axis=0)
    fm_te_ens = np.mean(np.stack(fm_te_scores), axis=0)
    fm_va_metrics = evaluate(uva, yva, fm_va_ens)
    fm_te_metrics = evaluate(ute, yte, fm_te_ens)
    print(f"  FM ensemble standalone: valid={fm_va_metrics['primary']:.5f} test={fm_te_metrics['primary']:.5f}", flush=True)

    print(f"\n=== blending (alpha={ALPHA_BLEND}, {1-ALPHA_BLEND:.0%} GBM / {ALPHA_BLEND:.0%} FM) ===", flush=True)
    gbm_va_norm, gbm_te_norm = minmax(gbm_va_raw), minmax(gbm_te_raw)
    blend_va = ALPHA_BLEND * fm_va_ens + (1 - ALPHA_BLEND) * gbm_va_norm
    blend_te = ALPHA_BLEND * fm_te_ens + (1 - ALPHA_BLEND) * gbm_te_norm
    va_metrics = evaluate(uva, yva, blend_va)
    te_metrics = evaluate(ute, yte, blend_te)
    print(f"\niter55 blend: valid primary={va_metrics['primary']:.5f}  test primary={te_metrics['primary']:.5f}", flush=True)

    raw_test_rows = load(DATA_DIR)['test']
    assert len(raw_test_rows) == len(dfs_gbm['test']), \
        f"row count mismatch: data.load() test={len(raw_test_rows)} vs GBM test={len(dfs_gbm['test'])}"
    write_submission(out_path, raw_test_rows, blend_te)
    print(f"wrote {out_path} ({len(raw_test_rows)} rows)")

    checked_scores = read_submission(out_path, raw_test_rows)
    assert len(checked_scores) == len(raw_test_rows)
    print(f"submit.py format/alignment check: PASSED ({len(checked_scores)} rows)")

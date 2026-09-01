"""Generates the final submission CSV using the current best model (see
experiments/MERGE_LEDGER.md's Round 1 and experiments/iterYIXI10_video_metadata/RESULT.md):
a within-user-percentile blend of three components --

  (a) an XGBRanker tuned by teammate yixi (experiments/iterYIXI5_xgboost_optimization),
      trained on native (un-bucketed) causal features (duration_ms, decay_tab_3,
      lastk_rate, gap, decay_rate_5, decay_act_5),

  (b) a LightGBM LGBMRanker (num_leaves=2, lr=0.10, n_estimators=500,
      min_child_samples=200, reg_lambda=1.0, linear_tree=True, lambdarank
      objective with truncation_level=50/sigmoid=2.0) trained on yixi's
      chained feature set through iterYIXI10: the reference columns plus a
      causal 5-day historical watch-depth decay feature
      (hist_watch_decay_mean_5, iterYIXI9) and a native `upload_type`
      categorical (meta_upload_type, iterYIXI10),

  (c) this project's own unchanged iter38 5-model FM+BPR ensemble (seeds 0-4,
      recency-decay/momentum features, activity-weighted sampling) -- the
      exact train_one_fm()/encode_ext() below, verified by yixi's own blend.py
      to be bit-identical to this file's implementation.

Independently reproduced from scratch (all three components retrained, no
reuse of any cached/frozen prediction artifact) in
experiments/iterMERGE1_verify_yixi10/verify.py: exact match to yixi's
claimed valid 0.69943440 / test 0.68432260 on every component and the final
blend (delta +0.00000000 throughout) -- see
experiments/iterMERGE1_verify_yixi10/RESULT.md. This supersedes iter63's
GBM+FM blend (valid 0.67606/test 0.65955, the project's own prior best,
which did not include an XGBoost component or yixi's LightGBM feature/
objective refinements).

Note: like iter63, this requires pandas and lightgbm; it additionally
requires xgboost (pip-installable; see README.md).

Usage: python3 make_submission.py [output_path]  (default: submission.csv)
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', 'iter27_triple_fusion'))
from data import load
from baseline import FM
from submit import write_submission, read_submission
from evaluate import evaluate
from data_ext import load_ext, encode_ext, compute_final_decayed_pos, HALFLIVES, TAB_HALFLIVES
from train import build_pos_neg_index, sample_pairs, bpr_step

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_YIXI10_DIR = os.path.join(_REPO_ROOT, 'experiments', 'iterYIXI10_video_metadata')
_YIXI5_RESULTS_PATH = os.path.join(_REPO_ROOT, 'experiments', 'iterYIXI5_xgboost_optimization', 'results.json')

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
SEEDS = (0, 1, 2, 3, 4)
TRAIN_SEED = 0  # the LGB/XGB seed behind yixi's headline valid 0.69943440 / test 0.68432260
WEIGHTS = {'fm': 0.10, 'lgb': 0.52, 'xgb': 0.38}

CAT_COLS = ['user_id', 'video_id', 'author_id', 'tab', 'last1']
LGB_CANDIDATE_COLUMNS = CAT_COLS + [
    'duration_ms', 'decay_rate_5', 'decay_act_5', 'lastk_rate', 'gap',
    'decay_tab_rate_3', 'hist_watch_decay_mean_5', 'meta_upload_type',
]
XGB_COLUMNS = CAT_COLS + ['duration_ms', 'decay_tab_3', 'lastk_rate', 'gap', 'decay_rate_5', 'decay_act_5']

LGB_CONFIG = dict(
    objective='lambdarank', lambdarank_truncation_level=50, sigmoid=2.0,
    metric='ndcg', eval_at=[5], num_leaves=2, learning_rate=0.10,
    n_estimators=500, min_child_samples=200, reg_lambda=1.0,
    verbosity=-1, n_jobs=-1, linear_tree=True,
)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def within_user_percentile(scores, user_ids):
    s = pd.Series(np.asarray(scores, dtype=np.float64), copy=False)
    u = pd.Series(np.asarray(user_ids), copy=False)
    return s.groupby(u, sort=False).rank(method='average', pct=True).to_numpy(dtype=np.float64)


def _stable_user_order(user_ids):
    values = np.asarray(user_ids)
    order = np.argsort(values, kind='stable')
    groups = np.unique(values[order], return_counts=True)[1]
    return order, groups


def _xgb_config():
    import json
    payload = json.load(open(_YIXI5_RESULTS_PATH, encoding='utf-8'))
    return payload['selected_on_validation']['config']


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


def _fit_lgb(frames, y, users, columns, seed):
    train_order, train_groups = _stable_user_order(users['train'])
    valid_order, valid_groups = _stable_user_order(users['valid'])
    Xtr = frames['train'][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames['valid'][columns].iloc[valid_order].reset_index(drop=True)
    model = lgb.LGBMRanker(**LGB_CONFIG, random_state=seed)
    model.fit(Xtr, y['train'][train_order], group=train_groups,
              eval_set=[(Xva, y['valid'][valid_order])], eval_group=[valid_groups],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    return model


def _fit_xgb(frames, y, users, columns, seed):
    train_order, train_groups = _stable_user_order(users['train'])
    valid_order, valid_groups = _stable_user_order(users['valid'])
    Xtr = frames['train'][columns].iloc[train_order].reset_index(drop=True)
    Xva = frames['valid'][columns].iloc[valid_order].reset_index(drop=True)
    model = xgb.XGBRanker(**_xgb_config(), random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(Xtr, y['train'][train_order], group=train_groups,
              eval_set=[(Xva, y['valid'][valid_order])], eval_group=[valid_groups], verbose=False)
    return model


if __name__ == '__main__':
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'submission.csv'

    print("=== [1/4] rebuilding yixi's chained causal feature frames (through iterYIXI10) ===", flush=True)
    features_mod = _load_module(os.path.join(_YIXI10_DIR, 'features.py'), 'make_submission_yixi10_features')
    frames, y, users, _meta = features_mod.load_frames()
    print(f"  train/valid/test rows = {len(frames['train'])}/{len(frames['valid'])}/{len(frames['test'])}", flush=True)

    print("\n=== [2/4] training LightGBM (yixi10 columns) and XGBoost (yixi5-tuned) at seed 0 ===", flush=True)
    lgb_model = _fit_lgb(frames, y, users, LGB_CANDIDATE_COLUMNS, TRAIN_SEED)
    xgb_model = _fit_xgb(frames, y, users, XGB_COLUMNS, TRAIN_SEED)
    lgb_va = lgb_model.predict(frames['valid'][LGB_CANDIDATE_COLUMNS])
    lgb_te = lgb_model.predict(frames['test'][LGB_CANDIDATE_COLUMNS])
    xgb_va = xgb_model.predict(frames['valid'][XGB_COLUMNS])
    xgb_te = xgb_model.predict(frames['test'][XGB_COLUMNS])
    print(f"  LGB standalone valid={evaluate(users['valid'], y['valid'], lgb_va)['primary']:.5f}", flush=True)
    print(f"  XGB standalone valid={evaluate(users['valid'], y['valid'], xgb_va)['primary']:.5f}", flush=True)

    print("\n=== [3/4] training FM 5-seed ensemble (iter38 exact config) ===", flush=True)
    splits = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    enc, dim = encode_ext(splits, feature_set=FEATURES, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                           alpha=0.5, n_buckets=20)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    assert np.array_equal(np.asarray(yva), y['valid']), "FM/native valid label order mismatch"
    assert np.array_equal(np.asarray(uva), np.asarray(users['valid'])), "FM/native valid user order mismatch"
    assert np.array_equal(np.asarray(yte), y['test']), "FM/native test label order mismatch"
    assert np.array_equal(np.asarray(ute), np.asarray(users['test'])), "FM/native test user order mismatch"

    fm_va_scores, fm_te_scores = [], []
    for seed in SEEDS:
        print(f"  training FM seed {seed}...", flush=True)
        m = train_one_fm(Xtr, ytr, utr, Xva, yva, uva, splits['train'], dim, seed)
        fm_va_scores.append(sigmoid(m.predict(Xva)))
        fm_te_scores.append(sigmoid(m.predict(Xte)))
    fm_va = np.mean(np.stack(fm_va_scores), axis=0)
    fm_te = np.mean(np.stack(fm_te_scores), axis=0)
    print(f"  FM ensemble standalone valid={evaluate(uva, yva, fm_va)['primary']:.5f}", flush=True)

    print(f"\n=== [4/4] blending (within-user percentile, {WEIGHTS['fm']:.0%} FM / {WEIGHTS['lgb']:.0%} LGB / {WEIGHTS['xgb']:.0%} XGB) ===", flush=True)
    va_components = {
        'fm': within_user_percentile(fm_va, users['valid']),
        'lgb': within_user_percentile(lgb_va, users['valid']),
        'xgb': within_user_percentile(xgb_va, users['valid']),
    }
    te_components = {
        'fm': within_user_percentile(fm_te, users['test']),
        'lgb': within_user_percentile(lgb_te, users['test']),
        'xgb': within_user_percentile(xgb_te, users['test']),
    }
    blend_va = sum(WEIGHTS[k] * va_components[k] for k in WEIGHTS)
    blend_te = sum(WEIGHTS[k] * te_components[k] for k in WEIGHTS)
    va_metrics = evaluate(users['valid'], y['valid'], blend_va)
    te_metrics = evaluate(users['test'], y['test'], blend_te)
    print(f"\nfinal blend: valid primary={va_metrics['primary']:.5f}  test primary={te_metrics['primary']:.5f}", flush=True)

    raw_test_rows = load(DATA_DIR)['test']
    assert len(raw_test_rows) == len(frames['test']), \
        f"row count mismatch: data.load() test={len(raw_test_rows)} vs feature frame test={len(frames['test'])}"
    write_submission(out_path, raw_test_rows, blend_te)
    print(f"wrote {out_path} ({len(raw_test_rows)} rows)")

    checked_scores = read_submission(out_path, raw_test_rows)
    assert len(checked_scores) == len(raw_test_rows)
    print(f"submit.py format/alignment check: PASSED ({len(checked_scores)} rows)")

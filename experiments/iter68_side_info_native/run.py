"""iter68: retest static side-info features on the GBM-NATIVE representation
(iter63's rate_only feature set), not the old FM-bucketed representation
iter15 rejected them on. Also adds a genuinely new file never touched by
ANY iteration so far in this project's history: video_features_basic_pure.csv
(video_type, upload_type, primary tag, music_id) -- iter15 only ever tried
user_features_pure.csv and video_features_statistic_pure.csv.

Motivation: the whole reason iter44 became "NEW BEST, promoted" was that
un-bucketed/native categorical+numeric handling let LightGBM see far more
signal than the FM's bucketed encoding did (iter41's shortfall). iter15's
static side-info rejection predates that finding by ~30 iterations and was
never retested under the native representation -- a real, previously
unexamined gap, not a new idea invented from nothing.

Ablation (single seed=0, LightGBM linear_tree=True config, iter55's winning
hyperparameters unchanged): baseline (iter63 rate_only, no side info) vs.
+user (demographic/account-state, 6 fields, iter15's exact selection) vs.
+video_stat (engagement counts, 5 fields, iter15's exact selection,
log1p-transformed here since native GBM doesn't need iter15's bucketing) vs.
+video_basic (NEW: video_type, upload_type, primary tag, music_id) vs. +all.
100% join coverage confirmed for all three side tables against every split's
video_id/user_id (checked directly, see driver notes) -- no UNK fallback path
is actually exercised, but one is still defined defensively.
"""
import os, sys, csv, importlib.util
import numpy as np
import pandas as pd
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
from evaluate import evaluate  # noqa: E402

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
_ITER63_DIR = os.path.join(_REPO_ROOT, 'experiments', 'iter63_decay_tab_rate')


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t63 = _load_module(os.path.join(_ITER63_DIR, 'train.py'), 'iter63_train_for_68')

USER_CSV_COLS = ['user_active_degree', 'is_live_streamer', 'is_video_author',
                  'follow_user_num_range', 'fans_user_num_range', 'register_days_range']
USER_FIELDS = ['u_active_degree', 'u_live_streamer', 'u_video_author',
               'u_follow_range', 'u_fans_range', 'u_register_range']

VIDEO_STAT_CSV_COLS = ['play_cnt', 'like_cnt', 'share_cnt', 'complete_play_cnt', 'follow_cnt']
VIDEO_STAT_FIELDS = ['v_play', 'v_like', 'v_share', 'v_complete', 'v_follow']

VIDEO_BASIC_CAT_FIELDS = ['v_type', 'v_upload_type', 'v_tag_primary']


def _load_user_lut():
    lut = {}
    with open(os.path.join(DATA_DIR, 'user_features_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            lut[r['user_id']] = tuple(r[c] for c in USER_CSV_COLS)
    return lut


def _load_video_stat_lut():
    lut = {}
    with open(os.path.join(DATA_DIR, 'video_features_statistic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            lut[r['video_id']] = tuple(float(r[c]) for c in VIDEO_STAT_CSV_COLS)
    return lut


def _load_video_basic_lut():
    lut = {}
    with open(os.path.join(DATA_DIR, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            tag = r['tag']
            primary_tag = tag.split(',')[0] if tag else 'UNK'
            lut[r['video_id']] = (r['video_type'], r['upload_type'], primary_tag)
    return lut


def augment(dfs, which):
    """which: subset of {'user', 'video_stat', 'video_basic'}"""
    new_cats, new_nums = [], []
    dfs = {name: df.copy() for name, df in dfs.items()}
    vid_raw = {name: dfs[name]['video_id'].astype(str) for name in dfs}
    uid_raw = {name: dfs[name]['user_id'].astype(str) for name in dfs}

    if 'user' in which:
        lut = _load_user_lut()
        for i, f in enumerate(USER_FIELDS):
            for name in dfs:
                dfs[name][f] = uid_raw[name].map(lambda k: lut.get(k, ('UNK',) * len(USER_FIELDS))[i])
            new_cats.append(f)

    if 'video_stat' in which:
        lut = _load_video_stat_lut()
        default = (0.0,) * len(VIDEO_STAT_FIELDS)
        for i, f in enumerate(VIDEO_STAT_FIELDS):
            for name in dfs:
                dfs[name][f] = vid_raw[name].map(lambda k: np.log1p(lut.get(k, default)[i]))
            new_nums.append(f)

    if 'video_basic' in which:
        lut = _load_video_basic_lut()
        default = ('UNK', 'UNK', 'UNK')
        for i, f in enumerate(VIDEO_BASIC_CAT_FIELDS):
            for name in dfs:
                dfs[name][f] = vid_raw[name].map(lambda k: lut.get(k, default)[i])
            new_cats.append(f)

    cats = {c: pd.CategoricalDtype(categories=dfs['train'][c].unique()) for c in new_cats}
    for name in dfs:
        for c in new_cats:
            dfs[name][c] = dfs[name][c].astype(cats[c])
    return dfs, new_cats, new_nums


def train_eval(dfs, y, u, cat_cols, num_cols, seed=0, verbose=False, tag=''):
    Xtr, ytr, utr = t63._sort_by_user(dfs['train'][cat_cols + num_cols], y['train'], u['train'])
    Xva, yva, uva = t63._sort_by_user(dfs['valid'][cat_cols + num_cols], y['valid'], u['valid'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]
    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=2, learning_rate=0.10, n_estimators=500, min_child_samples=200,
        reg_lambda=1.0, random_state=seed, verbosity=-1, n_jobs=-1, linear_tree=True,
    )
    model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    va_scores = model.predict(dfs['valid'][cat_cols + num_cols])
    te_scores = model.predict(dfs['test'][cat_cols + num_cols])
    va = evaluate(u['valid'], y['valid'], va_scores)
    te = evaluate(u['test'], y['test'], te_scores)
    if verbose:
        print(f"[{tag}] seed={seed} valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)
    return va, te


if __name__ == '__main__':
    print("=== preparing iter63 rate_only base features (cached) ===", flush=True)
    dfs0, y, u = t63.prepare(DATA_DIR, 'rate_only')
    base_cat, base_num = t63.CAT_COLS, t63.VARIANT_NUM_COLS['rate_only']

    print("\n=== harness-fidelity check: reproduce iter63 baseline exactly ===", flush=True)
    va0, te0 = train_eval(dfs0, y, u, base_cat, base_num, seed=0, verbose=True, tag='baseline (fidelity check)')
    print(f"  expect valid=0.67168 test=0.65353", flush=True)
    assert abs(va0['primary'] - 0.67168) < 1e-4 and abs(te0['primary'] - 0.65353) < 1e-4, "harness fidelity check FAILED"
    print("  PASS", flush=True)

    configs = [
        ('user', {'user'}),
        ('video_stat', {'video_stat'}),
        ('video_basic', {'video_basic'}),
        ('all', {'user', 'video_stat', 'video_basic'}),
    ]

    results = {'baseline': (va0['primary'], te0['primary'])}
    for tag, which in configs:
        print(f"\n=== variant: +{tag} ===", flush=True)
        dfs_aug, new_cats, new_nums = augment(dfs0, which)
        va, te = train_eval(dfs_aug, y, u, base_cat + new_cats, base_num + new_nums, seed=0, verbose=True, tag=tag)
        results[tag] = (va['primary'], te['primary'])

    print("\n=== summary (seed 0) ===")
    print(f"{'variant':<14} {'valid':>9} {'test':>9} {'Δvalid':>9} {'Δtest':>9}")
    bva, bte = results['baseline']
    for tag in ['baseline'] + [c[0] for c in configs]:
        v, tt = results[tag]
        print(f"{tag:<14} {v:>9.5f} {tt:>9.5f} {v-bva:>+9.5f} {tt-bte:>+9.5f}")

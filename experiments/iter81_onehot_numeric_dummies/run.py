"""iter81: does exposing low-cardinality static side-info as one-hot NUMERIC
dummy columns (instead of native LightGBM categoricals) let it matter at
num_leaves=2?

Direct follow-up to iter80's diagnosis: at num_leaves=2 there is exactly one
split in the whole tree, already won by tab/the decay-rate features, and a
*categorical* column only affects linear_tree's prediction by winning that
split -- the leaf-linear regression only regresses on NUMERIC inputs. Every
categorical side-info block tried so far (iter68's demographic fields,
iter75's video_type/music_type, iter80's onehot_feat0..17) was a clean
no-op under exactly this mechanism, regardless of content or cardinality.
This iteration tests whether that's really a content problem or purely a
representation problem, by re-encoding the SAME already-rejected
low-cardinality fields as binary 0/1 numeric dummy columns, which
linear_tree's leaf regression can use directly without needing to win any
split.

Scope: only LOW-cardinality fields (<=10 distinct train values), to keep the
dummy expansion bounded and the test cheap/interpretable:
  - iter68's 6 demographic fields (all already no-ops as categoricals)
  - iter75's video_type + upload_type (already no-ops as categoricals)
  - iter80's 8 lowest-cardinality onehot_feat columns (0,6,9,10,11,12,13,14,
    15,16,17 all have nunique<=7; capped at the lowest 8 to bound expansion)
High-cardinality fields (onehot_feat3 at 1471 distinct, video tag at 44,
etc.) are deliberately excluded -- one-hot expanding those would blow up
dimensionality far beyond what a bounded, interpretable single ablation
should attempt, and is a different (much larger) experiment if this one
finds signal.

Reuses iter63's own prepare()/hyperparameters unchanged. Ablates the three
blocks separately and combined; harness-fidelity check first.
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


t63 = _load_module(os.path.join(_ITER63_DIR, 'train.py'), 'iter63_train_for_81')

USER_CSV_COLS = ['user_active_degree', 'is_live_streamer', 'is_video_author',
                  'follow_user_num_range', 'fans_user_num_range', 'register_days_range']
ONEHOT_LOWCARD_COLS = ['onehot_feat0', 'onehot_feat6', 'onehot_feat9', 'onehot_feat10',
                        'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14']
VIDEO_BASIC_COLS = ['video_type', 'upload_type']


def _load_user_lut(cols):
    lut = {}
    with open(os.path.join(DATA_DIR, 'user_features_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            lut[r['user_id']] = tuple(('UNK' if r[c] == '' else r[c]) for c in cols)
    return lut


def _load_video_basic_lut(cols):
    lut = {}
    with open(os.path.join(DATA_DIR, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            lut[r['video_id']] = tuple(('UNK' if r[c] == '' else r[c]) for c in cols)
    return lut


def _dummy_block(dfs, lut, cols, join_key, prefix):
    """Returns (dfs_with_dummies, new_numeric_col_names). One-hot dummy
    columns fit on TRAIN's observed (col, value) pairs only; unseen
    values/keys at valid/test map to all-zero (standard held-out-category
    handling, matches how vocabs are built elsewhere in this project)."""
    default = ('UNK',) * len(cols)
    raw = {name: dfs[name][join_key].astype(str).map(lambda k: lut.get(k, default)) for name in dfs}
    new_cols = []
    for i, c in enumerate(cols):
        train_vals = sorted(set(v[i] for v in raw['train']))
        for val in train_vals:
            dcol = f'{prefix}_{c}_{val}'
            for name in dfs:
                dfs[name][dcol] = (raw[name].map(lambda v: v[i]) == val).astype(np.float32)
            new_cols.append(dcol)
    return dfs, new_cols


def augment(dfs0, which):
    dfs = {name: df.copy() for name, df in dfs0.items()}
    all_new = []
    if 'demo' in which:
        lut = _load_user_lut(USER_CSV_COLS)
        dfs, new = _dummy_block(dfs, lut, USER_CSV_COLS, 'user_id', 'demo')
        all_new += new
    if 'onehot' in which:
        lut = _load_user_lut(ONEHOT_LOWCARD_COLS)
        dfs, new = _dummy_block(dfs, lut, ONEHOT_LOWCARD_COLS, 'user_id', 'oh')
        all_new += new
    if 'video_basic' in which:
        lut = _load_video_basic_lut(VIDEO_BASIC_COLS)
        dfs, new = _dummy_block(dfs, lut, VIDEO_BASIC_COLS, 'video_id', 'vb')
        all_new += new
    return dfs, all_new


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
        print(f"[{tag}] seed={seed} valid={va['primary']:.5f} test={te['primary']:.5f} "
              f"(+{len(num_cols) - len(t63.VARIANT_NUM_COLS['rate_only'])} dummy cols)", flush=True)
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
        ('demo', {'demo'}),
        ('onehot', {'onehot'}),
        ('video_basic', {'video_basic'}),
        ('all', {'demo', 'onehot', 'video_basic'}),
    ]

    results = {'baseline': (va0['primary'], te0['primary'])}
    for tag, which in configs:
        dfs_aug, new_num = augment(dfs0, which)
        print(f"\n=== variant: +{tag} ({len(new_num)} dummy cols) ===", flush=True)
        va, te = train_eval(dfs_aug, y, u, base_cat, base_num + new_num, seed=0, verbose=True, tag=tag)
        results[tag] = (va['primary'], te['primary'])

    print("\n=== summary (seed 0) ===")
    print(f"{'variant':<14} {'valid':>9} {'test':>9} {'Δvalid':>9} {'Δtest':>9}")
    bva, bte = results['baseline']
    for tag in ['baseline'] + [c[0] for c in configs]:
        v, tt = results[tag]
        print(f"{tag:<14} {v:>9.5f} {tt:>9.5f} {v-bva:>+9.5f} {tt-bte:>+9.5f}")

"""6c: build leakage-free out-of-fold auxiliary-engagement features
(is_like/is_follow/is_comment/is_forward) for the GBM-native long_view
ranker, per XUXIA_INSTRUCTIONS.md Section 6c approach 1 (stacking, not
joint multi-task loss).

No-leakage argument (parallels iter31's for the FM-loss version, adapted
for stacking):
  - TRAIN rows: each row's aux-feature value comes from a K=5-fold
    out-of-fold LightGBM prediction -- the fold model that scores a given
    train row was never fit on that row (nor any other row in its fold),
    so a row's own aux label can't leak into its own feature value.
  - VALID/TEST rows: the aux-feature value comes from a model fit on
    100% of TRAIN only, then applied to valid/test purely from their
    input features -- valid/test aux labels are never read by any
    training step, so there is no leakage path into these splits either.
  - Inputs to the aux models are the same native feature set the main
    long_view ranker uses (categoricals + causal decay features). None of
    these encode the CURRENT row's own label (long_view or otherwise) --
    the decay features are strictly decayed counts of *past* long_view
    events (already causal by iter24/44's construction) -- so reusing
    them as inputs to predict a *different* same-row label (is_like etc.)
    introduces no new leakage beyond what's already verified for the main
    model's own features.

Alignment: `load_aux_labels` (iter31_multitask/data_ext.py) returns the 5
engagement labels in the same per-split row order as `_load_raw_time`'s
output, by construction (same two log files, same order, same disjoint
date-range predicate as every other data_ext.py in this repo) -- verified
below with an independent spot-check against iter63's own row order
(matching on user_id/video_id/tab, not just trusting the docstring).
"""
import os, sys, importlib.util, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)

DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')
AUX_TASKS = ('is_like', 'is_follow', 'is_comment', 'is_forward')
N_FOLDS = 5


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t63 = _load_module(os.path.join(_REPO_ROOT, 'experiments', 'iter63_decay_tab_rate', 'train.py'), 'iterXUXIA3_t63_train')
iter31_de = _load_module(os.path.join(_REPO_ROOT, 'experiments', 'iter31_multitask', 'data_ext.py'), 'iterXUXIA3_iter31_de')


def spot_check_alignment(dfs, aux_splits, n=30, seed=0):
    """Independently re-parses the raw CSVs (date-filtered, in file order)
    and checks that user_id/video_id/tab at a sample of row positions match
    dfs[...] at the same position -- confirms load_aux_labels' row order
    lines up with iter63's own row order, not just by docstring argument."""
    import csv
    from data import SPLITS
    vid2author = {}
    with open(os.path.join(DATA_DIR, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']
    raw = {name: [] for name in SPLITS}
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(DATA_DIR, f)) as fh:
            for r in csv.DictReader(fh):
                d = int(r['date'])
                for name, (lo, hi) in SPLITS.items():
                    if lo <= d <= hi:
                        raw[name].append((r['user_id'], r['video_id'], r['tab']))
                        break
    # user_id is cast to a pd.CategoricalDtype whose categories come from
    # TRAIN only (see prepare()) -- any valid/test user_id never seen in
    # train legitimately becomes NaN (out-of-vocabulary), same behavior the
    # main model's own user_id feature already has. That's an encoding
    # artifact, not a row-order bug, so it must not fail this check.
    # video_id + tab together are high-enough cardinality (and never
    # OOV-collapsed the same way here) to prove positional alignment on
    # their own.
    rng = np.random.default_rng(seed)
    n_checked, n_user_oov = 0, 0
    for name in ('train', 'valid', 'test'):
        n_rows = len(dfs[name])
        idxs = rng.choice(n_rows, size=min(n, n_rows), replace=False)
        for i in idxs:
            u_raw, v_raw, tab_raw = raw[name][i]
            u_df, v_df, tab_df = str(dfs[name]['user_id'].iloc[i]), str(dfs[name]['video_id'].iloc[i]), str(dfs[name]['tab'].iloc[i])
            assert v_raw == v_df and tab_raw == tab_df, \
                f"ALIGNMENT MISMATCH split={name} row={i}: raw=({u_raw},{v_raw},{tab_raw}) df=({u_df},{v_df},{tab_df})"
            if u_df == 'nan':
                n_user_oov += 1
            else:
                assert u_raw == u_df, \
                    f"ALIGNMENT MISMATCH (user_id) split={name} row={i}: raw={u_raw} df={u_df}"
            n_checked += 1
    print(f"  alignment spot-check PASSED ({n_checked} rows across train/valid/test, exact video_id/tab match; "
          f"{n_user_oov} rows had an out-of-vocabulary user_id -> NaN, expected pre-existing encoding behavior, "
          f"user_id matched exactly on the rest)")


def build_oof_features(variant='rate_only', use_cache=True):
    cache_path = os.path.join(_THIS_DIR, '.cache_aux_features.pkl')
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    dfs, y, u = t63.prepare(DATA_DIR, variant)
    aux_splits = iter31_de.load_aux_labels(DATA_DIR)
    for name in ('train', 'valid', 'test'):
        assert len(aux_splits[name][AUX_TASKS[0]]) == len(dfs[name]), \
            f"row count mismatch {name}: aux={len(aux_splits[name][AUX_TASKS[0]])} dfs={len(dfs[name])}"
    spot_check_alignment(dfs, aux_splits)

    cat_cols = ['user_id', 'video_id', 'author_id', 'tab', 'last1']
    n_train = len(dfs['train'])
    rng = np.random.default_rng(0)
    fold_id = rng.integers(0, N_FOLDS, size=n_train)

    oof = {'train': {}, 'valid': {}, 'test': {}}
    for task in AUX_TASKS:
        print(f"  aux task: {task}", flush=True)
        y_task_tr = aux_splits['train'][task]
        base_rate = y_task_tr.mean()
        print(f"    train base rate: {base_rate*100:.3f}%", flush=True)

        oof_train = np.zeros(n_train, dtype=np.float64)
        for k in range(N_FOLDS):
            tr_mask = fold_id != k
            va_mask = fold_id == k
            m = lgb.LGBMClassifier(objective='binary', num_leaves=2, linear_tree=True,
                                    learning_rate=0.10, n_estimators=300, min_child_samples=200,
                                    reg_lambda=1.0, random_state=0, verbosity=-1, n_jobs=-1)
            m.fit(dfs['train'].iloc[tr_mask], y_task_tr[tr_mask], categorical_feature=cat_cols)
            oof_train[va_mask] = m.predict_proba(dfs['train'].iloc[va_mask])[:, 1]
        oof['train'][task] = oof_train

        m_full = lgb.LGBMClassifier(objective='binary', num_leaves=2, linear_tree=True,
                                     learning_rate=0.10, n_estimators=300, min_child_samples=200,
                                     reg_lambda=1.0, random_state=0, verbosity=-1, n_jobs=-1)
        m_full.fit(dfs['train'], y_task_tr, categorical_feature=cat_cols)
        oof['valid'][task] = m_full.predict_proba(dfs['valid'])[:, 1]
        oof['test'][task] = m_full.predict_proba(dfs['test'])[:, 1]

    result = {'dfs': dfs, 'y': y, 'u': u, 'oof': oof}
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    return result


if __name__ == '__main__':
    r = build_oof_features(use_cache=False)
    for task in AUX_TASKS:
        for split in ('train', 'valid', 'test'):
            v = r['oof'][split][task]
            print(f"{task:12s} {split:6s} mean_pred={v.mean():.5f} min={v.min():.5f} max={v.max():.5f}")

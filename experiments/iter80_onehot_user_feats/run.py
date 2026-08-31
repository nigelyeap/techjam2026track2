"""iter80: user_features_pure.csv's 18 anonymized `onehot_feat0..17` columns,
GBM-native -- the one slice of the project's own data that has been
DELIBERATELY excluded by every iteration since iter15 ("anonymized/
undocumented -- left unused, out of scope"), and never revisited even after
iter68 retested the rest of user_features_pure.csv's non-onehot columns on
the GBM-native representation (iter15's 6 demographic fields were an exact
no-op there, 0 splits used).

These are NOT literal one-hot vectors despite the name -- inspection shows
each column is itself a discrete anonymized category code, cardinality
ranging from 2 (onehot_feat0) to 1471 (onehot_feat3), with 6 of the 18
columns (12-17) carrying ~2.6% nulls. Treated here as native LightGBM
categoricals (NaN -> 'UNK' sentinel), joined by user_id, exactly like
iter68's 'user' demographic block.

Motivation: iter68 already showed the *documented* demographic fields are
inert at num_leaves=2 (never selected as the tree's single split, since
'tab'/the decay-rate features already win that competition). These 18
columns are a categorically different resource -- anonymized, higher
cardinality in places (up to 1471 distinct values), never assessed at all.
Worth one clean, cheap ablation before concluding "no more user-side-info
signal exists in this dataset."

Reuses iter63's own prepare()/train harness unchanged (base cat/num cols,
hyperparameters); ablates all 18 columns together as a single block (not
one-by-one, to keep this a single cheap run) plus a fidelity check.
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


t63 = _load_module(os.path.join(_ITER63_DIR, 'train.py'), 'iter63_train_for_80')

ONEHOT_CSV_COLS = [f'onehot_feat{i}' for i in range(18)]
ONEHOT_FIELDS = [f'u_onehot{i}' for i in range(18)]


def _load_onehot_lut():
    lut = {}
    with open(os.path.join(DATA_DIR, 'user_features_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            lut[r['user_id']] = tuple(
                ('UNK' if r[c] == '' else r[c]) for c in ONEHOT_CSV_COLS)
    return lut


def augment_onehot(dfs):
    lut = _load_onehot_lut()
    default = ('UNK',) * len(ONEHOT_FIELDS)
    dfs = {name: df.copy() for name, df in dfs.items()}
    uid_raw = {name: dfs[name]['user_id'].astype(str) for name in dfs}
    for i, f in enumerate(ONEHOT_FIELDS):
        for name in dfs:
            dfs[name][f] = uid_raw[name].map(lambda k: lut.get(k, default)[i])
    cats = {c: pd.CategoricalDtype(categories=dfs['train'][c].unique()) for c in ONEHOT_FIELDS}
    for name in dfs:
        for c in ONEHOT_FIELDS:
            dfs[name][c] = dfs[name][c].astype(cats[c])
    return dfs, ONEHOT_FIELDS


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

    print("\n=== variant: +onehot (18 anonymized user categoricals) ===", flush=True)
    dfs_aug, new_cats = augment_onehot(dfs0)
    va1, te1 = train_eval(dfs_aug, y, u, base_cat + new_cats, base_num, seed=0, verbose=True, tag='onehot')

    print("\n=== summary (seed 0) ===")
    print(f"{'variant':<10} {'valid':>9} {'test':>9} {'Δvalid':>9} {'Δtest':>9}")
    bva, bte = va0['primary'], te0['primary']
    print(f"{'baseline':<10} {bva:>9.5f} {bte:>9.5f} {0.0:>+9.5f} {0.0:>+9.5f}")
    v, tt = va1['primary'], te1['primary']
    print(f"{'+onehot':<10} {v:>9.5f} {tt:>9.5f} {v-bva:>+9.5f} {tt-bte:>+9.5f}")

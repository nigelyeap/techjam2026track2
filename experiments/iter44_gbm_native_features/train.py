"""iter44: give GBMs (LightGBM/CatBoost) their OWN native feature
representation instead of reusing FM's.

Diagnosis from iter41/iter43: both LightGBM (best valid 0.6342 after a
7-point hyperparameter sweep) and CatBoost (best valid 0.6105 after a
5-point sweep, even at shallow depth=3) badly underperform FM (0.6389),
and neither responds much to capacity/regularization tuning. The common
suspect: every continuous signal (decay_rate, decay_act, decay_tab,
lastk_rate, gap) is pre-quantized into `n_buckets=20` categorical buckets
by `encode_ext` -- necessary for FM (which only knows how to embed
categories) but actively throws away ordering/magnitude information that
a GBM's own split-finding is specifically built to exploit. Forcing a
GBM through FM's discretization plays to none of its strengths.

This iteration builds a GBM-native encoding directly from iter27's raw
causal row tuples (via `load_ext`, unmodified): true categoricals
(user_id, video_id, author_id, tab, last1) stay categorical; every
continuous signal is passed as a raw float (ratios un-bucketed, gap's
"first row" case as NaN so the GBM can route it with its native
missing-value handling instead of a synthetic 'UNK' category).
"""
import os, sys, importlib.util
import numpy as np
import pandas as pd
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402

_ITER27_DIR = os.path.join(_THIS_DIR, '..', 'iter27_triple_fusion')


def _load_iter27_data_ext():
    path = os.path.join(_ITER27_DIR, 'data_ext.py')
    spec = importlib.util.spec_from_file_location('iter27_data_ext_v3', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_de = _load_iter27_data_ext()
ALPHA = 0.5  # matches iter27/iter38's winning Laplace-smoothing constant


def _row_to_dict(x):
    pcol, tcol = _de._halflife_col(2.5, _de.HALFLIVES)
    ttcol = _de._tab_halflife_col(3, _de.HALFLIVES, _de.TAB_HALFLIVES)
    gap = x[_de.IDX['gap_ms']]
    last1 = x[_de.IDX['last1']]
    return {
        'user_id': x[_de.IDX['user_id']],
        'video_id': x[_de.IDX['video_id']],
        'author_id': x[_de.IDX['author_id']],
        'tab': x[_de.IDX['tab']],
        'last1': str(int(last1)) if last1 != -1 else 'UNK',
        'duration_ms': float(x[_de.IDX['duration_ms']]),
        'decay_rate_2.5': (x[pcol] + ALPHA) / (x[tcol] + 2 * ALPHA),
        'decay_act_2.5': float(x[tcol]),
        'decay_tab_3': float(x[ttcol]),
        'lastk_rate': (x[_de.IDX['lastk_sum']] + ALPHA) / (x[_de.IDX['lastk_cnt']] + 2 * ALPHA),
        'gap': float(gap) if gap >= 0 else np.nan,
    }


CAT_COLS = ['user_id', 'video_id', 'author_id', 'tab', 'last1']
NUM_COLS = ['duration_ms', 'decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'lastk_rate', 'gap']


def prepare(data_dir, use_cache=True):
    splits = _de.load_ext(data_dir, use_cache=use_cache)
    dfs = {}
    for name in ('train', 'valid', 'test'):
        rows = [_row_to_dict(x) for x in splits[name]]
        df = pd.DataFrame(rows)
        dfs[name] = df
    # fit categorical dtype on TRAIN categories only; unseen valid/test values
    # fall back to NaN (a real "unknown category" at inference time -- no
    # leakage, since the categories are fit from train only).
    cats = {c: pd.CategoricalDtype(categories=dfs['train'][c].unique()) for c in CAT_COLS}
    for name in dfs:
        for c in CAT_COLS:
            dfs[name][c] = dfs[name][c].astype(cats[c])
    y = {name: np.array([x[_de.IDX['label']] for x in splits[name]], dtype=np.float32) for name in dfs}
    u = {name: [x[_de.IDX['user_id']] for x in splits[name]] for name in dfs}
    return dfs, y, u


def _sort_by_user(df, y, u):
    u = np.asarray(u)
    order = np.argsort(u, kind='stable')
    return df.iloc[order].reset_index(drop=True), y[order], u[order]


def run(data_dir, num_leaves=15, learning_rate=0.05, n_estimators=500,
        min_child_samples=200, reg_lambda=1.0, seed=0, verbose=False, _cache=None):
    dfs, y, u = prepare(data_dir) if _cache is None else _cache
    Xtr, ytr, utr = _sort_by_user(dfs['train'], y['train'], u['train'])
    Xva, yva, uva = _sort_by_user(dfs['valid'], y['valid'], u['valid'])
    Xte, yte, ute = _sort_by_user(dfs['test'], y['test'], u['test'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]

    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=num_leaves, learning_rate=learning_rate,
        n_estimators=n_estimators, min_child_samples=min_child_samples,
        reg_lambda=reg_lambda, random_state=seed, verbosity=-1, n_jobs=-1,
    )
    model.fit(
        Xtr, ytr, group=gtr,
        eval_set=[(Xva, yva)], eval_group=[gva],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    va_scores = model.predict(dfs['valid'])
    te_scores = model.predict(dfs['test'])
    va = evaluate(u['valid'], y['valid'], va_scores)
    te = evaluate(u['test'], y['test'], te_scores)
    if verbose:
        print(f"best_iteration={model.best_iteration_}  valid={va}  test={te}")
    return model, va, te, (dfs, y, u)


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
    model, va, te, _cache = run(DATA_DIR, verbose=True)
    print("valid:", va)
    print("test:", te)

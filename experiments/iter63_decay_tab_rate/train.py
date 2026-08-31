"""iter63: decayed per-tab RATE (decay_tab_rate_3) as a GBM-native feature,
replacing/augmenting the raw decay_tab_3 COUNT that's been in every feature
set since iter24.

Uses this dir's own data_ext.py (extended to track decayed_tab_total, see
its module docstring), otherwise a straight copy of iter44's GBM-native
`_row_to_dict`/`prepare`/`run` plumbing plus iter51's linear_tree=True
model config (this run's actual best-known GBM setup, per iter55).

Three NUM_COLS variants compared, single seed=0, iter55's exact winning
hyperparameters (linear_tree=True, learning_rate=0.10, num_leaves=2,
n_estimators=500, min_child_samples=200, reg_lambda=1.0) unchanged:
  - baseline: decay_tab_3 (count) only, exactly like iter44/51/55
  - +rate:    decay_tab_3 (count) AND decay_tab_rate_3 (rate), both present
  - rate_only: decay_tab_rate_3 (rate) REPLACES decay_tab_3 (count)
"""
import os, sys, importlib.util
import numpy as np
import pandas as pd
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_de = _load_module(os.path.join(_THIS_DIR, 'data_ext.py'), 'iter63_data_ext')
ALPHA = 0.5  # matches iter27/iter38/iter44's winning Laplace-smoothing constant

CAT_COLS = ['user_id', 'video_id', 'author_id', 'tab', 'last1']
BASE_NUM_COLS = ['duration_ms', 'decay_rate_2.5', 'decay_act_2.5', 'lastk_rate', 'gap']


def _row_to_dict(x, variant):
    pcol, tcol = _de._halflife_col(2.5, _de.HALFLIVES)
    ttcol_pos = _de._tab_halflife_col(3, _de.HALFLIVES, _de.TAB_HALFLIVES)
    ttcol_tot = _de._tab_halflife_total_col(3, _de.HALFLIVES, _de.TAB_HALFLIVES)
    gap = x[_de.IDX['gap_ms']]
    last1 = x[_de.IDX['last1']]
    d = {
        'user_id': x[_de.IDX['user_id']],
        'video_id': x[_de.IDX['video_id']],
        'author_id': x[_de.IDX['author_id']],
        'tab': x[_de.IDX['tab']],
        'last1': str(int(last1)) if last1 != -1 else 'UNK',
        'duration_ms': float(x[_de.IDX['duration_ms']]),
        'decay_rate_2.5': (x[pcol] + ALPHA) / (x[tcol] + 2 * ALPHA),
        'decay_act_2.5': float(x[tcol]),
        'lastk_rate': (x[_de.IDX['lastk_sum']] + ALPHA) / (x[_de.IDX['lastk_cnt']] + 2 * ALPHA),
        'gap': float(gap) if gap >= 0 else np.nan,
    }
    tab_pos = x[ttcol_pos]
    tab_tot = x[ttcol_tot]
    tab_rate = (tab_pos + ALPHA) / (tab_tot + 2 * ALPHA)
    if variant == 'baseline':
        d['decay_tab_3'] = float(tab_pos)
    elif variant == 'plus_rate':
        d['decay_tab_3'] = float(tab_pos)
        d['decay_tab_rate_3'] = float(tab_rate)
    elif variant == 'rate_only':
        d['decay_tab_rate_3'] = float(tab_rate)
    else:
        raise ValueError(variant)
    return d


VARIANT_NUM_COLS = {
    'baseline': BASE_NUM_COLS + ['decay_tab_3'],
    'plus_rate': BASE_NUM_COLS + ['decay_tab_3', 'decay_tab_rate_3'],
    'rate_only': BASE_NUM_COLS + ['decay_tab_rate_3'],
}


def prepare(data_dir, variant, use_cache=True):
    splits = _de.load_ext(data_dir, use_cache=use_cache)
    dfs = {}
    for name in ('train', 'valid', 'test'):
        rows = [_row_to_dict(x, variant) for x in splits[name]]
        df = pd.DataFrame(rows)
        dfs[name] = df
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


def run(data_dir, variant, linear_tree=True, num_leaves=2, learning_rate=0.10,
        n_estimators=500, min_child_samples=200, reg_lambda=1.0, seed=0,
        verbose=False, _cache=None):
    dfs, y, u = prepare(data_dir, variant) if _cache is None else _cache
    Xtr, ytr, utr = _sort_by_user(dfs['train'], y['train'], u['train'])
    Xva, yva, uva = _sort_by_user(dfs['valid'], y['valid'], u['valid'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]

    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=num_leaves, learning_rate=learning_rate,
        n_estimators=n_estimators, min_child_samples=min_child_samples,
        reg_lambda=reg_lambda, random_state=seed, verbosity=-1, n_jobs=-1,
        linear_tree=linear_tree,
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
        print(f"[{variant}] best_iteration={model.best_iteration_}  valid={va['primary']:.5f}  test={te['primary']:.5f}", flush=True)
    return model, va, te, (dfs, y, u)


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

    for variant in ('baseline', 'plus_rate', 'rate_only'):
        _, va, te, _ = run(DATA_DIR, variant, seed=0, verbose=True)
    print("\n[expect baseline to reproduce iter55: valid=0.67052 test=0.65277]")

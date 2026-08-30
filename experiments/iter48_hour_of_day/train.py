"""iter48: add time-of-day as a GBM-native feature.

`hourmin` (HHMM, e.g. 1900 = 19:00) has been carried in every row tuple
since iter18 (via iter27's data_ext.py `IDX['hourmin']`), but has only
ever been used internally for time-ordering (sorting by time_ms) -- it
has never once been passed to a model as an actual input feature across
44+ iterations. Time-of-day is a classic recsys signal (viewing behavior
differs by hour) and is known at inference time (it's intrinsic to the
current impression, same causal status as `tab` or `duration_ms`), so
this is a genuinely untried, cheap lever, distinct from the decay/
recency-window feature family iter18-44 already explored.

Built directly on iter44's GBM-native pipeline (same CAT_COLS/NUM_COLS,
same winning hyperparameters) with hour-of-day added as two additional
NUM_COLS: sin/cos of the fractional hour, to preserve the 24h wraparound
(23:00 and 00:00 are adjacent, which a raw linear hour feature would not
capture).
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
    spec = importlib.util.spec_from_file_location('iter48_data_ext_v3', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_de = _load_iter27_data_ext()
ALPHA = 0.5


def _row_to_dict(x):
    pcol, tcol = _de._halflife_col(2.5, _de.HALFLIVES)
    ttcol = _de._tab_halflife_col(3, _de.HALFLIVES, _de.TAB_HALFLIVES)
    gap = x[_de.IDX['gap_ms']]
    last1 = x[_de.IDX['last1']]
    hourmin = int(x[_de.IDX['hourmin']])
    hour_frac = (hourmin // 100) + (hourmin % 100) / 60.0
    theta = 2 * np.pi * hour_frac / 24.0
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
        'hour_sin': float(np.sin(theta)),
        'hour_cos': float(np.cos(theta)),
    }


CAT_COLS = ['user_id', 'video_id', 'author_id', 'tab', 'last1']
NUM_COLS = ['duration_ms', 'decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'lastk_rate', 'gap',
            'hour_sin', 'hour_cos']


def prepare(data_dir, use_cache=True):
    splits = _de.load_ext(data_dir, use_cache=use_cache)
    dfs = {}
    for name in ('train', 'valid', 'test'):
        rows = [_row_to_dict(x) for x in splits[name]]
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


def run(data_dir, num_leaves=2, learning_rate=0.05, n_estimators=500,
        min_child_samples=200, reg_lambda=1.0, seed=0, verbose=False, _cache=None):
    dfs, y, u = prepare(data_dir) if _cache is None else _cache
    Xtr, ytr, utr = _sort_by_user(dfs['train'], y['train'], u['train'])
    Xva, yva, uva = _sort_by_user(dfs['valid'], y['valid'], u['valid'])
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
    # iter44's exact winning hyperparameters (num_leaves=2 default above)
    model, va, te, cache = run(DATA_DIR, verbose=True)
    print("valid:", va)
    print("test:", te)
    print(f"\n[harness check] baseline iter44 (no hour feature): valid=0.66135 test=0.64794")
    print(f"[iter48] with hour_sin/hour_cos: valid={va['primary']:.5f} test={te['primary']:.5f}")

    # seed robustness, only if the first result clears the promotion-look threshold
    if va['primary'] > 0.66135 + 0.0003:
        print("\n=== gain clears 0.0003 look-threshold, checking 4 more seeds ===")
        vas = [va['primary']]
        for s in (1, 2, 3, 4):
            _, va_s, te_s, _ = run(DATA_DIR, seed=s, _cache=cache)
            print(f"  seed={s} valid={va_s['primary']:.5f} test={te_s['primary']:.5f}")
            vas.append(va_s['primary'])
        print(f"  mean valid over 5 seeds: {np.mean(vas):.5f} (baseline 0.66135)")
    else:
        print("\n(gain does not clear the 0.0003 look-threshold -- no further seeds run)")

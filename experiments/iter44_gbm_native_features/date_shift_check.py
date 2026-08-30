"""iter44g: date-shift robustness check for the num_leaves=2 GBM result,
mirroring iter29/iter35's methodology (rerun the exact winning config on a
3-day-earlier-shifted split: train 2022-04-05..18 / valid 04-19..25 /
test 04-26..05-05) to check the result isn't an artifact of the specific
official date split.

This is a robustness/diagnostic check, not a promotion candidate -- a
shifted split is not the official split; nothing here changes the
official-split verdict already in RESULT.md regardless of the numbers.
"""
import os, sys, importlib.util
import numpy as np
import pandas as pd
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402

_ITER27_DIR = os.path.join(_THIS_DIR, '..', 'iter27_triple_fusion')
DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

SPLITS_SHIFTED = {'train': (20220405, 20220418),
                   'valid': (20220419, 20220425),
                   'test':  (20220426, 20220505)}


def _load_iter27_data_ext():
    path = os.path.join(_ITER27_DIR, 'data_ext.py')
    spec = importlib.util.spec_from_file_location('iter27_data_ext_shifted', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row_to_dict(de, x):
    pcol, tcol = de._halflife_col(2.5, de.HALFLIVES)
    ttcol = de._tab_halflife_col(3, de.HALFLIVES, de.TAB_HALFLIVES)
    gap = x[de.IDX['gap_ms']]
    last1 = x[de.IDX['last1']]
    ALPHA = 0.5
    return {
        'user_id': x[de.IDX['user_id']],
        'video_id': x[de.IDX['video_id']],
        'author_id': x[de.IDX['author_id']],
        'tab': x[de.IDX['tab']],
        'last1': str(int(last1)) if last1 != -1 else 'UNK',
        'duration_ms': float(x[de.IDX['duration_ms']]),
        'decay_rate_2.5': (x[pcol] + ALPHA) / (x[tcol] + 2 * ALPHA),
        'decay_act_2.5': float(x[tcol]),
        'decay_tab_3': float(x[ttcol]),
        'lastk_rate': (x[de.IDX['lastk_sum']] + ALPHA) / (x[de.IDX['lastk_cnt']] + 2 * ALPHA),
        'gap': float(gap) if gap >= 0 else np.nan,
    }


CAT_COLS = ['user_id', 'video_id', 'author_id', 'tab', 'last1']


def _sort_by_user(df, y, u):
    u = np.asarray(u)
    order = np.argsort(u, kind='stable')
    return df.iloc[order].reset_index(drop=True), y[order], u[order]


def run_one(de, splits, num_leaves, seed=0):
    dfs = {}
    for name in ('train', 'valid', 'test'):
        rows = [_row_to_dict(de, x) for x in splits[name]]
        dfs[name] = pd.DataFrame(rows)
    cats = {c: pd.CategoricalDtype(categories=dfs['train'][c].unique()) for c in CAT_COLS}
    for name in dfs:
        for c in CAT_COLS:
            dfs[name][c] = dfs[name][c].astype(cats[c])
    y = {name: np.array([x[de.IDX['label']] for x in splits[name]], dtype=np.float32) for name in dfs}
    u = {name: [x[de.IDX['user_id']] for x in splits[name]] for name in dfs}

    Xtr, ytr, utr = _sort_by_user(dfs['train'], y['train'], u['train'])
    Xva, yva, uva = _sort_by_user(dfs['valid'], y['valid'], u['valid'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]

    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=num_leaves, learning_rate=0.05, n_estimators=500,
        min_child_samples=200, reg_lambda=1.0, random_state=seed, verbosity=-1, n_jobs=-1,
    )
    model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    va_scores = model.predict(dfs['valid'])
    te_scores = model.predict(dfs['test'])
    va = evaluate(u['valid'], y['valid'], va_scores)
    te = evaluate(u['test'], y['test'], te_scores)
    return va, te


def main():
    de = _load_iter27_data_ext()

    print("=== OFFICIAL split (harness-fidelity reference vs known RESULT.md numbers) ===", flush=True)
    splits_official = de.load_ext(DATA_DIR, use_cache=True)
    for nl in [2, 7, 15]:
        va, te = run_one(de, splits_official, num_leaves=nl)
        print(f"  num_leaves={nl}: valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)

    print("\n=== SHIFTED split (train 04-05..18 / valid 04-19..25 / test 04-26..05-05) ===", flush=True)
    de.SPLITS = SPLITS_SHIFTED
    splits_shifted = de.load_ext(DATA_DIR, use_cache=False)
    for nl in [2, 7, 15]:
        va, te = run_one(de, splits_shifted, num_leaves=nl)
        print(f"  num_leaves={nl}: valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)


if __name__ == '__main__':
    main()

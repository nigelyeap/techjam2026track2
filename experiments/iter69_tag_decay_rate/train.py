"""iter69: decayed per-tag RATE as a GBM-native feature, added on top of
iter63's rate_only feature set (which already includes decay_tab_rate_3).
Same Laplace-smoothed-rate mechanism as iter63, applied to the new
(user, content-tag) key instead of (user, tab) -- see this dir's
data_ext.py module docstring for the full motivation (iter68 found raw
tag-as-categorical hurts badly; this tests the smoothed-rate reshaping
that worked for tab in iter63).

Variants compared, single seed=0 first (harness-fidelity check against
iter63's exact rate_only baseline), then 5-seed confirm whichever wins by
>0.001 valid:
  - baseline:      iter63's rate_only feature set, unchanged
  - +tag_rate_h3:  baseline + decay_tag_rate (halflife=3d)
  - +tag_rate_h7:  baseline + decay_tag_rate (halflife=7d)
  - +tag_rate_both: baseline + both halflives
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


_de = _load_module(os.path.join(_THIS_DIR, 'data_ext.py'), 'iter69_data_ext')
ALPHA = 0.5  # matches iter63/iter27/iter38/iter44's winning Laplace-smoothing constant

CAT_COLS = ['user_id', 'video_id', 'author_id', 'tab', 'last1']
BASE_NUM_COLS = ['duration_ms', 'decay_rate_2.5', 'decay_act_2.5', 'lastk_rate', 'gap', 'decay_tab_rate_3']


def _row_to_dict(x, variant):
    pcol, tcol = _de._halflife_col(2.5, _de.HALFLIVES)
    ttcol_pos = _de._tab_halflife_col(3, _de.HALFLIVES, _de.TAB_HALFLIVES)
    ttcol_tot = _de._tab_halflife_total_col(3, _de.HALFLIVES, _de.TAB_HALFLIVES)
    gap = x[_de.IDX['gap_ms']]
    last1 = x[_de.IDX['last1']]
    tab_pos = x[ttcol_pos]
    tab_tot = x[ttcol_tot]
    tab_rate = (tab_pos + ALPHA) / (tab_tot + 2 * ALPHA)
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
        'decay_tab_rate_3': float(tab_rate),
    }
    for h in _de.TAG_HALFLIVES:
        tgcol_pos = _de._tag_halflife_col(h, _de.HALFLIVES, _de.TAB_HALFLIVES, _de.TAG_HALFLIVES)
        tgcol_tot = _de._tag_halflife_total_col(h, _de.HALFLIVES, _de.TAB_HALFLIVES, _de.TAG_HALFLIVES)
        tag_pos = x[tgcol_pos]
        tag_tot = x[tgcol_tot]
        d[f'decay_tag_rate_{h}'] = float((tag_pos + ALPHA) / (tag_tot + 2 * ALPHA))
    return d


VARIANT_NUM_COLS = {
    'baseline': BASE_NUM_COLS,
    'tag_rate_h3': BASE_NUM_COLS + ['decay_tag_rate_3'],
    'tag_rate_h7': BASE_NUM_COLS + ['decay_tag_rate_7'],
    'tag_rate_both': BASE_NUM_COLS + ['decay_tag_rate_3', 'decay_tag_rate_7'],
}


def prepare(data_dir, use_cache=True):
    splits = _de.load_ext(data_dir, use_cache=use_cache)
    dfs = {}
    for name in ('train', 'valid', 'test'):
        rows = [_row_to_dict(x, None) for x in splits[name]]
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


def run(dfs, y, u, num_cols, seed=0, verbose=False, tag=''):
    Xtr, ytr, utr = _sort_by_user(dfs['train'][CAT_COLS + num_cols], y['train'], u['train'])
    Xva, yva, uva = _sort_by_user(dfs['valid'][CAT_COLS + num_cols], y['valid'], u['valid'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]
    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=2, learning_rate=0.10, n_estimators=500, min_child_samples=200,
        reg_lambda=1.0, random_state=seed, verbosity=-1, n_jobs=-1, linear_tree=True,
    )
    model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    va_scores = model.predict(dfs['valid'][CAT_COLS + num_cols])
    te_scores = model.predict(dfs['test'][CAT_COLS + num_cols])
    va = evaluate(u['valid'], y['valid'], va_scores)
    te = evaluate(u['test'], y['test'], te_scores)
    if verbose:
        print(f"[{tag}] seed={seed} valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)
    return model, va, te


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
    print("=== preparing features ===", flush=True)
    dfs, y, u = prepare(DATA_DIR)

    print("\n=== harness-fidelity check: reproduce iter63 rate_only baseline exactly ===", flush=True)
    _, va0, te0 = run(dfs, y, u, VARIANT_NUM_COLS['baseline'], seed=0, verbose=True, tag='baseline (fidelity check)')
    print("  expect valid=0.67168 test=0.65353", flush=True)
    assert abs(va0['primary'] - 0.67168) < 1e-4 and abs(te0['primary'] - 0.65353) < 1e-4, "harness fidelity check FAILED"
    print("  PASS", flush=True)

    results = {'baseline': (va0['primary'], te0['primary'])}
    for tag in ('tag_rate_h3', 'tag_rate_h7', 'tag_rate_both'):
        print(f"\n=== variant: {tag} ===", flush=True)
        _, va, te = run(dfs, y, u, VARIANT_NUM_COLS[tag], seed=0, verbose=True, tag=tag)
        results[tag] = (va['primary'], te['primary'])

    print("\n=== summary (seed 0) ===")
    print(f"{'variant':<16} {'valid':>9} {'test':>9} {'Δvalid':>9} {'Δtest':>9}")
    bva, bte = results['baseline']
    for tag in ['baseline', 'tag_rate_h3', 'tag_rate_h7', 'tag_rate_both']:
        v, tt = results[tag]
        print(f"{tag:<16} {v:>9.5f} {tt:>9.5f} {v-bva:>+9.5f} {tt-bte:>+9.5f}")

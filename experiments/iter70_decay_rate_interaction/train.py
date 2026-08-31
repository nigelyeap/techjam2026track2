"""iter70: explicit interaction terms between iter63's TWO proven decayed-rate
features (decay_rate_2.5, global per-user recency rate; decay_tab_rate_3,
per-(user,tab) recency rate) -- both already in the rate_only feature set,
never combined via an explicit cross term. linear_tree=True's leaf regression
is linear in the input features, so it cannot represent a genuine
multiplicative interaction between two numerics on its own; an explicit
product/ratio/diff term gives it that capacity directly. No new data
extraction needed -- both base features already exist in iter63's pipeline.
"""
import os, sys, importlib.util
import numpy as np
import pandas as pd
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _REPO_ROOT)
from evaluate import evaluate  # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t63 = _load_module(os.path.join(_REPO_ROOT, 'experiments', 'iter63_decay_tab_rate', 'train.py'), 'iter63_train_for_70')
DATA_DIR = os.path.join(_REPO_ROOT, 'KuaiRand-Pure', 'data')

BASE_NUM_COLS = t63.VARIANT_NUM_COLS['rate_only']  # duration_ms, decay_rate_2.5, decay_act_2.5, lastk_rate, gap, decay_tab_rate_3

VARIANTS = {
    'baseline': [],
    'product': ['x_rate_product'],
    'ratio': ['x_rate_ratio'],
    'diff': ['x_rate_diff'],
    'all3': ['x_rate_product', 'x_rate_ratio', 'x_rate_diff'],
}


def add_interactions(dfs):
    for name in dfs:
        df = dfs[name]
        r = df['decay_rate_2.5'].astype(float)
        tr = df['decay_tab_rate_3'].astype(float)
        df['x_rate_product'] = r * tr
        df['x_rate_ratio'] = (r + 0.01) / (tr + 0.01)
        df['x_rate_diff'] = r - tr
    return dfs


def train_eval(dfs, y, u, num_cols, seed=0, verbose=False, tag=''):
    cols = t63.CAT_COLS + num_cols
    Xtr, ytr, utr = t63._sort_by_user(dfs['train'][cols], y['train'], u['train'])
    Xva, yva, uva = t63._sort_by_user(dfs['valid'][cols], y['valid'], u['valid'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]
    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=2, learning_rate=0.10, n_estimators=500, min_child_samples=200,
        reg_lambda=1.0, random_state=seed, verbosity=-1, n_jobs=-1, linear_tree=True,
    )
    model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    va_scores = model.predict(dfs['valid'][cols])
    te_scores = model.predict(dfs['test'][cols])
    va = evaluate(u['valid'], y['valid'], va_scores)
    te = evaluate(u['test'], y['test'], te_scores)
    if verbose:
        print(f"[{tag}] seed={seed} valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)
    return model, va, te


if __name__ == '__main__':
    print("=== preparing rate_only base + interaction terms ===", flush=True)
    dfs, y, u = t63.prepare(DATA_DIR, 'rate_only')
    dfs = add_interactions(dfs)

    print("\n=== harness-fidelity check ===", flush=True)
    _, va0, te0 = train_eval(dfs, y, u, BASE_NUM_COLS, seed=0, verbose=True, tag='baseline (fidelity check)')
    print("  expect valid=0.67168 test=0.65353", flush=True)
    assert abs(va0['primary'] - 0.67168) < 1e-4 and abs(te0['primary'] - 0.65353) < 1e-4, "harness fidelity check FAILED"
    print("  PASS", flush=True)

    results = {'baseline': (va0['primary'], te0['primary'])}
    for tag, extra in VARIANTS.items():
        if tag == 'baseline':
            continue
        _, va, te = train_eval(dfs, y, u, BASE_NUM_COLS + extra, seed=0, verbose=True, tag=tag)
        results[tag] = (va['primary'], te['primary'])

    print("\n=== summary (seed 0) ===")
    print(f"{'variant':<10} {'valid':>9} {'test':>9} {'Δvalid':>9} {'Δtest':>9}")
    bva, bte = results['baseline']
    for tag in VARIANTS:
        v, tt = results[tag]
        print(f"{tag:<10} {v:>9.5f} {tt:>9.5f} {v-bva:>+9.5f} {tt-bte:>+9.5f}")

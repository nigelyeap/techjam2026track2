"""iter49: monotonic constraints on the engagement-rate features.

A structural lever, not a hyperparameter or feature-engineering change --
none of iter44-48 touched *how* the tree is allowed to split, only what
features/hyperparameters it sees. Domain knowledge: `decay_rate_2.5`
(recency-decayed positive-interaction rate), `decay_act_2.5` (recency-
decayed activity count), `decay_tab_3` (same, per-tab), and `lastk_rate`
(recent-window positive rate) should all have a non-negative relationship
with `long_view` probability -- more/recenter positive engagement should
never *decrease* predicted long-view likelihood. Forcing this via
LightGBM's `monotone_constraints` acts as a strong structural regularizer,
which may matter specifically at `num_leaves=2` (iter44's winning
capacity): with only one split per tree and a documented valid/test gap
that widens as capacity shrinks (iter44's caveat), a tree this shallow has
very little room to "average out" a spurious anti-correlated split found
by chance in a single low-cardinality bucket.

`duration_ms` and `gap` are left unconstrained (duration's relationship to
the label is definitionally two-sided via the `long_view` threshold
formula per iter44's own finding; gap/recency-since-last-activity has no
clear-cut monotonic direction -- could plausibly be U-shaped).

Built directly on iter44's exact pipeline/hyperparameters; only
`monotone_constraints` is added.
"""
import os, sys, importlib.util
import numpy as np
import pandas as pd
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402

_ITER44_DIR = os.path.join(_THIS_DIR, '..', 'iter44_gbm_native_features')


def _load_iter44_train():
    path = os.path.join(_ITER44_DIR, 'train.py')
    spec = importlib.util.spec_from_file_location('iter49_iter44_train', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gbm44 = _load_iter44_train()

# column order matches iter44's dict-insertion order: CAT_COLS then NUM_COLS
_ALL_COLS = _gbm44.CAT_COLS + _gbm44.NUM_COLS
# 0 = unconstrained, +1 = monotonically non-decreasing
_CONSTRAINT_MAP = {
    'duration_ms': 0, 'decay_rate_2.5': 1, 'decay_act_2.5': 1,
    'decay_tab_3': 1, 'lastk_rate': 1, 'gap': 0,
}
MONOTONE_CONSTRAINTS = [_CONSTRAINT_MAP.get(c, 0) for c in _ALL_COLS]


def run(data_dir, num_leaves=2, learning_rate=0.05, n_estimators=500,
        min_child_samples=200, reg_lambda=1.0, seed=0, verbose=False, _cache=None):
    dfs, y, u = _gbm44.prepare(data_dir) if _cache is None else _cache
    Xtr, ytr, utr = _gbm44._sort_by_user(dfs['train'], y['train'], u['train'])
    Xva, yva, uva = _gbm44._sort_by_user(dfs['valid'], y['valid'], u['valid'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]

    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=num_leaves, learning_rate=learning_rate,
        n_estimators=n_estimators, min_child_samples=min_child_samples,
        reg_lambda=reg_lambda, random_state=seed, verbosity=-1, n_jobs=-1,
        monotone_constraints=MONOTONE_CONSTRAINTS,
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
    print(f"columns: {_ALL_COLS}")
    print(f"monotone_constraints: {MONOTONE_CONSTRAINTS}")
    model, va, te, cache = run(DATA_DIR, verbose=True)
    print(f"\n[harness check] baseline iter44 (no constraints): valid=0.66135 test=0.64794")
    print(f"[iter49] with monotone_constraints: valid={va['primary']:.5f} test={te['primary']:.5f}")

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

"""iter50: GOSS boosting type at num_leaves=2.

iter46's boosting-type/sampling sweep covered gbdt (baseline), dart, and
row/column subsampling, but never GOSS (Gradient-based One-Side Sampling)
specifically -- a distinct boosting algorithm, not a sampling-rate
variant of gbdt. Cheap, single-axis test on iter44's exact pipeline and
hyperparameters, boosting_type swapped only.
"""
import os, sys
import numpy as np
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', 'iter44_gbm_native_features'))
from evaluate import evaluate  # noqa: E402
import train as gbm44  # noqa: E402


def run(data_dir, boosting_type='goss', num_leaves=2, learning_rate=0.05,
        n_estimators=500, min_child_samples=200, reg_lambda=1.0, seed=0,
        verbose=False, _cache=None):
    dfs, y, u = gbm44.prepare(data_dir) if _cache is None else _cache
    Xtr, ytr, utr = gbm44._sort_by_user(dfs['train'], y['train'], u['train'])
    Xva, yva, uva = gbm44._sort_by_user(dfs['valid'], y['valid'], u['valid'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]

    model = lgb.LGBMRanker(
        objective='lambdarank', metric='ndcg', eval_at=[5],
        num_leaves=num_leaves, learning_rate=learning_rate,
        n_estimators=n_estimators, min_child_samples=min_child_samples,
        reg_lambda=reg_lambda, random_state=seed, verbosity=-1, n_jobs=-1,
        boosting_type=boosting_type,
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
    _, va_g, te_g, cache = run(DATA_DIR, boosting_type='gbdt', verbose=True)
    print(f"[harness check] gbdt: valid={va_g['primary']:.5f} test={te_g['primary']:.5f} (expect 0.66135/0.64794)")
    _, va, te, _ = run(DATA_DIR, boosting_type='goss', _cache=cache, verbose=True)
    print(f"[iter50] goss: valid={va['primary']:.5f} test={te['primary']:.5f}")

"""iter44e: ablation -- does the sweep2 gain (num_leaves=2..7) survive
without `duration_ms`, the one feature iter44 added that was never in the
FM baseline's 6-feature set?

Empirical check first (outside this script): duration_ms is a per-video
constant (nunique==1 per video_id across the whole log) and correlates
with long_view at only 0.0073 -- a legitimate, non-leaky, pre-impression
item covariate, but too weakly correlated on its own to plausibly explain
a +0.02 valid jump. This ablation confirms that directly by dropping it
from NUM_COLS and re-running the two strongest sweep2 configs.
"""
import os, sys, copy
import numpy as np
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402
import train as gbm_train

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')


def run_no_duration(data, num_leaves, learning_rate=0.05, n_estimators=500,
                     min_child_samples=200, reg_lambda=1.0, seed=0):
    dfs, y, u = data
    dfs = {name: df.drop(columns=['duration_ms']) for name, df in dfs.items()}
    Xtr, ytr, utr = gbm_train._sort_by_user(dfs['train'], y['train'], u['train'])
    Xva, yva, uva = gbm_train._sort_by_user(dfs['valid'], y['valid'], u['valid'])
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
    return va, te, model.best_iteration_


def main():
    data = gbm_train.prepare(DATA_DIR)
    print("=== WITH duration_ms (sweep2 originals, for reference) ===")
    print("  num_leaves=2: valid=0.66135 test=0.64794")
    print("  num_leaves=7: valid=0.64632 test=0.64412")
    print()
    print("=== WITHOUT duration_ms ===")
    for nl in [2, 3, 5, 7, 15]:
        va, te, best_iter = run_no_duration(data, num_leaves=nl)
        print(f"num_leaves={nl}: valid={va['primary']:.5f} test={te['primary']:.5f} "
              f"best_iter={best_iter}  (full: {va})")


if __name__ == '__main__':
    main()

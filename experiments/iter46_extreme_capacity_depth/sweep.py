"""iter46: iter44 swept num_leaves down to LightGBM's hard floor (2) and
stopped there -- every other hyperparameter (learning_rate, n_estimators,
min_child_samples, reg_lambda, subsampling, boosting_type) was only ever
swept AROUND num_leaves=7 (sweep2.py's last 5 rows), never at the actual
winner num_leaves=2. This axis-by-axis sweep, fixed at num_leaves=2,
checks whether any of those axes still has headroom at the true best
capacity, plus tries DART boosting and row/column subsampling (neither
tried anywhere in the project so far).
"""
import os, sys, json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', 'iter44_gbm_native_features'))
import train as gbm_train

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
BASE = dict(num_leaves=2, learning_rate=0.05, n_estimators=500, min_child_samples=200, reg_lambda=1.0)

GRID = [
    dict(BASE),  # repeat of iter44's winner, sanity check
    # learning_rate axis
    {**BASE, 'learning_rate': 0.01, 'n_estimators': 2000},
    {**BASE, 'learning_rate': 0.02, 'n_estimators': 1500},
    {**BASE, 'learning_rate': 0.08},
    {**BASE, 'learning_rate': 0.1},
    # min_child_samples axis
    {**BASE, 'min_child_samples': 50},
    {**BASE, 'min_child_samples': 100},
    {**BASE, 'min_child_samples': 400},
    {**BASE, 'min_child_samples': 800},
    {**BASE, 'min_child_samples': 1600},
    # reg_lambda axis
    {**BASE, 'reg_lambda': 0.1},
    {**BASE, 'reg_lambda': 0.3},
    {**BASE, 'reg_lambda': 3.0},
    {**BASE, 'reg_lambda': 10.0},
]

# extra kwargs not in iter44's run() signature -- passed through via **cfg
EXTRA_GRID = [
    {**BASE, 'extra': dict(subsample=0.7, subsample_freq=1)},
    {**BASE, 'extra': dict(subsample=0.9, subsample_freq=1)},
    {**BASE, 'extra': dict(colsample_bytree=0.7)},
    {**BASE, 'extra': dict(colsample_bytree=0.9)},
    {**BASE, 'extra': dict(boosting_type='dart', n_estimators=300)},
]


def run_with_extra(cache, cfg):
    """Like gbm_train.run but forwards arbitrary extra LightGBM kwargs."""
    import lightgbm as lgb
    from evaluate import evaluate
    extra = cfg.pop('extra', {})
    for k in extra:
        cfg.pop(k, None)
    dfs, y, u = cache
    Xtr, ytr, utr = gbm_train._sort_by_user(dfs['train'], y['train'], u['train'])
    Xva, yva, uva = gbm_train._sort_by_user(dfs['valid'], y['valid'], u['valid'])
    gtr = np.unique(utr, return_counts=True)[1]
    gva = np.unique(uva, return_counts=True)[1]
    params = dict(objective='lambdarank', metric='ndcg', eval_at=[5],
                  random_state=0, verbosity=-1, n_jobs=-1, **cfg, **extra)
    model = lgb.LGBMRanker(**params)
    model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    va_scores, te_scores = model.predict(dfs['valid']), model.predict(dfs['test'])
    va, te = evaluate(u['valid'], y['valid'], va_scores), evaluate(u['test'], y['test'], te_scores)
    return model, va, te


def main():
    data = gbm_train.prepare(DATA_DIR)
    results = []
    for i, cfg in enumerate(GRID):
        cfg = dict(cfg)
        model, va, te, _ = gbm_train.run(DATA_DIR, seed=0, verbose=False, _cache=data, **cfg)
        print(f"[{i}] {cfg} -> valid={va['primary']:.5f} test={te['primary']:.5f} "
              f"best_iter={model.best_iteration_}", flush=True)
        results.append({'cfg': cfg, 'valid': float(va['primary']), 'test': float(te['primary'])})
    for i, cfg in enumerate(EXTRA_GRID):
        cfg = dict(cfg)
        model, va, te = run_with_extra(data, cfg)
        print(f"[extra {i}] {cfg} -> valid={va['primary']:.5f} test={te['primary']:.5f} "
              f"best_iter={model.best_iteration_}", flush=True)
        results.append({'cfg': cfg, 'valid': float(va['primary']), 'test': float(te['primary'])})
    results.sort(key=lambda r: -r['valid'])
    print("\n=== ranked by valid primary ===")
    for r in results:
        print(f"  valid={r['valid']:.5f} test={r['test']:.5f}  {r['cfg']}")
    with open(os.path.join(_THIS_DIR, 'sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote sweep_results.json")


if __name__ == '__main__':
    main()

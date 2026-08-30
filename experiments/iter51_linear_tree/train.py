"""iter51: LightGBM linear_tree=True at num_leaves=2.

At num_leaves=2, each tree makes exactly one split and predicts a
constant per leaf -- a piecewise-constant step function per tree.
`linear_tree=True` instead fits a (regularized) linear regression per
leaf, so with only 2 leaves the tree becomes a genuine piecewise-*linear*
function: still one split, but each side gets its own linear model over
the continuous features instead of a flat constant. This is a structural
change to what a "split" even buys the model -- distinct from every
hyperparameter (iter44/46), boosting-algorithm (iter50), feature (iter48),
and constraint (iter49) variant tried so far, and a natural fit for the
un-bucketed continuous features iter44's native encoding already exposes.

Single-run result was a real gain (valid 0.66932 vs iter44's 0.66135
baseline, harness-checked via linear_tree=False reproducing the baseline
exactly) -- well above both the 0.0003 look-threshold and the 0.001
confirmed-gain threshold on the very first run. This script reruns that
check plus the mandated 5-seed confirmation before treating it as real.
"""
import os, sys, importlib.util
import numpy as np
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402


def _load_module(path, name):
    # unique module name avoids sys.modules collisions with other
    # experiment directories' generically-named train.py when this file
    # is imported alongside them in the same process (e.g. from blend.py)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gbm44 = _load_module(os.path.join(_THIS_DIR, '..', 'iter44_gbm_native_features', 'train.py'), 'iter51_iter44_train')


def run(data_dir, linear_tree=True, num_leaves=2, learning_rate=0.05,
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
        print(f"best_iteration={model.best_iteration_}  valid={va}  test={te}")
    return model, va, te, (dfs, y, u)


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

    _, va_f, te_f, cache = run(DATA_DIR, linear_tree=False, verbose=True)
    print(f"[harness check] linear_tree=False: valid={va_f['primary']:.5f} test={te_f['primary']:.5f} "
          f"(expect 0.66135/0.64794)")

    _, va0, te0, _ = run(DATA_DIR, linear_tree=True, seed=0, _cache=cache, verbose=True)
    print(f"[iter51] linear_tree=True seed=0: valid={va0['primary']:.5f} test={te0['primary']:.5f}")

    gain = va0['primary'] - va_f['primary']
    print(f"\ngain over baseline: {gain:.5f}")
    if gain > 0.0003:
        print("=== clears 0.0003 look-threshold, running 4 more seeds ===")
        vas, tes = [va0['primary']], [te0['primary']]
        for s in (1, 2, 3, 4):
            _, va_s, te_s, _ = run(DATA_DIR, linear_tree=True, seed=s, _cache=cache)
            print(f"  seed={s} valid={va_s['primary']:.5f} test={te_s['primary']:.5f}")
            vas.append(va_s['primary']); tes.append(te_s['primary'])
        print(f"\n5-seed valid: mean={np.mean(vas):.5f} min={np.min(vas):.5f} max={np.max(vas):.5f}")
        print(f"5-seed test:  mean={np.mean(tes):.5f} min={np.min(tes):.5f} max={np.max(tes):.5f}")
        print(f"baseline (no linear_tree): valid=0.66135 test=0.64794")
    else:
        print("(gain does not clear the 0.0003 look-threshold -- no further seeds run)")

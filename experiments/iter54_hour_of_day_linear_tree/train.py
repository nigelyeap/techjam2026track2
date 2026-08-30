"""iter54: retest iter48's hour-of-day feature under linear_tree=True.

iter48 (hour_sin/hour_cos added to iter44's GBM-native feature set) was a
clean REJECT against the OLD constant-leaf GBM (valid 0.66054 vs baseline
0.66135). But that test predates iter51's structural finding that
linear_tree=True changes what each split buys the model -- a feature that
adds nothing to a piecewise-constant tree could still help a piecewise-
linear one, since the leaf's own linear model can now use hour-of-day as
a continuous regressor rather than only as a split candidate. Re-tests
the identical feature addition (same _row_to_dict, same CAT_COLS/NUM_COLS
as iter48) with linear_tree=True and iter51's winning hyperparameters
otherwise unchanged.
"""
import os, sys, importlib.util
import numpy as np
import lightgbm as lgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402

_iter48 = None


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter48 = _load_module(
    os.path.join(_THIS_DIR, '..', 'iter48_hour_of_day', 'train.py'), 'iter54_iter48_train')
prepare = _iter48.prepare
_sort_by_user = _iter48._sort_by_user


def run(data_dir, linear_tree=True, num_leaves=2, learning_rate=0.05,
        n_estimators=500, min_child_samples=200, reg_lambda=1.0, seed=0,
        verbose=False, _cache=None):
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

    _, va0, te0, cache = run(DATA_DIR, linear_tree=True, seed=0, verbose=True)
    print(f"[iter54] linear_tree=True + hour_sin/hour_cos: valid={va0['primary']:.5f} test={te0['primary']:.5f}")
    print(f"[compare] iter51 baseline (no hour feature): valid=0.66932 test=0.65146")

    gain = va0['primary'] - 0.66932
    print(f"\ngain over iter51 baseline: {gain:.5f}")
    if gain > 0.0003:
        print("=== clears 0.0003 look-threshold, running 4 more seeds ===")
        vas, tes = [va0['primary']], [te0['primary']]
        for s in (1, 2, 3, 4):
            _, va_s, te_s, _ = run(DATA_DIR, linear_tree=True, seed=s, _cache=cache)
            print(f"  seed={s} valid={va_s['primary']:.5f} test={te_s['primary']:.5f}", flush=True)
            vas.append(va_s['primary']); tes.append(te_s['primary'])
        print(f"\n5-seed valid: mean={np.mean(vas):.5f} min={np.min(vas):.5f} max={np.max(vas):.5f}")
        print(f"5-seed test:  mean={np.mean(tes):.5f} min={np.min(tes):.5f} max={np.max(tes):.5f}")
    else:
        print("(gain does not clear the 0.0003 look-threshold -- no further seeds run)")

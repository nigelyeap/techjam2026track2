"""iter78: learning_rate / reg_lambda / min_child_samples staleness resweep
on iter63's actual rate_only feature set.

These three hyperparameters were last tuned in iter55-58, BEFORE iter63's
decay_tab_rate_3-replaces-decay_tab_3 feature swap -- the exact same
staleness gap iter74 closed for num_leaves (iter74 found num_leaves=2 still
optimal post-swap). Mirrors iter74's rigor for the three remaining
untested-post-swap hyperparameters, reusing iter63's own prepare()/run()
unchanged (features cached once, shared across the whole sweep).

Order: learning_rate first (largest expected effect per iter55's own
finding), then reg_lambda, then min_child_samples, each staged on top of
the previous stage's winner (coordinate descent), matching iter76's
staged-sweep pattern.
"""
import os, sys, importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ITER63_DIR = os.path.join(_THIS_DIR, '..', 'iter63_decay_tab_rate')


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t63 = _load_module(os.path.join(_ITER63_DIR, 'train.py'), 'iter63_train')

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')


def main():
    print("=== preparing features (rate_only, cached once) ===", flush=True)
    cache = t63.prepare(DATA_DIR, 'rate_only')

    print("\n=== harness-fidelity check (defaults) ===", flush=True)
    _, va0, te0, _ = t63.run(DATA_DIR, 'rate_only', seed=0, verbose=True, _cache=cache)
    print("  expect valid=0.67168 test=0.65353", flush=True)
    assert abs(va0['primary'] - 0.67168) < 1e-4 and abs(te0['primary'] - 0.65353) < 1e-4, "harness fidelity check FAILED"
    print("  PASS", flush=True)

    results = {}

    print("\n=== Stage 1: learning_rate sweep ===", flush=True)
    lr_grid = [0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25]
    best_lr, best_lr_va = 0.10, va0['primary']
    for lr in lr_grid:
        _, va, te, _ = t63.run(DATA_DIR, 'rate_only', learning_rate=lr, seed=0, verbose=False, _cache=cache)
        tag = f"lr={lr}"
        results[tag] = (va['primary'], te['primary'])
        print(f"  [{tag}] valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)
        if va['primary'] > best_lr_va:
            best_lr, best_lr_va = lr, va['primary']
    print(f"  best learning_rate so far: {best_lr} (valid={best_lr_va:.5f})", flush=True)

    print("\n=== Stage 2: reg_lambda sweep (at best learning_rate) ===", flush=True)
    rl_grid = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    best_rl, best_rl_va = 1.0, best_lr_va
    for rl in rl_grid:
        if rl == 1.0 and best_lr == 0.10:
            results[f"lr={best_lr},reg_lambda={rl}"] = (best_lr_va, te0['primary'])
            continue
        _, va, te, _ = t63.run(DATA_DIR, 'rate_only', learning_rate=best_lr, reg_lambda=rl, seed=0, verbose=False, _cache=cache)
        tag = f"lr={best_lr},reg_lambda={rl}"
        results[tag] = (va['primary'], te['primary'])
        print(f"  [{tag}] valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)
        if va['primary'] > best_rl_va:
            best_rl, best_rl_va = rl, va['primary']
    print(f"  best reg_lambda so far: {best_rl} (valid={best_rl_va:.5f})", flush=True)

    print("\n=== Stage 3: min_child_samples sweep (at best lr, reg_lambda) ===", flush=True)
    mcs_grid = [50, 100, 200, 400, 800, 1600]
    best_mcs, best_mcs_va = 200, best_rl_va
    for mcs in mcs_grid:
        if mcs == 200:
            results[f"lr={best_lr},reg_lambda={best_rl},mcs={mcs}"] = (best_rl_va, None)
            continue
        _, va, te, _ = t63.run(DATA_DIR, 'rate_only', learning_rate=best_lr, reg_lambda=best_rl,
                                min_child_samples=mcs, seed=0, verbose=False, _cache=cache)
        tag = f"lr={best_lr},reg_lambda={best_rl},mcs={mcs}"
        results[tag] = (va['primary'], te['primary'])
        print(f"  [{tag}] valid={va['primary']:.5f} test={te['primary']:.5f}", flush=True)
        if va['primary'] > best_mcs_va:
            best_mcs, best_mcs_va = mcs, va['primary']
    print(f"  best min_child_samples so far: {best_mcs} (valid={best_mcs_va:.5f})", flush=True)

    print("\n=== final best config ===", flush=True)
    print(f"  learning_rate={best_lr}, reg_lambda={best_rl}, min_child_samples={best_mcs}", flush=True)
    _, va_final, te_final, _ = t63.run(DATA_DIR, 'rate_only', learning_rate=best_lr, reg_lambda=best_rl,
                                        min_child_samples=best_mcs, seed=0, verbose=True, _cache=cache)
    print(f"\n  baseline (iter63 defaults): valid={va0['primary']:.5f} test={te0['primary']:.5f}")
    print(f"  best found config:          valid={va_final['primary']:.5f} test={te_final['primary']:.5f}")
    print(f"  delta:                      valid={va_final['primary']-va0['primary']:+.5f} test={te_final['primary']-te0['primary']:+.5f}")


if __name__ == '__main__':
    main()

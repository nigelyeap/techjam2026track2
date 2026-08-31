"""iter74: num_leaves resweep on iter63's CURRENT final feature set
(rate_only: decay_tab_rate_3 replacing decay_tab_3), which did not exist
when num_leaves was last swept (iter44's original floor-sweep, iter52's
linear_tree resweep -- both pre-iter63). Structurally different lever from
Round 22's decayed-rate-generalization family (iter68-73, now closed 4/4
null): no new causal feature, just re-checking whether the capacity choice
that was tuned on an earlier, weaker feature set is still optimal now that
a genuinely new proven feature (decay_tab_rate_3) has been added.

Reuses iter63's own run() unchanged (accepts num_leaves as a parameter
already) -- only num_leaves varies, everything else (learning_rate=0.10,
n_estimators=500, min_child_samples=200, reg_lambda=1.0, linear_tree=True,
rate_only feature set) held fixed at iter63's exact winning config.
"""
import os, sys, importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ITER63_DIR = os.path.join(_THIS_DIR, '..', 'iter63_decay_tab_rate')


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_t63 = _load_module(os.path.join(_ITER63_DIR, 'train.py'), 'iter74_t63')

NUM_LEAVES_GRID = [2, 3, 4, 5, 6, 7, 8, 10]

if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

    print("=== preparing features (rate_only, shared across all num_leaves) ===", flush=True)
    cache = _t63.prepare(DATA_DIR, 'rate_only')

    print("\n=== harness-fidelity check (num_leaves=2) ===", flush=True)
    _, va0, te0, _ = _t63.run(DATA_DIR, 'rate_only', num_leaves=2, seed=0, verbose=True, _cache=cache)
    print("  expect valid=0.67168 test=0.65353", flush=True)
    assert abs(va0['primary'] - 0.67168) < 1e-4 and abs(te0['primary'] - 0.65353) < 1e-4, "harness fidelity check FAILED"
    print("  PASS", flush=True)

    results = {2: (va0['primary'], te0['primary'])}
    for nl in NUM_LEAVES_GRID:
        if nl == 2:
            continue
        print(f"\n=== num_leaves={nl} ===", flush=True)
        _, va, te, _ = _t63.run(DATA_DIR, 'rate_only', num_leaves=nl, seed=0, verbose=True, _cache=cache)
        results[nl] = (va['primary'], te['primary'])

    print("\n=== summary (seed 0, rate_only feature set) ===")
    print(f"{'num_leaves':<12} {'valid':>9} {'test':>9} {'Δvalid':>9} {'Δtest':>9}")
    bva, bte = results[2]
    for nl in NUM_LEAVES_GRID:
        v, tt = results[nl]
        print(f"{nl:<12} {v:>9.5f} {tt:>9.5f} {v-bva:>+9.5f} {tt-bte:>+9.5f}")

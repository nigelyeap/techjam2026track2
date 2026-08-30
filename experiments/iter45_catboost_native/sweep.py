"""iter45 sweep: iter44 found LightGBM's optimum at the shallowest possible
tree (num_leaves=2). Test whether the same shrink-capacity pattern holds
for CatBoost on the identical native encoding -- default depth=6 badly
underperforms (valid 0.62127 vs FM's 0.63988), so sweep depth down to
CatBoost's floor (1), plus loss_function and learning_rate/l2 variants.
"""
import os, sys, json
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import train as cb_train

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
BASE = dict(depth=6, iterations=500, learning_rate=0.05, l2_leaf_reg=3.0, loss_function='YetiRank')

GRID = [
    dict(BASE),  # repeat, sanity check
    {**BASE, 'depth': 1},
    {**BASE, 'depth': 2},
    {**BASE, 'depth': 3},
    {**BASE, 'depth': 4},
    {**BASE, 'depth': 5},
    {**BASE, 'depth': 2, 'loss_function': 'PairLogitPairwise'},
    {**BASE, 'depth': 2, 'loss_function': 'PairLogit'},
    {**BASE, 'depth': 2, 'l2_leaf_reg': 1.0},
    {**BASE, 'depth': 2, 'l2_leaf_reg': 10.0},
    {**BASE, 'depth': 2, 'learning_rate': 0.02, 'iterations': 1500},
    {**BASE, 'depth': 2, 'learning_rate': 0.1},
]


def main():
    data = cb_train._t44.prepare(DATA_DIR)
    results = []
    for i, cfg in enumerate(GRID):
        model, va, te, _ = cb_train.run(DATA_DIR, seed=0, verbose=False, _cache=data, **cfg)
        print(f"[{i}] {cfg} -> valid={va['primary']:.5f} test={te['primary']:.5f} "
              f"best_iter={model.get_best_iteration()}", flush=True)
        results.append({'cfg': cfg, 'valid': float(va['primary']), 'test': float(te['primary'])})
    results.sort(key=lambda r: -r['valid'])
    print("\n=== ranked by valid primary ===")
    for r in results:
        print(f"  valid={r['valid']:.5f} test={r['test']:.5f}  {r['cfg']}")
    with open(os.path.join(_THIS_DIR, 'sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nwrote sweep_results.json")


if __name__ == '__main__':
    main()

import sys, os, json, time
sys.path.insert(0, '../..')
from data import load
from train import run_hybrid

splits = load('../../KuaiRand-Pure/data')

BW = 0.5
results = {}
for seed in (0, 1, 2, 3, 4):
    t0 = time.time()
    res = run_hybrid(splits, epochs=40, patience=4, seed=seed, bpr_weight=BW, verbose=False)
    dt = time.time() - t0
    key = f"seed{seed}"
    results[key] = {'valid': {k: float(v) for k, v in res['valid'].items()},
                     'test': {k: float(v) for k, v in res['test'].items()}}
    print(f"{key}: valid primary {res['valid']['primary']:.4f} test primary {res['test']['primary']:.4f} ({dt:.1f}s)", flush=True)

with open('final_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("done")

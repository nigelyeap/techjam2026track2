"""iter12 sweep driver: combines new item-side causal features (video_pop,
author_rate) with iter9's winning user-side set (activity, tab, rate).
Loads the extended dataset ONCE (the causal-feature pass is the expensive
part), then runs each (combo, seed) as an ordinary foreground call, writing
incremental results to results.json after every run so partial progress
survives interruption.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext
from train import run_bpr_ext

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'KuaiRand-Pure', 'data')
OUT = os.path.join(os.path.dirname(__file__), 'results.json')

COMBOS = {
    'user_only_iter9':        ('activity', 'tab', 'rate'),           # reference, not re-run if cached
    'user+video_pop':         ('activity', 'tab', 'rate', 'video_pop'),
    'user+author_rate':       ('activity', 'tab', 'rate', 'author_rate'),
    'user+video_pop+author_rate': ('activity', 'tab', 'rate', 'video_pop', 'author_rate'),
    'item_only':               ('video_pop', 'author_rate'),
}

SEEDS = [0, 1, 2]


def _clean(d):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}


def load_results():
    if os.path.exists(OUT):
        with open(OUT) as fh:
            return json.load(fh)
    return {}


def save_results(res):
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)


def main():
    print(f"loading extended dataset (single causal-feature pass) from {DATA_DIR} ...")
    t0 = time.time()
    splits = load_ext(DATA_DIR)
    print(f"  loaded in {time.time()-t0:.1f}s")

    results = load_results()

    for combo_name, feature_set in COMBOS.items():
        results.setdefault(combo_name, {})
        for seed in SEEDS:
            key = str(seed)
            if key in results[combo_name]:
                print(f"[skip cached] {combo_name} seed={seed}")
                continue
            print(f"\n=== {combo_name} features={feature_set} seed={seed} ===")
            t0 = time.time()
            res = run_bpr_ext(DATA_DIR, feature_set=feature_set, seed=seed, verbose=False,
                               preloaded=splits)
            dt = time.time() - t0
            entry = {'valid': _clean(res['valid']), 'test': _clean(res['test']), 'time_s': dt}
            results[combo_name][key] = entry
            save_results(results)
            print(f"  valid primary={entry['valid']['primary']:.5f}  "
                  f"test primary={entry['test']['primary']:.5f}  ({dt:.1f}s)")

    print("\n\n=== SUMMARY (mean over available seeds) ===")
    for combo_name, feature_set in COMBOS.items():
        seeds_done = results.get(combo_name, {})
        if not seeds_done:
            continue
        vmeans = [v['valid']['primary'] for v in seeds_done.values()]
        tmeans = [v['test']['primary'] for v in seeds_done.values()]
        print(f"{combo_name:30s} n={len(vmeans)}  valid mean={np.mean(vmeans):.5f} "
              f"std={np.std(vmeans):.5f}  test mean={np.mean(tmeans):.5f} std={np.std(tmeans):.5f}")


if __name__ == '__main__':
    main()

"""Sweep driver for iter24. Loads the (cached) extended dataset ONCE, then
loops over configs x seeds, writing incremental results to results.json after
every run so partial progress survives interruption.

Step 1: re-sweep decay_rate_H+decay_act_H halflife (fine grid, {2,2.5,3,3.5}
days), WITH iter19's momentum fields (last1,lastk_rate,gap) and flat `tab`
present throughout -- checking whether 2.5d (iter20's finding, momentum-free)
is still optimal once momentum is in the mix. 3 seeds each.

Step 2: with the winning H* from step 1, sweep decay_tab_H2 (iter20's grid,
{3,7} days) in place of flat `tab`, on top of decay_rate_H*/decay_act_H* +
momentum. 3 seeds each. Compare against the H* config's flat-tab baseline
(already have it from step 1).

Step 3 (conditional): whichever config (best of step1 flat-tab winner vs
best of step2 decay-tab variants) beats iter19's 5-seed valid mean (0.62898)
by a real margin (>0.001-0.002) gets extended to a full 5-seed run
(seeds 0-4).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES, TAB_HALFLIVES
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

ITER19_BEST_VALID_5SEED = 0.62898  # reference: decay_rate_3+decay_act_3+tab+last1+lastk_rate+gap, 5-seed


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as fh:
            return json.load(fh)
    return []


def save_results(res):
    with open(RESULTS_PATH, 'w') as fh:
        json.dump(res, fh, indent=2)


def already_done(res, tag, seed):
    return any(r['tag'] == tag and r['seed'] == seed for r in res)


def run_config(tag, feature_set, seeds, results, epochs=40):
    for seed in seeds:
        if already_done(results, tag, seed):
            print(f"[skip] {tag} seed={seed} already done")
            continue
        t0 = time.time()
        out = run_bpr_ext(DATA_DIR, feature_set=feature_set, halflives=HALFLIVES,
                           tab_halflives=TAB_HALFLIVES, seed=seed, verbose=False, epochs=epochs,
                           splits_cache=SPLITS_CACHE)
        dt = time.time() - t0
        rec = {'tag': tag, 'features': list(feature_set), 'seed': seed,
               'valid_primary': float(out['valid']['primary']), 'valid_gauc': float(out['valid']['GAUC']),
               'valid_ndcg5': float(out['valid']['nDCG@5']),
               'test_primary': float(out['test']['primary']), 'test_gauc': float(out['test']['GAUC']),
               'test_ndcg5': float(out['test']['nDCG@5']), 'seconds': dt}
        results.append(rec)
        save_results(results)
        print(f"[done] {tag:45s} seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)", flush=True)


def mean_valid(results, tag):
    vals = [r['valid_primary'] for r in results if r['tag'] == tag]
    return sum(vals) / len(vals) if vals else None


if __name__ == '__main__':
    print("loading extended dataset (cached after first run)...")
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s")

    results = load_results()
    seeds3 = [0, 1, 2]

    # ---- Step 1: fine halflife grid for decay_rate_H+decay_act_H+tab+momentum, 3 seeds ----
    print("\n=== Step 1: fine halflife sweep (decay_rate_H+decay_act_H+tab+last1+lastk_rate+gap) ===")
    for h in HALFLIVES:
        tag = f'decay_rate_{h}+decay_act_{h}+tab+mom'
        feats = (f'decay_rate_{h}', f'decay_act_{h}', 'tab', 'last1', 'lastk_rate', 'gap')
        run_config(tag, feats, seeds3, results)

    step1_means = {h: mean_valid(results, f'decay_rate_{h}+decay_act_{h}+tab+mom') for h in HALFLIVES}
    print("\nStep 1 valid means:", {h: round(v, 5) for h, v in step1_means.items()})
    best_h = max(step1_means, key=lambda h: step1_means[h])
    print(f"Step 1 best halflife: {best_h} (valid mean {step1_means[best_h]:.5f})")

    # ---- Step 2: decayed tab_pos at best_h, vs flat-tab baseline (already have it from Step 1) ----
    print(f"\n=== Step 2: decayed tab_pos (base halflife={best_h}) + momentum ===")
    for th in TAB_HALFLIVES:
        tag = f'decay_rate_{best_h}+decay_act_{best_h}+decay_tab_{th}+mom'
        feats = (f'decay_rate_{best_h}', f'decay_act_{best_h}', f'decay_tab_{th}',
                  'last1', 'lastk_rate', 'gap')
        run_config(tag, feats, seeds3, results)

    step2_means = {th: mean_valid(results, f'decay_rate_{best_h}+decay_act_{best_h}+decay_tab_{th}+mom')
                   for th in TAB_HALFLIVES}
    flat_tab_mean = step1_means[best_h]
    print("Step 2 valid means (decayed tab):", {th: round(v, 5) for th, v in step2_means.items()})
    print(f"Flat-tab baseline (best_h={best_h}) valid mean: {flat_tab_mean:.5f}")

    best_tab_th = max(step2_means, key=lambda th: step2_means[th])
    candidates = [
        (f'decay_rate_{best_h}+decay_act_{best_h}+tab+mom',
         (f'decay_rate_{best_h}', f'decay_act_{best_h}', 'tab', 'last1', 'lastk_rate', 'gap'),
         flat_tab_mean),
        (f'decay_rate_{best_h}+decay_act_{best_h}+decay_tab_{best_tab_th}+mom',
         (f'decay_rate_{best_h}', f'decay_act_{best_h}', f'decay_tab_{best_tab_th}',
          'last1', 'lastk_rate', 'gap'),
         step2_means[best_tab_th]),
    ]
    best_overall_tag, best_overall_mean, best_overall_feats = None, -1, None
    for tag, feats, mean in candidates:
        if mean > best_overall_mean:
            best_overall_mean, best_overall_tag, best_overall_feats = mean, tag, feats

    print(f"\nBest overall 3-seed config: {best_overall_tag} (valid mean {best_overall_mean:.5f}) "
          f"vs iter19 5-seed reference {ITER19_BEST_VALID_5SEED:.5f}")

    # ---- Step 3 (conditional): 5-seed confirmation if a real margin over iter19 ----
    margin = best_overall_mean - ITER19_BEST_VALID_5SEED
    print(f"Margin vs iter19 5-seed valid mean: {margin:+.5f}")
    if margin > 0.001:
        print(f"\n=== Step 3: 5-seed confirmation of {best_overall_tag} ===")
        seeds5 = [0, 1, 2, 3, 4]
        run_config(best_overall_tag, best_overall_feats, seeds5, results)
        conf_mean = mean_valid(results, best_overall_tag)
        print(f"5-seed valid mean for {best_overall_tag}: {conf_mean:.5f}")
    else:
        print("\nMargin below confirmation threshold (~0.001-0.002) -- no 5-seed run triggered by driver.")

    print("\nSweep complete.")

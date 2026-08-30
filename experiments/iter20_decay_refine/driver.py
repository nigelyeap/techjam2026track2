"""Sweep driver for iter20. Loads the (cached) extended dataset ONCE, then
loops over configs x seeds, writing incremental results to results.json after
every run so partial progress survives interruption. Foreground script, no
backgrounding needed for the sweep phases themselves — each run is ~15-90s.

Axis A: fine halflife grid (HALFLIVES) for decay_rate_H+decay_act_H+tab (tab
flat, iter16-style) — 3 seeds each.
Axis B: with the best halflife H* found in Axis A, decayed-tab variants
decay_rate_H*+decay_act_H*+decay_tab_h for h in TAB_HALFLIVES, vs the flat-tab
baseline (=the H* run from Axis A) — 3 seeds each.
Phase 3 (conditional): 5-seed confirmation (seeds 3,4 added) of whichever
config(s) beat iter16's 5-seed valid mean (0.62030) by a real margin.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES, TAB_HALFLIVES
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

ITER16_BEST_VALID_5SEED = 0.62030  # reference: decay_rate_3+decay_act_3+tab, 5-seed


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
        print(f"[done] {tag:35s} seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)")


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

    # ---- Axis A: fine halflife grid, decay_rate_H+decay_act_H+tab, 3 seeds ----
    print("\n=== Axis A: fine halflife sweep (decay_rate_H+decay_act_H+tab) ===")
    for h in HALFLIVES:
        tag = f'decay_rate_{h}+decay_act_{h}+tab'
        run_config(tag, (f'decay_rate_{h}', f'decay_act_{h}', 'tab'), seeds3, results)

    axisA_means = {h: mean_valid(results, f'decay_rate_{h}+decay_act_{h}+tab') for h in HALFLIVES}
    print("\nAxis A valid means:", {h: round(v, 5) for h, v in axisA_means.items()})
    best_h = max(axisA_means, key=lambda h: axisA_means[h])
    print(f"Axis A best halflife: {best_h} (valid mean {axisA_means[best_h]:.5f})")

    # ---- Axis B: decayed tab_pos at best_h, vs flat-tab baseline (already have it from Axis A) ----
    print(f"\n=== Axis B: decayed tab_pos (base halflife={best_h}) ===")
    for th in TAB_HALFLIVES:
        tag = f'decay_rate_{best_h}+decay_act_{best_h}+decay_tab_{th}'
        run_config(tag, (f'decay_rate_{best_h}', f'decay_act_{best_h}', f'decay_tab_{th}'), seeds3, results)

    axisB_means = {th: mean_valid(results, f'decay_rate_{best_h}+decay_act_{best_h}+decay_tab_{th}')
                   for th in TAB_HALFLIVES}
    flat_tab_mean = axisA_means[best_h]
    print("Axis B valid means (decayed tab):", {th: round(v, 5) for th, v in axisB_means.items()})
    print(f"Flat-tab baseline (best_h={best_h}) valid mean: {flat_tab_mean:.5f}")

    best_tab_th = max(axisB_means, key=lambda th: axisB_means[th])
    best_overall_tag, best_overall_mean, best_overall_feats = None, -1, None
    # candidates: best flat-tab config from Axis A, and best decayed-tab config from Axis B
    candidates = [
        (f'decay_rate_{best_h}+decay_act_{best_h}+tab',
         (f'decay_rate_{best_h}', f'decay_act_{best_h}', 'tab'), flat_tab_mean),
        (f'decay_rate_{best_h}+decay_act_{best_h}+decay_tab_{best_tab_th}',
         (f'decay_rate_{best_h}', f'decay_act_{best_h}', f'decay_tab_{best_tab_th}'), axisB_means[best_tab_th]),
    ]
    for tag, feats, mean in candidates:
        if mean > best_overall_mean:
            best_overall_mean, best_overall_tag, best_overall_feats = mean, tag, feats

    print(f"\nBest overall 3-seed config: {best_overall_tag} (valid mean {best_overall_mean:.5f}) "
          f"vs iter16 5-seed reference {ITER16_BEST_VALID_5SEED:.5f}")

    # ---- Phase 3 (conditional): 5-seed confirmation if a real margin over iter16 ----
    margin = best_overall_mean - ITER16_BEST_VALID_5SEED
    print(f"Margin vs iter16 5-seed valid mean: {margin:+.5f}")
    if margin > 0.001:
        print(f"\n=== Phase 3: 5-seed confirmation of {best_overall_tag} ===")
        seeds5 = [0, 1, 2, 3, 4]
        run_config(best_overall_tag, best_overall_feats, seeds5, results)
        conf_mean = mean_valid(results, best_overall_tag)
        print(f"5-seed valid mean for {best_overall_tag}: {conf_mean:.5f}")
    else:
        print("\nMargin below confirmation threshold (~0.001-0.002) — no 5-seed run triggered by driver.")

    print("\nSweep complete.")

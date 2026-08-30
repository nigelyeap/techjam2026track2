"""Sweep driver for iter32. Loads the (cached) extended dataset ONCE, then
loops over configs x seeds, writing incremental results to results.json after
EVERY single run so partial progress survives interruption (this session's
platform has killed multiple agents mid-run without warning in prior rounds).

Step 0 (harness-fidelity check): run iter24's exact winning feature set
(decay_rate_2.5, decay_act_2.5, decay_tab_3, last1, lastk_rate, gap) --
i.e. attention features excluded entirely -- through THIS iteration's
harness, 3 seeds (0,1,2). Must closely reproduce iter24's own published
per-seed numbers (seed0 valid 0.63260/test 0.62839, seed1 valid
0.63308/test 0.62942, seed2 valid 0.63179/test 0.62687) to prove this
harness is not a reimplementation with subtle drift.

Step 1: sweep the new target-attention feature, ONE extra field added on
top of iter24's base 6-field set, across:
  - attn_rate_W for W in {10, 20, 40}  (pure dot-product attention)
  - attn_decay_rate_H for H in {3.0, 7.0}  (recency-decay fallback variant)
5 configs x 3 seeds each. Selection on VALID ONLY.

Step 2 (conditional): if the best Step 1 config beats iter24's own 5-seed
valid reference (0.63251) by more than ~0.001, extend that single config to
a full 5-seed run (seeds 0-4) for a real promote/reject decision. Otherwise
report the 3-seed non-promotion honestly -- no 5-seed run is spent chasing
noise.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES, TAB_HALFLIVES, WINDOWS, ATTN_DECAY_HALFLIVES, K_EMB, EMB_EPOCHS
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

BASE_FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')

ITER24_5SEED_VALID = {0: 0.63260, 1: 0.63308, 2: 0.63179, 3: 0.63208, 4: 0.63298}
ITER24_5SEED_TEST = {0: 0.62839, 1: 0.62942, 2: 0.62687, 3: 0.62855, 4: 0.62892}
ITER24_BEST_VALID_5SEED = 0.63251  # mean, current standing best (5-seed)


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
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)", flush=True)


def mean_valid(results, tag):
    vals = [r['valid_primary'] for r in results if r['tag'] == tag]
    return sum(vals) / len(vals) if vals else None


def std(vals):
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


if __name__ == '__main__':
    print("loading extended dataset (cached after first run)...")
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                             windows=WINDOWS, decay_halflives=ATTN_DECAY_HALFLIVES,
                             k_emb=K_EMB, emb_epochs=EMB_EPOCHS)
    print(f"loaded in {time.time()-t0:.1f}s")

    results = load_results()
    seeds3 = [0, 1, 2]

    # ---- Step 0: harness-fidelity check (iter24's exact feature set, no attention) ----
    print("\n=== Step 0: harness-fidelity check (iter24 feature set, attention excluded) ===")
    hf_tag = 'harness_fidelity_iter24_base'
    run_config(hf_tag, BASE_FEATURES, seeds3, results)
    print("\nHarness-fidelity comparison vs iter24's own published per-seed numbers:")
    max_abs_diff = 0.0
    for seed in seeds3:
        got = [r for r in results if r['tag'] == hf_tag and r['seed'] == seed]
        if not got:
            continue
        gv, gt = got[0]['valid_primary'], got[0]['test_primary']
        rv, rt = ITER24_5SEED_VALID[seed], ITER24_5SEED_TEST[seed]
        dv, dt = gv - rv, gt - rt
        max_abs_diff = max(max_abs_diff, abs(dv), abs(dt))
        print(f"  seed={seed}  valid got={gv:.5f} ref={rv:.5f} Δ={dv:+.5f}  |  "
              f"test got={gt:.5f} ref={rt:.5f} Δ={dt:+.5f}")
    print(f"  max abs diff vs iter24 reference: {max_abs_diff:.5f}")
    if max_abs_diff > 0.003:
        print("  WARNING: harness fidelity diff exceeds 0.003 -- investigate before trusting sweep results.")
    else:
        print("  Harness fidelity confirmed (diff within noise).")

    # ---- Step 1: attention feature sweep, one extra field on top of BASE_FEATURES ----
    print("\n=== Step 1: target-attention sweep (BASE_FEATURES + 1 attention field), 3 seeds ===")
    step1_tags = []
    for w in WINDOWS:
        tag = f'base+attn_rate_{w}'
        feats = BASE_FEATURES + (f'attn_rate_{w}',)
        run_config(tag, feats, seeds3, results)
        step1_tags.append((tag, feats))
    for h in ATTN_DECAY_HALFLIVES:
        tag = f'base+attn_decay_rate_{h}'
        feats = BASE_FEATURES + (f'attn_decay_rate_{h}',)
        run_config(tag, feats, seeds3, results)
        step1_tags.append((tag, feats))

    print("\nStep 1 valid means:")
    step1_means = {}
    for tag, feats in step1_tags:
        m = mean_valid(results, tag)
        step1_means[tag] = m
        print(f"  {tag:30s} valid mean={m:.5f}")

    base_mean = mean_valid(results, hf_tag)
    print(f"\nBaseline (no attention, this harness's own 3-seed mean): {base_mean:.5f}")

    best_tag = max(step1_means, key=lambda t: step1_means[t])
    best_mean = step1_means[best_tag]
    best_feats = dict(step1_tags)[best_tag]
    print(f"\nBest Step 1 config: {best_tag} (valid mean {best_mean:.5f}), "
          f"vs iter24 5-seed reference {ITER24_BEST_VALID_5SEED:.5f}")

    # ---- Step 2 (conditional): 5-seed confirmation ----
    margin = best_mean - ITER24_BEST_VALID_5SEED
    print(f"Margin vs iter24 5-seed valid mean: {margin:+.5f}")
    if margin > 0.001:
        print(f"\n=== Step 2: 5-seed confirmation of {best_tag} ===")
        seeds5 = [0, 1, 2, 3, 4]
        run_config(best_tag, best_feats, seeds5, results)
        conf_vals = [r['valid_primary'] for r in results if r['tag'] == best_tag]
        conf_mean = sum(conf_vals) / len(conf_vals)
        print(f"5-seed valid mean for {best_tag}: {conf_mean:.5f}")
    else:
        print("\nMargin below confirmation threshold (~0.001) -- no 5-seed run triggered. "
              "Reporting honest 3-seed non-promotion.")

    print("\nSweep complete.")

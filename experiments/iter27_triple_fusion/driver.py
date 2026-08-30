"""Sweep driver for iter27 (triple fusion of iter24's refined features +
iter23's decay-aware BPR sampling weight + iter25's Laplace-alpha/n_buckets).

RERUN NOTE (Round 9): this is a rerun of iter27 after the original agent's
session was killed by a platform session-limit error mid-run, before any
results.json was written (total loss, source code survived). This driver
was revised on rerun to:
  (a) tighten test-reporting discipline per this round's new rule: valid-only
      selection, test_primary NOT printed in sweep console output/tables,
      computed+reported only ONCE at the very end for the single final
      winning config (Step 0's fidelity check is exempt -- it's comparing
      against iter24's ALREADY-PUBLISHED numbers, not new test-peeking).
  (b) fix the 5-seed-extension threshold to match protocol exactly: extend
      to 5 seeds ONLY if the best 3-seed valid config beats iter24's 5-seed
      valid reference (0.63251) by >0.001; otherwise stop at 3 seeds and
      report a non-promotion honestly. (The original driver extended to 5
      seeds unconditionally "for an honest read" -- superseded by the
      explicit new instruction not to bother with 5 seeds below the bar.)
  (c) add an n_buckets=10 variant to the Step 1 sweep, addressing iter29's
      finding (discovered AFTER iter27 was originally written) that
      n_buckets=20's gain does not robustly replicate on a date-shifted
      split -- worth checking whether n_buckets=20 remains the valid-winner
      when combined with the other two ingredients, on the official split.

Step 0: harness-fidelity check -- run this dir's code at flat sampling
(sampling_mode='flat', sampling_alpha=1.0) with iter24's EXACT feature set
and iter19/iter24 default constants (alpha=1.0 Laplace, n_buckets=10),
seeds 0/1/2, and diff against iter24's own published RESULT.md numbers.
Must match (near-)exactly before anything else is trusted. (Legitimately
uses test here -- comparing against already-known/published values, not a
new peek.)

Step 1: 3-seed sweep over the free-parameter combinations worth checking on
top of the fused feature set (iter23's decay_halflife=3 kept fixed
throughout -- not re-swept, not this round's focus):
  - triple_fusion_default:    sampling_alpha=0.5, alpha=0.5, n_buckets=20
  - triple_fusion_nbuckets10: sampling_alpha=0.5, alpha=0.5, n_buckets=10
  - fusion_sampling_alpha0.25: sampling_alpha=0.25, alpha=0.5, n_buckets=20
  - fusion_sampling_alpha0.75: sampling_alpha=0.75, alpha=0.5, n_buckets=20
Only VALID means are printed/compared during this step.

Step 2 (conditional): if the single best-by-valid config from Step 1 beats
iter24's 5-seed valid reference (0.63251) by >0.001, extend ONLY that config
to a full 5-seed run (seeds 0-4, i.e. add seeds 3/4). Otherwise stop here --
report the 3-seed result as a non-promotion, no 5-seed run.

Step 3 (only if Step 2 ran): compute+print test_primary for the winning
config ONCE, for the record -- this is the single sanctioned test read.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES, TAB_HALFLIVES, ALPHA
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

ITER24_FEATS = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
ITER24_VALID_5SEED = 0.63251  # reference: iter24's own 5-seed mean
ITER24_TEST_5SEED = 0.62843
ITER23_VALID_5SEED = 0.63109
ITER23_TEST_5SEED = 0.62929
ITER25_VALID_5SEED = 0.63028
ITER25_TEST_5SEED = 0.63185

# iter24's own published seed 0/1/2 (from RESULT.md 5-seed table)
ITER24_PUBLISHED = {0: (0.63260, 0.62839), 1: (0.63308, 0.62942), 2: (0.63179, 0.62687)}

PROMOTION_MARGIN = 0.001  # per protocol: >~0.001 over iter24's 5-seed valid ref triggers 5-seed extension


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


def run_config(tag, feature_set, seeds, results, epochs=40,
               sampling_mode='flat', sampling_alpha=1.0, decay_halflife=3,
               alpha=ALPHA, n_buckets=10, report_test=False):
    """report_test: if False (default, used for all sweep steps per this
    round's discipline), console output shows ONLY valid_primary. test is
    still computed (evaluate() is cheap and run_bpr_ext always returns it)
    and still stored in results.json for the permanent record / eventual
    single final report, but it is NOT printed/surfaced during the sweep."""
    for seed in seeds:
        if already_done(results, tag, seed):
            print(f"[skip] {tag} seed={seed} already done")
            continue
        t0 = time.time()
        out = run_bpr_ext(DATA_DIR, feature_set=feature_set, halflives=HALFLIVES,
                           tab_halflives=TAB_HALFLIVES, seed=seed, verbose=False, epochs=epochs,
                           splits_cache=SPLITS_CACHE,
                           sampling_mode=sampling_mode, sampling_alpha=sampling_alpha,
                           decay_halflife=decay_halflife, alpha=alpha, n_buckets=n_buckets)
        dt = time.time() - t0
        rec = {'tag': tag, 'features': list(feature_set), 'seed': seed,
               'sampling_mode': sampling_mode, 'sampling_alpha': sampling_alpha,
               'decay_halflife': decay_halflife, 'alpha': alpha, 'n_buckets': n_buckets,
               'valid_primary': float(out['valid']['primary']), 'valid_gauc': float(out['valid']['GAUC']),
               'valid_ndcg5': float(out['valid']['nDCG@5']),
               'test_primary': float(out['test']['primary']), 'test_gauc': float(out['test']['GAUC']),
               'test_ndcg5': float(out['test']['nDCG@5']), 'seconds': dt}
        results.append(rec)
        save_results(results)
        if report_test:
            print(f"[done] {tag:32s} seed={seed} valid={out['valid']['primary']:.5f} "
                  f"test={out['test']['primary']:.5f}  ({dt:.1f}s)", flush=True)
        else:
            print(f"[done] {tag:32s} seed={seed} valid={out['valid']['primary']:.5f}  "
                  f"(test computed+stored, not reported per this round's valid-only discipline)  ({dt:.1f}s)",
                  flush=True)


def mean_valid(results, tag):
    vals = [r['valid_primary'] for r in results if r['tag'] == tag]
    return sum(vals) / len(vals) if vals else None


def mean_test(results, tag):
    vals = [r['test_primary'] for r in results if r['tag'] == tag]
    return sum(vals) / len(vals) if vals else None


if __name__ == '__main__':
    print("loading extended dataset (cached after first run)...")
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    print(f"loaded in {time.time()-t0:.1f}s")

    results = load_results()
    seeds3 = [0, 1, 2]
    seeds5 = [0, 1, 2, 3, 4]

    # ---- Step 0: harness-fidelity check vs iter24's own published seed 0/1/2 ----
    # (test IS printed here -- legitimate, comparing to already-published numbers)
    print("\n=== Step 0: harness-fidelity check (flat sampling, alpha=1.0 Laplace, "
          "n_buckets=10, iter24's exact feature set) ===")
    fid_tag = 'fidelity_vs_iter24'
    run_config(fid_tag, ITER24_FEATS, seeds3, results,
               sampling_mode='flat', sampling_alpha=1.0, decay_halflife=3,
               alpha=1.0, n_buckets=10, report_test=True)
    print("\nFidelity check vs iter24 published:")
    all_match = True
    for seed in seeds3:
        rec = next(r for r in results if r['tag'] == fid_tag and r['seed'] == seed)
        pub_valid, pub_test = ITER24_PUBLISHED[seed]
        dv, dt_ = rec['valid_primary'] - pub_valid, rec['test_primary'] - pub_test
        match = abs(dv) < 1e-4 and abs(dt_) < 1e-4
        all_match &= match
        print(f"  seed {seed}: valid this={rec['valid_primary']:.5f} pub={pub_valid:.5f} "
              f"(Δ{dv:+.5f}) | test this={rec['test_primary']:.5f} pub={pub_test:.5f} "
              f"(Δ{dt_:+.5f})  {'OK' if match else 'MISMATCH'}")
    if not all_match:
        print("\n*** HARNESS FIDELITY FAILED -- STOPPING. Do not trust downstream results. ***")
        sys.exit(1)
    print("\nHarness fidelity CONFIRMED (matches iter24's published numbers within 1e-4). "
          "Proceeding to triple-fusion sweep.")

    # ---- Step 1: 3-seed sweep over free-parameter combinations ----
    print("\n=== Step 1: 3-seed sweep (valid-only reporting) ===")
    SWEEP_CONFIGS = {
        'triple_fusion_default':    dict(sampling_alpha=0.5, alpha=0.5, n_buckets=20),
        'triple_fusion_nbuckets10': dict(sampling_alpha=0.5, alpha=0.5, n_buckets=10),
        'fusion_sampling_alpha0.25': dict(sampling_alpha=0.25, alpha=0.5, n_buckets=20),
        'fusion_sampling_alpha0.75': dict(sampling_alpha=0.75, alpha=0.5, n_buckets=20),
    }
    for tag, cfg in SWEEP_CONFIGS.items():
        run_config(tag, ITER24_FEATS, seeds3, results,
                   sampling_mode='decay', decay_halflife=3, report_test=False, **cfg)

    print("\nStep 1 sweep valid means (3-seed):")
    sweep_means = {}
    for tag in SWEEP_CONFIGS:
        mv = mean_valid(results, tag)
        sweep_means[tag] = mv
        margin = mv - ITER24_VALID_5SEED
        print(f"  {tag:32s} valid mean {mv:.5f}  (margin vs iter24 5-seed ref: {margin:+.5f})")

    best_tag = max(sweep_means, key=lambda t: sweep_means[t])
    best_valid_3 = sweep_means[best_tag]
    best_cfg = SWEEP_CONFIGS[best_tag]
    margin = best_valid_3 - ITER24_VALID_5SEED
    print(f"\nBest 3-seed config: '{best_tag}' (valid mean {best_valid_3:.5f}, margin {margin:+.5f})")

    # ---- Step 2 (conditional): 5-seed confirmation of ONLY the best config ----
    if margin > PROMOTION_MARGIN:
        print(f"\nMargin {margin:+.5f} clears the {PROMOTION_MARGIN} confirmation threshold "
              f"-- extending '{best_tag}' to 5 seeds.")
        run_config(best_tag, ITER24_FEATS, seeds5, results,
                   sampling_mode='decay', decay_halflife=3, report_test=False, **best_cfg)
        conf_valid = mean_valid(results, best_tag)
        print(f"5-seed '{best_tag}': valid mean {conf_valid:.5f}")

        # ---- Step 3: the ONE sanctioned test read, for the final winning config ----
        conf_test = mean_test(results, best_tag)
        print(f"\n=== Step 3: final test read (single sanctioned peek) for '{best_tag}' ===")
        print(f"5-seed '{best_tag}': valid mean {conf_valid:.5f}  test mean {conf_test:.5f}")
    else:
        print(f"\nMargin {margin:+.5f} does NOT clear the {PROMOTION_MARGIN} confirmation threshold "
              f"-- per protocol, NOT extending to 5 seeds. Reporting the 3-seed result honestly as "
              f"a non-promotion candidate.")

    print("\niter27 triple-fusion sweep complete.")

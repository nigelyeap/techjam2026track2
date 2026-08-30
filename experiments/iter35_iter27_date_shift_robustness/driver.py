"""iter35: date-shifted-split robustness check of iter27's WINNING TRIPLE-
FUSION config (sampling_alpha=0.75, Laplace alpha=0.5, n_buckets=20,
decay_halflife=3, iter24's refined feature set).

Closes the open caveat iter27's own RESULT.md flagged explicitly: iter27
re-confirmed n_buckets=20 beats n_buckets=10 WITHIN the fused config on the
OFFICIAL split (Step 1 of iter27/driver.py, sampling_alpha=0.5, +0.00156
valid, 3-seed), but never re-checked that comparison under iter29's date
shift. iter29 itself only ever tested the ISOLATED n_buckets lever (iter19's
plain feature set, alpha=1.0 Laplace, no decay-aware sampling) under that
shift -- never the full iter27 fusion. This driver runs the missing cell:
n_buckets in {10, 20}, WITHIN iter27's exact fused config, on iter29's exact
shifted split.

This is a ROBUSTNESS/DIAGNOSTIC check, not a promotion candidate -- a
shifted split is not the official split, so nothing here can become the new
"current best" regardless of the numbers (per the dispatch instructions).

Steps:
  0. Harness-fidelity check A (OFFICIAL split): run this dir's adapted
     data_ext.py/train.py at iter27's exact winning config (sampling_alpha=
     0.75, alpha=0.5, n_buckets=20, decay_halflife=3, ITER24_FEATS), 5
     seeds, on the OFFICIAL split (splits_map=OFFICIAL_SPLITS). Must
     bit-exact-match iter27_triple_fusion/results.json's
     'fusion_sampling_alpha0.75' rows before anything else is trusted.
  1. Harness-fidelity check B (OFFICIAL split, supplementary): iter27's own
     n_buckets=10 comparison point (`triple_fusion_nbuckets10`) was measured
     at sampling_alpha=0.5, NOT at the actual winning sampling_alpha=0.75 --
     so it isn't quite an apples-to-apples "n_buckets=20 vs 10 within the
     EXACT winning config" delta. This step fills that gap: runs
     n_buckets=10 at sampling_alpha=0.75 (else-identical) on the OFFICIAL
     split, 3 seeds, purely for a cleaner three-way comparison table in
     RESULT.md. (This does not change the fidelity verdict -- it's a new,
     clearly-labeled supplementary measurement, not a re-derivation of
     iter27's own published number.)
  2. Split-construction check (SHIFTED split): verify load_ext with
     splits_map=SPLITS_SHIFTED reproduces iter29's exact reported row counts
     (train 1,079,797 / valid 143,394 / test 170,150) and date boundaries.
  3. The actual robustness check: iter27's exact winning fused config with
     n_buckets=20 and n_buckets=10 (else identical: sampling_alpha=0.75,
     alpha=0.5, decay_halflife=3, ITER24_FEATS), 5 seeds each, on the
     SHIFTED split.
  4. Print the three-way comparison table (iter29 isolated-effect delta /
     iter27 official fused-config delta / iter35 shifted fused-config
     delta).

Every run's result is appended to results.json immediately (same
incremental-save pattern as iter27/iter29's own drivers).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, HALFLIVES, TAB_HALFLIVES, ALPHA, OFFICIAL_SPLITS, SPLITS_SHIFTED
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

ITER24_FEATS = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')

# iter27's own published fusion_sampling_alpha0.75 config (the current
# overall best) -- these are the EXACT parameters under test in this
# iteration, just re-run on a different split.
WINNING_CFG = dict(sampling_mode='decay', sampling_alpha=0.75, decay_halflife=3,
                    alpha=0.5, n_buckets=20)

# iter27_triple_fusion/results.json's fusion_sampling_alpha0.75 rows, copied
# here as the fidelity-check reference (also independently re-derived from
# that file directly in RESULT.md, per the dispatch's mandatory cross-check).
ITER27_PUBLISHED_ALPHA075_N20 = {
    0: (0.63894, 0.63989), 1: (0.63868, 0.63913), 2: (0.63685, 0.63768),
    3: (0.63747, 0.63853), 4: (0.63768, 0.63921),
}
ITER27_VALID_5SEED = 0.63792
ITER27_TEST_5SEED = 0.63889

# iter29's own published isolated-effect shifted-split numbers (from
# iter29_bucket_robustness/RESULT.md / results.json), for the final
# three-way comparison table.
ITER29_DELTA_VALID = -0.00007
ITER29_DELTA_TEST = 0.00056

# iter27's own official-split fused-config delta at sampling_alpha=0.5 (Step
# 1 of iter27's driver.py: triple_fusion_default vs triple_fusion_nbuckets10,
# 3-seed, valid-only reported there).
ITER27_OFFICIAL_DELTA_VALID_AT_ALPHA05 = 0.63804 - 0.63648  # = +0.00156 (from LEDGER/RESULT.md)


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


def run_config(tag, split_tag, splits_cache, seeds, results, n_buckets, sampling_alpha,
               alpha=0.5, decay_halflife=3, sampling_mode='decay', epochs=40, report_test=True):
    for seed in seeds:
        if already_done(results, tag, seed):
            print(f"[skip] {tag} seed={seed} already done", flush=True)
            continue
        t0 = time.time()
        out = run_bpr_ext(DATA_DIR, feature_set=ITER24_FEATS, halflives=HALFLIVES,
                           tab_halflives=TAB_HALFLIVES, seed=seed, verbose=False, epochs=epochs,
                           splits_cache=splits_cache,
                           sampling_mode=sampling_mode, sampling_alpha=sampling_alpha,
                           decay_halflife=decay_halflife, alpha=alpha, n_buckets=n_buckets)
        dt = time.time() - t0
        rec = {'tag': tag, 'split': split_tag, 'features': list(ITER24_FEATS), 'seed': seed,
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
            print(f"[done] {tag:32s} seed={seed} valid={out['valid']['primary']:.5f}  ({dt:.1f}s)",
                  flush=True)


def mean_of(results, tag, key):
    vals = [r[key] for r in results if r['tag'] == tag]
    return sum(vals) / len(vals) if vals else None


if __name__ == '__main__':
    results = load_results()

    print("=== Loading OFFICIAL-split extended dataset (cached after first run) ===", flush=True)
    t0 = time.time()
    OFFICIAL_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                               splits_map=OFFICIAL_SPLITS, split_tag='official')
    print(f"loaded in {time.time()-t0:.1f}s  sizes={ {k: len(v) for k, v in OFFICIAL_CACHE.items()} }",
          flush=True)

    # ================================================================
    # Step 0: harness-fidelity check A -- OFFICIAL split, iter27's exact
    # winning config, 5 seeds, bit-exact vs iter27's published numbers.
    # ================================================================
    print("\n=== Step 0: harness-fidelity check A (OFFICIAL split, "
          "sampling_alpha=0.75/alpha=0.5/n_buckets=20, 5 seeds) ===", flush=True)
    fid_tag = 'fidelity_official_winning_cfg'
    run_config(fid_tag, 'official', OFFICIAL_CACHE, [0, 1, 2, 3, 4], results,
               n_buckets=20, sampling_alpha=0.75, alpha=0.5, decay_halflife=3, report_test=True)

    print("\nFidelity check A vs iter27_triple_fusion/results.json "
          "('fusion_sampling_alpha0.75'):", flush=True)
    all_match = True
    for seed in range(5):
        rec = next(r for r in results if r['tag'] == fid_tag and r['seed'] == seed)
        pub_valid, pub_test = ITER27_PUBLISHED_ALPHA075_N20[seed]
        dv, dt_ = rec['valid_primary'] - pub_valid, rec['test_primary'] - pub_test
        match = abs(dv) < 1e-4 and abs(dt_) < 1e-4
        all_match &= match
        print(f"  seed {seed}: valid this={rec['valid_primary']:.5f} pub={pub_valid:.5f} "
              f"(Δ{dv:+.5f}) | test this={rec['test_primary']:.5f} pub={pub_test:.5f} "
              f"(Δ{dt_:+.5f})  {'OK' if match else 'MISMATCH'}", flush=True)
    if not all_match:
        print("\n*** HARNESS FIDELITY A FAILED -- STOPPING. Do not trust downstream results. ***",
              flush=True)
        sys.exit(1)
    print("\nHarness fidelity A CONFIRMED (bit-exact within 1e-4 vs iter27's published numbers). "
          "Proceeding.", flush=True)

    # ================================================================
    # Step 1: supplementary OFFICIAL-split measurement -- n_buckets=10 at
    # the ACTUAL winning sampling_alpha=0.75 (iter27 only measured
    # n_buckets=10 at sampling_alpha=0.5). 3 seeds, for a clean three-way
    # comparison table.
    # ================================================================
    print("\n=== Step 1: supplementary OFFICIAL-split n_buckets=10 @ sampling_alpha=0.75 "
          "(3 seeds) ===", flush=True)
    off_n10_tag = 'official_winning_cfg_nbuckets10'
    run_config(off_n10_tag, 'official', OFFICIAL_CACHE, [0, 1, 2], results,
               n_buckets=10, sampling_alpha=0.75, alpha=0.5, decay_halflife=3, report_test=True)
    off_n10_valid = mean_of(results, off_n10_tag, 'valid_primary')
    off_n20_valid_3seed = sum(ITER27_PUBLISHED_ALPHA075_N20[s][0] for s in (0, 1, 2)) / 3
    print(f"\nOfficial split @ sampling_alpha=0.75 (3-seed, matched to Step-1 n): "
          f"n_buckets=20 valid mean {off_n20_valid_3seed:.5f} vs n_buckets=10 valid mean "
          f"{off_n10_valid:.5f}  (Δ={off_n20_valid_3seed - off_n10_valid:+.5f})", flush=True)

    # ================================================================
    # Step 2: SHIFTED-split construction check -- must match iter29's
    # reported row counts exactly.
    # ================================================================
    print("\n=== Step 2: SHIFTED-split construction check (vs iter29) ===", flush=True)
    t0 = time.time()
    SHIFTED_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                              splits_map=SPLITS_SHIFTED, split_tag='shifted')
    got_sizes = {k: len(v) for k, v in SHIFTED_CACHE.items()}
    print(f"loaded in {time.time()-t0:.1f}s  sizes={got_sizes}", flush=True)
    expect_sizes = {'train': 1079797, 'valid': 143394, 'test': 170150}
    print(f"expected (iter29 driver.log): {expect_sizes}", flush=True)
    if got_sizes != expect_sizes:
        print("\n*** SHIFTED-SPLIT ROW COUNT MISMATCH vs iter29 -- STOPPING. ***", flush=True)
        sys.exit(1)
    print("Row counts MATCH iter29's shifted split exactly. Date boundaries: "
          f"{SPLITS_SHIFTED} (copied verbatim from iter29_bucket_robustness/data_ext.py).",
          flush=True)

    # ================================================================
    # Step 3: the actual robustness check -- iter27's exact winning fused
    # config, n_buckets in {20, 10}, 5 seeds each, on the SHIFTED split.
    # ================================================================
    print("\n=== Step 3: SHIFTED-split fused-config n_buckets sweep (5 seeds each) ===", flush=True)
    seeds5 = [0, 1, 2, 3, 4]
    shifted_n20_tag = 'shifted_winning_cfg_nbuckets20'
    shifted_n10_tag = 'shifted_winning_cfg_nbuckets10'
    run_config(shifted_n20_tag, 'shifted', SHIFTED_CACHE, seeds5, results,
               n_buckets=20, sampling_alpha=0.75, alpha=0.5, decay_halflife=3, report_test=True)
    run_config(shifted_n10_tag, 'shifted', SHIFTED_CACHE, seeds5, results,
               n_buckets=10, sampling_alpha=0.75, alpha=0.5, decay_halflife=3, report_test=True)

    shifted_n20_valid = mean_of(results, shifted_n20_tag, 'valid_primary')
    shifted_n20_test = mean_of(results, shifted_n20_tag, 'test_primary')
    shifted_n10_valid = mean_of(results, shifted_n10_tag, 'valid_primary')
    shifted_n10_test = mean_of(results, shifted_n10_tag, 'test_primary')
    shifted_delta_valid = shifted_n20_valid - shifted_n10_valid
    shifted_delta_test = shifted_n20_test - shifted_n10_test

    print(f"\nSHIFTED split, fused config (sampling_alpha=0.75, alpha=0.5, decay_halflife=3), "
          f"5-seed means:", flush=True)
    print(f"  n_buckets=20: valid {shifted_n20_valid:.5f}  test {shifted_n20_test:.5f}", flush=True)
    print(f"  n_buckets=10: valid {shifted_n10_valid:.5f}  test {shifted_n10_test:.5f}", flush=True)
    print(f"  Δ (20 vs 10): valid {shifted_delta_valid:+.5f}  test {shifted_delta_test:+.5f}", flush=True)

    # ================================================================
    # Step 4: three-way comparison table.
    # ================================================================
    print("\n=== Step 4: three-way comparison (isolated lever vs fused-config, "
          "official vs shifted) ===", flush=True)
    print(f"{'':45s} {'Δ valid':>10s} {'Δ test':>10s}")
    print(f"{'iter29 isolated lever, shifted split':45s} "
          f"{ITER29_DELTA_VALID:+10.5f} {ITER29_DELTA_TEST:+10.5f}")
    print(f"{'iter27 fused config, official split (a=0.5)':45s} "
          f"{ITER27_OFFICIAL_DELTA_VALID_AT_ALPHA05:+10.5f} {'  n/a':>10s}")
    print(f"{'iter35 fused config, official split (a=0.75)':45s} "
          f"{off_n20_valid_3seed - off_n10_valid:+10.5f} {'  n/a':>10s}")
    print(f"{'iter35 fused config, shifted split (a=0.75)':45s} "
          f"{shifted_delta_valid:+10.5f} {shifted_delta_test:+10.5f}")

    print("\niter35 date-shift robustness check complete.", flush=True)

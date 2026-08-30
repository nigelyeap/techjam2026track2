"""Sweep driver for iter31 (multi-task auxiliary loss). Loads the (cached)
extended dataset and the auxiliary engagement labels ONCE, then loops over
configs x seeds, writing incremental results to results.json after every
single run so partial progress survives interruption (per protocol).

Phase 0 (harness-fidelity): aux_weight=0.0, seeds 0-2, on iter24's exact
winning feature set (decay_rate_2.5, decay_act_2.5, decay_tab_3, last1,
lastk_rate, gap). Must reproduce iter24's own published per-seed numbers
(from experiments/iter24_decay_tab_refine/results.json) closely -- proves
this harness isn't a reimplementation drift before any multi-task change
is switched on.

Phase 1 (main sweep): aux_weight in {0.1, 0.2, 0.3}, all 5 auxiliary tasks
(is_click, is_like, is_follow, is_comment, is_forward) combined with equal
weight (mean over tasks), 3 seeds each. Selection is on VALID ONLY -- test
is never inspected per sweep row.

Phase 2 (conditional 5-seed confirmation): if the best aux_weight beats
iter24's 5-seed valid reference (0.63251) by more than ~0.001, extend that
single config to seeds 3-4 for confirmation. Otherwise report the 3-seed
sweep as a non-promotion, honestly, without extending.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ext import load_ext, load_aux_labels, HALFLIVES, TAB_HALFLIVES, AUX_LABELS
from train import run_bpr_ext

DATA_DIR = '../../KuaiRand-Pure/data'
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')

FEATURE_SET = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
ITER24_5SEED_VALID = 0.63251  # reference: iter24's exact published 5-seed valid mean

# iter24's own per-seed reference numbers (from experiments/iter24_decay_tab_refine/results.json,
# tag 'decay_rate_2.5+decay_act_2.5+decay_tab_3+mom'), used for the harness-fidelity check.
ITER24_REF = {
    0: {'valid': 0.6326001882553101, 'test': 0.6283901929855347},
    1: {'valid': 0.6330777406692505, 'test': 0.6294211149215698},
    2: {'valid': 0.6317896842956543, 'test': 0.6268689632415771},
    3: {'valid': 0.6320827603340149, 'test': 0.6285477876663208},
    4: {'valid': 0.632981538772583,  'test': 0.6289214491844177},
}


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


def run_config(tag, aux_weight, aux_tasks, seeds, results, epochs=40):
    for seed in seeds:
        if already_done(results, tag, seed):
            print(f"[skip] {tag} seed={seed} already done")
            continue
        t0 = time.time()
        out = run_bpr_ext(DATA_DIR, feature_set=FEATURE_SET, halflives=HALFLIVES,
                           tab_halflives=TAB_HALFLIVES, seed=seed, verbose=False, epochs=epochs,
                           splits_cache=SPLITS_CACHE, aux_weight=aux_weight, aux_tasks=aux_tasks,
                           aux_cache=AUX_CACHE)
        dt = time.time() - t0
        rec = {'tag': tag, 'aux_weight': aux_weight, 'aux_tasks': list(aux_tasks), 'seed': seed,
               'valid_primary': float(out['valid']['primary']), 'valid_gauc': float(out['valid']['GAUC']),
               'valid_ndcg5': float(out['valid']['nDCG@5']),
               'test_primary': float(out['test']['primary']), 'test_gauc': float(out['test']['GAUC']),
               'test_ndcg5': float(out['test']['nDCG@5']), 'seconds': dt}
        results.append(rec)
        save_results(results)
        print(f"[done] {tag:30s} seed={seed} valid={out['valid']['primary']:.5f} "
              f"test={out['test']['primary']:.5f}  ({dt:.1f}s)", flush=True)


def mean_valid(results, tag):
    vals = [r['valid_primary'] for r in results if r['tag'] == tag]
    return sum(vals) / len(vals) if vals else None


if __name__ == '__main__':
    print("loading extended dataset (cached after first run)...")
    t0 = time.time()
    SPLITS_CACHE = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)
    print(f"loaded features in {time.time()-t0:.1f}s")
    t0 = time.time()
    AUX_CACHE = load_aux_labels(DATA_DIR)
    print(f"loaded aux labels in {time.time()-t0:.1f}s "
          f"(train rows={len(AUX_CACHE['train'][AUX_LABELS[0]])})")

    results = load_results()
    seeds3 = [0, 1, 2]

    # ---- Phase 0: harness-fidelity check (aux_weight=0.0 must reproduce iter24) ----
    print("\n=== Phase 0: harness-fidelity check (aux_weight=0.0 vs iter24 published seeds 0-2) ===")
    run_config('harness_aux0', 0.0, AUX_LABELS, seeds3, results)
    max_abs_err = 0.0
    for seed in seeds3:
        rec = next(r for r in results if r['tag'] == 'harness_aux0' and r['seed'] == seed)
        ref = ITER24_REF[seed]
        err_v = abs(rec['valid_primary'] - ref['valid'])
        err_t = abs(rec['test_primary'] - ref['test'])
        max_abs_err = max(max_abs_err, err_v, err_t)
        print(f"  seed={seed} valid got={rec['valid_primary']:.7f} ref={ref['valid']:.7f} err={err_v:.2e} | "
              f"test got={rec['test_primary']:.7f} ref={ref['test']:.7f} err={err_t:.2e}")
    print(f"Harness-fidelity max abs err vs iter24's published seeds 0-2: {max_abs_err:.2e}")
    if max_abs_err > 1e-4:
        print("WARNING: harness fidelity error exceeds 1e-4 -- investigate before trusting the sweep!")
    else:
        print("Harness fidelity OK (aux_weight=0.0 reproduces iter24 within float noise).")

    # ---- Phase 1: aux_weight sweep, all 5 tasks, 3 seeds ----
    print("\n=== Phase 1: aux_weight sweep (all 5 aux tasks, 3 seeds) ===")
    aux_weights = [0.1, 0.2, 0.3]
    for w in aux_weights:
        tag = f'mtl_all5_w{w}'
        run_config(tag, w, AUX_LABELS, seeds3, results)

    sweep_means = {w: mean_valid(results, f'mtl_all5_w{w}') for w in aux_weights}
    print("\nPhase 1 valid means:", {w: round(v, 5) for w, v in sweep_means.items()})
    best_w = max(sweep_means, key=lambda w: sweep_means[w])
    best_tag = f'mtl_all5_w{best_w}'
    best_mean = sweep_means[best_w]
    print(f"Best aux_weight: {best_w} (3-seed valid mean {best_mean:.5f}) "
          f"vs iter24 5-seed reference {ITER24_5SEED_VALID:.5f}")

    margin = best_mean - ITER24_5SEED_VALID
    print(f"Margin vs iter24 5-seed valid mean: {margin:+.5f}")

    # ---- Phase 2 (conditional): 5-seed confirmation if a real margin over iter24 ----
    if margin > 0.001:
        print(f"\n=== Phase 2: 5-seed confirmation of {best_tag} (aux_weight={best_w}) ===")
        seeds5 = [0, 1, 2, 3, 4]
        run_config(best_tag, best_w, AUX_LABELS, seeds5, results)
        conf_mean = mean_valid(results, best_tag)
        print(f"5-seed valid mean for {best_tag}: {conf_mean:.5f}")
    else:
        print("\nMargin below confirmation threshold (~0.001) -- no 5-seed run triggered by driver. "
              "Reporting the 3-seed sweep as a non-promotion.")

    print("\niter31 multi-task sweep complete.")

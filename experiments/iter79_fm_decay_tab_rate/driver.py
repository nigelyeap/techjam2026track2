"""iter79: does the FM side benefit from the same tab-decay COUNT-to-RATE
swap that gave the GBM side a real gain in iter63?

The FM's live promoted feature set (driver.py's ITER24_FEATS in
iter27_triple_fusion, used by iter38's 5-seed ensemble -- the FM half of
the current final submission) is still
('decay_rate_2.5','decay_act_2.5','decay_tab_3','last1','lastk_rate','gap')
-- decay_tab_3 is a bucketed COUNT (decayed positive-row count in the same
tab), unchanged since Round 7/iter24. It has never received iter63's
insight (Laplace-smoothed decayed RATE = pos/(pos+neg) instead of a raw
decayed positive count) even though that swap is exactly what turned a
flat GBM feature into a real +0.00438 valid / +0.00602 test GBM gain.
This is a genuinely new-in-kind test (touches the FM's own feature
encoding for the first time since Round 18), not another GBM variant.

data_ext.py here = iter27_triple_fusion/data_ext.py (has encode_ext,
compute_final_decayed_pos) + iter63_decay_tab_rate/data_ext.py's verified
compute_decay_tab_features/load_ext diff (adds decayed_tab_total tracking)
+ a new 'decay_tab_rate' kind added to encode_ext's three dispatch points
(parse_feat, edge-building, extra_val). train.py = byte-for-byte copy of
iter27_triple_fusion/train.py (local import now resolves to this dir's own
data_ext.py).

Protocol: harness-fidelity check first (reproduce iter27's own seed=0
result for the exact promoted config, from results.json), then swap
decay_tab_3 -> decay_tab_rate_3 in the feature set, single seed=0
(exploratory, per project convention -- 5-seed reserved for
surprising/borderline results).
"""
import os, sys, json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from train import run_bpr_ext          # noqa: E402  this dir's own train.py
from data_ext import load_ext, HALFLIVES, TAB_HALFLIVES, ALPHA   # noqa: E402

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

PROMOTED_KW = dict(sampling_mode='decay', sampling_alpha=0.75, decay_halflife=3,
                    alpha=0.5, n_buckets=20)
COUNT_FEATS = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')
RATE_FEATS = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_rate_3', 'last1', 'lastk_rate', 'gap')

EXPECT_VALID = 0.6389358639717102
EXPECT_TEST = 0.6398863196372986


def _clean(d):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}


def main():
    results = []
    out_path = os.path.join(_THIS_DIR, 'results.json')

    def _flush():
        with open(out_path, 'w') as fh:
            json.dump(results, fh, indent=2)

    print("=== loading features (cached once across both runs) ===", flush=True)
    splits = load_ext(DATA_DIR, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES)

    print("\n=== harness-fidelity check (decay_tab_3 count, seed 0) ===", flush=True)
    res0 = run_bpr_ext(DATA_DIR, feature_set=COUNT_FEATS, seed=0, verbose=True,
                        splits_cache=splits, **PROMOTED_KW)
    va0, te0 = res0['valid']['primary'], res0['test']['primary']
    print(f"  got valid={va0:.7f} test={te0:.7f}", flush=True)
    print(f"  expect valid={EXPECT_VALID:.7f} test={EXPECT_TEST:.7f}", flush=True)
    assert abs(va0 - EXPECT_VALID) < 1e-4 and abs(te0 - EXPECT_TEST) < 1e-4, \
        "harness fidelity check FAILED"
    print("  PASS", flush=True)
    results.append({'tag': 'baseline_decay_tab_3_count', 'features': list(COUNT_FEATS),
                     'seed': 0, **PROMOTED_KW,
                     'valid': _clean(res0['valid']), 'test': _clean(res0['test'])})
    _flush()

    print("\n=== decay_tab_rate_3 swap (seed 0) ===", flush=True)
    res1 = run_bpr_ext(DATA_DIR, feature_set=RATE_FEATS, seed=0, verbose=True,
                        splits_cache=splits, **PROMOTED_KW)
    va1, te1 = res1['valid']['primary'], res1['test']['primary']
    print(f"  valid={va1:.7f} test={te1:.7f}", flush=True)
    results.append({'tag': 'decay_tab_rate_3', 'features': list(RATE_FEATS),
                     'seed': 0, **PROMOTED_KW,
                     'valid': _clean(res1['valid']), 'test': _clean(res1['test'])})
    _flush()

    print("\n=== summary ===", flush=True)
    print(f"  baseline (decay_tab_3 count):  valid={va0:.5f} test={te0:.5f}")
    print(f"  decay_tab_rate_3:               valid={va1:.5f} test={te1:.5f}")
    print(f"  delta:                          valid={va1-va0:+.5f} test={te1-te0:+.5f}")


if __name__ == '__main__':
    main()

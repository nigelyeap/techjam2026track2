"""iter63 (part 2): 5-seed confirmation of the single-seed finding --
`rate_only` (decay_tab_rate_3 REPLACING decay_tab_3) beat `baseline`
(decay_tab_3 count, iter55's exact feature set) by +0.00116 valid /
+0.00076 test at seed=0. Per protocol, a >0.001 valid gain gets a 5-seed
confirmation before being treated as unambiguously real.
"""
import os, sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
import train as t  # noqa: E402

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
SEEDS = (0, 1, 2, 3, 4)

if __name__ == '__main__':
    cache_base = t.prepare(DATA_DIR, 'baseline')
    cache_rate = t.prepare(DATA_DIR, 'rate_only')

    vas_b, tes_b, vas_r, tes_r = [], [], [], []
    for seed in SEEDS:
        _, va_b, te_b, _ = t.run(DATA_DIR, 'baseline', seed=seed, _cache=cache_base)
        _, va_r, te_r, _ = t.run(DATA_DIR, 'rate_only', seed=seed, _cache=cache_rate)
        d = va_r['primary'] - va_b['primary']
        print(f"seed={seed}  baseline valid={va_b['primary']:.5f} test={te_b['primary']:.5f}  |  "
              f"rate_only valid={va_r['primary']:.5f} test={te_r['primary']:.5f}  |  delta={d:+.5f}", flush=True)
        vas_b.append(va_b['primary']); tes_b.append(te_b['primary'])
        vas_r.append(va_r['primary']); tes_r.append(te_r['primary'])

    vas_b, vas_r = np.array(vas_b), np.array(vas_r)
    tes_b, tes_r = np.array(tes_b), np.array(tes_r)
    wins = int((vas_r > vas_b).sum())
    print(f"\nbaseline  5-seed valid: mean={vas_b.mean():.5f} std={vas_b.std():.5f}  "
          f"test: mean={tes_b.mean():.5f} std={tes_b.std():.5f}")
    print(f"rate_only 5-seed valid: mean={vas_r.mean():.5f} std={vas_r.std():.5f}  "
          f"test: mean={tes_r.mean():.5f} std={tes_r.std():.5f}")
    print(f"mean delta valid = {vas_r.mean()-vas_b.mean():+.5f}   wins={wins}/5")

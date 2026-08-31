"""Trains iter64's SASRec history encoder (a given seed) and saves raw score
arrays to disk. Kept in its own process (no lightgbm import) after
discovering torch+lightgbm in the same process reliably segfaults here.
Usage: python3 gen_sasrec_scores.py [seed]
"""
import os, sys
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from train import run  # noqa: E402

DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.set_num_threads(4)
    model, va_m, te_m, cache = run(DATA_DIR, epochs=25, patience=4, bs=4096, seed=seed, verbose=True)
    va_scores, te_scores, y_va, u_va, y_te, u_te = cache
    print(f"[seed {seed}] valid={va_m['primary']:.5f} test={te_m['primary']:.5f}", flush=True)
    np.savez(os.path.join(_THIS_DIR, f'sasrec_scores_seed{seed}.npz'),
              va_scores=va_scores, te_scores=te_scores,
              y_va=np.asarray(y_va), y_te=np.asarray(y_te),
              u_va=np.asarray(u_va, dtype=object), u_te=np.asarray(u_te, dtype=object))
    print(f"SAVED sasrec_scores_seed{seed}.npz", flush=True)

"""iter6: 与 baseline.py --model fm 完全相同的 pointwise-logloss FM 训练循环
(k=16, lr=0.001, bs=8192, epochs=40, patience=4)，唯一区别是喂给它的是
data_ext.py 产出的 6 特征域编码（多了 hist_affinity），而不是 data.py 的
5 特征域编码。不改动 loss / 优化器 / early-stop 逻辑 —— 这个实验只隔离
"加一个因果历史特征"这一个变量。

直接复用 ../../baseline.py 里的 FM 类（未修改），以及 ../../evaluate.py
的 evaluate()（未修改，口径不能动）。
"""
import argparse, os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluate import evaluate          # noqa: E402  官方评测口径，未改
from baseline import FM                # noqa: E402  同一个 FM 实现，未改

from data_ext import load_ext, encode_ext, FIELDS_EXT  # noqa: E402


def run_fm_ext(data_dir, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    splits = load_ext(data_dir)
    enc, dim = encode_ext(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(os.path.dirname(__file__), '..', '..',
                                                         'KuaiRand-Pure', 'data'))
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ... fields={FIELDS_EXT}")
    res = run_fm_ext(a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, bs=a.bs,
                      patience=a.patience, seed=a.seed, verbose=not a.quiet)
    print(f"\n=== iter6 fm+hist_affinity (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
    def _clean(d):
        return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}
    print(json.dumps({'seed': a.seed, 'valid': _clean(res['valid']), 'test': _clean(res['test'])}))

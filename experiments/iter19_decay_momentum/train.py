"""iter19: FM + activity-weighted BPR pairwise loss (identical loss/sampling
mechanism to iter3/iter9/iter16/iter18), fed the FUSED extended feature
encoding from this dir's data_ext.py, which combines iter16's decay features
with iter18's momentum features (both stacked on iter9's flat base).

Loss, optimizer, sampling, hyperparams are all a line-for-line copy of
iter16/iter18's train.py (themselves copied from iter3/iter9). Reuses
baseline.FM unmodified and evaluate.py's evaluate() unmodified.
"""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate           # noqa: E402  official eval, unmodified
from baseline import FM, sigmoid        # noqa: E402  same FM class as iter3/iter9

from data_ext import load_ext, encode_ext, BASE_FIELDS, HALFLIVES  # noqa: E402


def build_pos_neg_index(y, users):
    """Identical to iter3/iter9/iter16/iter18's train.py -- copied verbatim."""
    by_user_pos, by_user_neg = {}, {}
    for i, (yi, u) in enumerate(zip(y, users)):
        (by_user_pos if yi == 1 else by_user_neg).setdefault(u, []).append(i)
    eligible = sorted(set(by_user_pos) & set(by_user_neg))

    def flatten(by_user):
        starts, lens, flat = [], [], []
        off = 0
        for u in eligible:
            idx = by_user[u]
            starts.append(off); lens.append(len(idx)); flat.extend(idx)
            off += len(idx)
        return (np.array(flat, dtype=np.int64), np.array(starts, dtype=np.int64),
                np.array(lens, dtype=np.int64))

    pos_flat, pos_start, pos_len = flatten(by_user_pos)
    neg_flat, neg_start, neg_len = flatten(by_user_neg)
    return eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len


def sample_pairs(rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                  user_cumw=None, user_totalw=None):
    """Identical to iter3/iter9/iter16/iter18's train.py -- copied verbatim."""
    if user_cumw is not None:
        picked = np.searchsorted(user_cumw, rng.random(bs) * user_totalw, side='right')
        picked = np.minimum(picked, n_users - 1)
    else:
        picked = rng.integers(0, n_users, size=bs)
    pos_rand = (rng.random(bs) * pos_len[picked]).astype(np.int64)
    neg_rand = (rng.random(bs) * neg_len[picked]).astype(np.int64)
    pos_rows = pos_flat[pos_start[picked] + pos_rand]
    neg_rows = neg_flat[neg_start[picked] + neg_rand]
    return pos_rows, neg_rows


def bpr_step(m, Xpos, Xneg):
    """Identical to iter3/iter9/iter16/iter18's train.py -- copied verbatim."""
    B = len(Xpos)
    zpos, Epos, Spos = m.logits(Xpos)
    zneg, Eneg, Sneg = m.logits(Xneg)
    d = zpos - zneg
    gpos = ((sigmoid(d) - 1) / B).astype(np.float32)
    gneg = -gpos

    X = np.concatenate([Xpos, Xneg], axis=0)
    E = np.concatenate([Epos, Eneg], axis=0)
    S = np.concatenate([Spos, Sneg], axis=0)
    g = np.concatenate([gpos, gneg], axis=0)

    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, X, g[:, None])
    np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
    gV += m.l2 * m.V; gW += m.l2 * m.W
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.b -= m.lr * g.sum()
    return float(-np.mean(np.log(sigmoid(d) + 1e-9)))


def run_bpr_ext(data_dir, feature_set=('decay_rate_3', 'decay_act_3', 'tab'), halflives=HALFLIVES,
                 k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
                 steps_mult=1, K=5, splits_cache=None):
    """splits_cache: optional pre-loaded load_ext() result, to avoid recomputing
    the (expensive, 3-traversal) feature pass across many sweep configs."""
    splits = splits_cache if splits_cache is not None else load_ext(data_dir, halflives=halflives, K=K)
    enc, dim = encode_ext(splits, feature_set=feature_set, halflives=halflives)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs))) * steps_mult

    user_cumw = np.cumsum(pos_len.astype(np.float64))
    user_totalw = user_cumw[-1]

    if verbose:
        fields = list(BASE_FIELDS) + list(feature_set)
        print(f"  fields={fields} dim={dim}")
        print(f"  BPR-eligible train users: {n_users} (>=1 pos & >=1 neg) | "
              f"{steps_per_epoch} steps/epoch | activity-weighted user sampling")

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(steps_per_epoch):
            Xpos_rows, Xneg_rows = sample_pairs(
                rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                user_cumw=user_cumw, user_totalw=user_totalw)
            losses.append(bpr_step(m, Xtr[Xpos_rows], Xtr[Xneg_rows]))
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
    ap.add_argument('--data_dir', default='../../KuaiRand-Pure/data')
    ap.add_argument('--features', default='decay_rate_3,decay_act_3,tab',
                    help="comma-separated subset/order of "
                         "{activity,tab,rate,decay_rate_3,decay_act_3,last1,lastk_rate,gap}")
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps_mult', type=int, default=1)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    feature_set = tuple(f for f in a.features.split(',') if f)
    print(f"loading {a.data_dir} ... extra features={feature_set}")
    res = run_bpr_ext(a.data_dir, feature_set=feature_set, k=a.k, lr=a.lr, epochs=a.epochs,
                       bs=a.bs, patience=a.patience, seed=a.seed, verbose=not a.quiet,
                       steps_mult=a.steps_mult, K=a.K)
    print(f"\n=== iter19 fm+bpr_weighted+{'+'.join(feature_set)} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
    def _clean(d):
        return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}
    print(json.dumps({'seed': a.seed, 'features': list(feature_set),
                       'valid': _clean(res['valid']), 'test': _clean(res['test'])}))

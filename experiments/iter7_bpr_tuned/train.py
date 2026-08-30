"""iter7: same FM+BPR architecture as iter3 (experiments/iter3_bpr_weighted/train.py),
generalizing iter3's activity-weighted user sampling from a fixed linear weight
(pos_len^1) to a tunable exponent: P(user) proportional to pos_len[user]^alpha.

alpha=1.0 reproduces iter3 exactly (sanity check on this reimplementation).
alpha=0.0 would be uniform (iter2, already known worse, not re-run here).
alpha<1 moves sampling toward uniform (flatter); alpha>1 moves it toward being
even more skewed to highly active users than iter3 already is.

Also exposes --k and --lr as CLI flags (iter3 hardcoded k=16, lr=0.001) to sweep
embedding capacity and step size at the best-found alpha.

Everything else (FM class, BPR loss, pos/neg-per-user sampling within a chosen
user, cumsum+searchsorted user-pick mechanism, bs/patience/epochs defaults) is
copied unmodified from iter3.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data import load, encode, FIELDS
from evaluate import evaluate
from baseline import FM, sigmoid


def build_pos_neg_index(y, users):
    """For each user with >=1 positive AND >=1 negative row, build a CSR-style
    flat index of positive/negative row positions so sampling is vectorizable.
    Users that are all-positive or all-negative are dropped (no pairs to form)."""
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
    """user_cumw/user_totalw: optional cumulative-sum sampling table over eligible
    users, weighted by pos_len**alpha (see run_bpr). If given, users are picked
    proportional to their weight via cumsum + searchsorted (vectorized, avoids the
    slow rng.choice(..., p=...) path for large n_users). If omitted, falls back to
    uniform sampling over eligible users (iter2 behaviour)."""
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
    """Same Adam/FM update math as baseline.FM.step, but the per-row logit
    gradient comes from the BPR pairwise loss instead of pointwise logloss."""
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


def run_bpr(splits, k=16, lr=0.001, alpha=1.0, epochs=40, bs=8192, patience=4, seed=0,
            verbose=True, steps_mult=1):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs))) * steps_mult

    # iter7: generalize iter3's linear (pos_len^1) activity weighting to a
    # tunable exponent alpha: P(user) proportional to pos_len[user]**alpha.
    # alpha=1.0 reproduces iter3 exactly. alpha=0.0 would be uniform (iter2).
    if alpha == 0.0:
        user_cumw, user_totalw = None, None
    else:
        weights = pos_len.astype(np.float64) ** alpha
        user_cumw = np.cumsum(weights)
        user_totalw = user_cumw[-1]

    if verbose:
        print(f"  BPR-eligible train users: {n_users} (>=1 pos & >=1 neg) | "
              f"{steps_per_epoch} steps/epoch | alpha={alpha} k={k} lr={lr}")

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
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--alpha', type=float, default=1.0,
                    help='user-sampling weight exponent: P(user) ~ pos_len[user]**alpha. '
                         '0=uniform (iter2), 1=iter3 linear weighting (default).')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps_mult', type=int, default=1,
                    help='multiply steps/epoch (baseline touches ~139 batches/epoch; '
                         'BPR only touches ~47 by default since it is keyed off positive count)')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = run_bpr(splits, k=a.k, lr=a.lr, alpha=a.alpha, epochs=a.epochs, bs=a.bs,
                   patience=a.patience, seed=a.seed, steps_mult=a.steps_mult)
    print(f"\n=== fm+bpr (alpha={a.alpha} k={a.k} lr={a.lr} seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")

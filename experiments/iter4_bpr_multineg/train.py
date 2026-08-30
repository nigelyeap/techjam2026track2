"""BPR (pairwise ranking) training on top of the same FM architecture as baseline.py,
now with MULTIPLE negatives per positive per training step.

Starting point: experiments/iter2_bpr_uniform/train.py (single random negative per
positive, uniform-user sampling). iter2 was REJECTED (test primary 0.5923 vs iter1's
0.5953) -- hypothesis in the ledger was that a single random negative gives a very
noisy pairwise gradient estimate. Standard implicit-feedback BPR practice is to sample
several negatives per positive and average the pairwise loss/gradient, which should
reduce gradient variance per step without changing what the loss is an unbiased
estimate of.

Change vs iter2: for each sampled (user, positive row), sample N_NEG=4 negatives
(with replacement) from that same user's negative rows instead of 1. The BPR loss for
that positive is the average of the 4 per-negative pairwise losses. Correspondingly:
  - each positive row's gradient = the AVERAGE of its gradient across the 4
    (pos, neg_i) comparisons (not the sum) -- see bpr_step_multineg below.
  - each of the 4 negative rows gets 1/4 the gradient weight it would carry in the
    1:1 (iter2) case.
This keeps the gradient magnitude per step comparable to iter2 rather than scaling up
4x just because more rows are touched per step.

Batch-size choice: positive batch size bs=8192 is left UNCHANGED (same as iter2), so
each step now touches 8192 positives + 8192*4=32768 negatives = 40960 rows total (vs
16384 in iter2). Rationale (per assignment instructions): more signal per step should
mean each step is worth more / fewer steps are needed to converge, not that we should
hold compute-per-step fixed by shrinking the positive batch. steps_per_epoch is left
keyed off positive-row count (same formula as iter2), so epoch semantics ("one pass
over positives") are preserved -- only the negative sampling and loss/gradient inside
each step changed.

Reuses baseline.FM unmodified (same V/W/b, same Adam optimizer, same logits()).
"""
import argparse, sys, time
sys.path.insert(0, '../..')
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate
from baseline import FM, sigmoid

N_NEG_DEFAULT = 4


def build_pos_neg_index(y, users):
    """For each user with >=1 positive AND >=1 negative row, build a CSR-style
    flat index of positive/negative row positions so sampling is vectorizable.
    Users that are all-positive or all-negative are dropped (no pairs to form).
    Identical to iter2's build_pos_neg_index -- unchanged."""
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


def sample_pairs_multineg(rng, n_users, bs, n_neg, pos_flat, pos_start, pos_len,
                           neg_flat, neg_start, neg_len):
    """Sample bs (user, positive-row) pairs, then n_neg negative rows per pair
    (with replacement) from the SAME user. Returns pos_rows (bs,) and
    neg_rows (bs, n_neg)."""
    picked = rng.integers(0, n_users, size=bs)
    pos_rand = (rng.random(bs) * pos_len[picked]).astype(np.int64)
    pos_rows = pos_flat[pos_start[picked] + pos_rand]

    neg_rand = (rng.random((bs, n_neg)) * neg_len[picked][:, None]).astype(np.int64)
    neg_rows = neg_flat[neg_start[picked][:, None] + neg_rand]
    return pos_rows, neg_rows


def bpr_step_multineg(m, Xpos, Xneg, n_neg):
    """Same Adam/FM update math as baseline.FM.step / iter2's bpr_step, but each
    positive is compared against n_neg negatives and the per-row gradients are
    averaged over the n_neg comparisons (not summed), so the gradient magnitude
    stays comparable to the 1:1 (iter2) case.

    Xpos: (B, F) -- one row per sampled positive.
    Xneg: (B, n_neg, F) -- n_neg negative rows per sampled positive.
    """
    B = len(Xpos)
    zpos, Epos, Spos = m.logits(Xpos)                       # (B,), (B,F,k), (B,k)

    Xneg_flat = Xneg.reshape(-1, Xneg.shape[-1])             # (B*n_neg, F)
    zneg_flat, Eneg_flat, Sneg_flat = m.logits(Xneg_flat)    # (B*n_neg,), ...

    zneg = zneg_flat.reshape(B, n_neg)
    d = zpos[:, None] - zneg                                 # (B, n_neg)
    sig = sigmoid(d)

    # Per-(pos, neg_i) pairwise gradient, pre-divided by n_neg so that summing
    # over the n_neg axis for the positive gives the AVERAGE gradient across its
    # n_neg comparisons, and each individual negative carries 1/n_neg the weight
    # it would carry in the 1:1 case. Also divided by B, same normalization as
    # iter2's per-batch mean.
    gpos_per_j = ((sig - 1) / (n_neg * B)).astype(np.float32)  # (B, n_neg)
    gpos = gpos_per_j.sum(axis=1)                               # (B,)
    gneg_flat = (-gpos_per_j).reshape(-1).astype(np.float32)    # (B*n_neg,)

    X = np.concatenate([Xpos, Xneg_flat], axis=0)
    E = np.concatenate([Epos, Eneg_flat], axis=0)
    S = np.concatenate([Spos, Sneg_flat], axis=0)
    g = np.concatenate([gpos, gneg_flat], axis=0)

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
    return float(-np.mean(np.log(sig + 1e-9)))


def run_bpr_multineg(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0,
                      n_neg=N_NEG_DEFAULT, verbose=True, steps_mult=1):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs))) * steps_mult
    if verbose:
        print(f"  BPR-eligible train users: {n_users} (>=1 pos & >=1 neg) | "
              f"{steps_per_epoch} steps/epoch | n_neg={n_neg} | "
              f"rows/step = {bs} pos + {bs*n_neg} neg = {bs*(1+n_neg)}")

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(steps_per_epoch):
            Xpos_rows, Xneg_rows = sample_pairs_multineg(
                rng, n_users, bs, n_neg, pos_flat, pos_start, pos_len,
                neg_flat, neg_start, neg_len)
            losses.append(bpr_step_multineg(m, Xtr[Xpos_rows], Xtr[Xneg_rows], n_neg))
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
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_neg', type=int, default=N_NEG_DEFAULT)
    ap.add_argument('--steps_mult', type=int, default=1,
                    help='multiply steps/epoch (kept off positive count, same as iter2)')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = run_bpr_multineg(splits, k=a.k, lr=a.lr, epochs=a.epochs, bs=a.bs,
                            patience=a.patience, seed=a.seed, n_neg=a.n_neg,
                            steps_mult=a.steps_mult)
    print(f"\n=== fm+bpr_multineg (n_neg={a.n_neg}, seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")

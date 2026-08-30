"""iter28: DeepFM-style model (baseline.FM's linear+pairwise terms unchanged,
plus a hand-rolled MLP deep part -- see model.py, copied VERBATIM from
iter26_deepfm/model.py) trained with the SAME activity-weighted BPR pairwise
loss / sampling / optimizer hyperparameters as iter19/iter24 (itself a
line-for-line copy of iter3/iter9/iter16/iter18/iter19/iter20/iter24's
train.py), fed iter24's REFINED feature set (`decay_rate_2.5, decay_act_2.5,
decay_tab_3, last1, lastk_rate, gap`) via this dir's data_ext.py, which is a
byte-identical copy of `iter24_decay_tab_refine/data_ext.py`.

This combines two independent Round 7 wins that were never stacked:
  - iter26: DeepFM deep part (architecture axis) on iter19's ORIGINAL
    feature set (decay_rate_3, decay_act_3, tab (flat), last1, lastk_rate,
    gap).
  - iter24: refined feature set (halflife 3d->2.5d, decayed tab_pos instead
    of flat tab) on the plain FM (no deep part).

build_pos_neg_index / sample_pairs / deepfm_bpr_step are copied VERBATIM from
iter26/train.py (byte-identical) -- these are architecture/optimizer code,
independent of the input feature set. The only change from iter26/train.py is
which data_ext module is imported (this dir's iter24-derived copy instead of
iter19's) and DEFAULT_FEATURES (iter24's winning 6-field set instead of
iter19's). run_deepfm_bpr also threads through `tab_halflives`, which
iter24's data_ext.load_ext/encode_ext require and iter19's data_ext did not
have.

See iter26/model.py and iter26/RESULT.md for the full hand-derived
forward/backward math of the deep part -- not re-derived here.
"""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate           # noqa: E402  official eval, unmodified
from baseline import sigmoid            # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from model import DeepFM                # noqa: E402
from data_ext import load_ext, encode_ext, BASE_FIELDS, HALFLIVES, TAB_HALFLIVES  # noqa: E402

DEFAULT_FEATURES = ('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3', 'last1', 'lastk_rate', 'gap')


def build_pos_neg_index(y, users):
    """Identical to iter26/train.py (itself identical to iter3/iter9/.../iter24's
    train.py) -- copied verbatim."""
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
    """Identical to iter26/train.py -- copied verbatim."""
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


def deepfm_bpr_step(m, Xpos, Xneg):
    """DeepFM analogue of iter19/iter24's bpr_step. Copied VERBATIM from
    iter26/train.py's deepfm_bpr_step -- the FM-part gradient block (gV/gW/b
    update) is byte-for-byte the same code as iter19/iter24's bpr_step. The
    only addition is the deep part's own forward/backward/Adam step, whose
    gradient does NOT feed back into V/W (see model.py docstring)."""
    B = len(Xpos)
    zpos_fm, Epos, Spos = m.logits(Xpos)
    zneg_fm, Eneg, Sneg = m.logits(Xneg)
    zpos_deep, in_pos, pre_pos = m.deep_forward(Epos)
    zneg_deep, in_neg, pre_neg = m.deep_forward(Eneg)
    zpos = zpos_fm + zpos_deep
    zneg = zneg_fm + zneg_deep
    d = zpos - zneg
    gpos = ((sigmoid(d) - 1) / B).astype(np.float32)
    gneg = -gpos

    X = np.concatenate([Xpos, Xneg], axis=0)
    E = np.concatenate([Epos, Eneg], axis=0)
    S = np.concatenate([Spos, Sneg], axis=0)
    g = np.concatenate([gpos, gneg], axis=0)

    # ---- FM-part gradient: identical formula/code to baseline.FM / iter19 / iter24 ----
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

    # ---- deep-part gradient (own Adam parameter group) ----
    if m.use_deep:
        inputs = [np.concatenate([ip, iN], axis=0) for ip, iN in zip(in_pos, in_neg)]
        preacts = [np.concatenate([pp, pn], axis=0) for pp, pn in zip(pre_pos, pre_neg)]
        gWd, gbd, _dE_discarded = m.deep_backward(g, inputs, preacts)
        m.mlp_adam_step(gWd, gbd)

    return float(-np.mean(np.log(sigmoid(d) + 1e-9)))


def run_deepfm_bpr(data_dir, feature_set=DEFAULT_FEATURES, halflives=HALFLIVES,
                    tab_halflives=TAB_HALFLIVES,
                    k=16, hidden=(32,), use_deep=True, lr=0.001, mlp_lr=None, l2_mlp=1e-5,
                    epochs=40, bs=8192, patience=4, seed=0, verbose=True,
                    K=5, splits_cache=None):
    splits = splits_cache if splits_cache is not None else \
        load_ext(data_dir, halflives=halflives, tab_halflives=tab_halflives, K=K)
    enc, dim = encode_ext(splits, feature_set=feature_set, halflives=halflives, tab_halflives=tab_halflives)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    num_fields = Xtr.shape[1]

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs)))

    user_cumw = np.cumsum(pos_len.astype(np.float64))
    user_totalw = user_cumw[-1]

    if verbose:
        fields = list(BASE_FIELDS) + list(feature_set)
        print(f"  fields={fields} dim={dim} num_fields={num_fields} hidden={hidden} use_deep={use_deep}")
        print(f"  BPR-eligible train users: {n_users} | {steps_per_epoch} steps/epoch")

    m = DeepFM(dim, num_fields=num_fields, k=k, hidden=hidden, lr=lr, mlp_lr=mlp_lr,
               l2_mlp=l2_mlp, use_deep=use_deep, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    diverged = False
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(steps_per_epoch):
            Xpos_rows, Xneg_rows = sample_pairs(
                rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                user_cumw=user_cumw, user_totalw=user_totalw)
            losses.append(deepfm_bpr_step(m, Xtr[Xpos_rows], Xtr[Xneg_rows]))
        mean_loss = float(np.mean(losses))
        if not np.isfinite(mean_loss):
            diverged = True
            if verbose: print(f"  epoch {ep:2d} | loss NaN/Inf -- diverged, stopping")
            break
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {mean_loss:.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b),
                           [w.copy() for w in m.mlp_W], [b.copy() for b in m.mlp_b])
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    if diverged and best_state is None:
        return {'valid': {'GAUC': 0.0, 'nDCG@5': 0.0, 'primary': 0.0},
                'test': {'GAUC': 0.0, 'nDCG@5': 0.0, 'primary': 0.0}, 'diverged': True}
    m.V, m.W, m.b, m.mlp_W, m.mlp_b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte)), 'diverged': diverged}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='../../KuaiRand-Pure/data')
    ap.add_argument('--features', default=','.join(DEFAULT_FEATURES))
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--hidden', default='32', help="comma-separated hidden widths, e.g. '32' or '32,16'")
    ap.add_argument('--no_deep', action='store_true')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--mlp_lr', type=float, default=None)
    ap.add_argument('--l2_mlp', type=float, default=1e-5)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    feature_set = tuple(f for f in a.features.split(',') if f)
    hidden = tuple(int(x) for x in a.hidden.split(',') if x)
    print(f"loading {a.data_dir} ... extra features={feature_set} hidden={hidden} use_deep={not a.no_deep}")
    res = run_deepfm_bpr(a.data_dir, feature_set=feature_set, k=a.k, hidden=hidden,
                          use_deep=not a.no_deep, lr=a.lr, mlp_lr=a.mlp_lr, l2_mlp=a.l2_mlp,
                          epochs=a.epochs, bs=a.bs, patience=a.patience, seed=a.seed,
                          verbose=not a.quiet, K=a.K)
    print(f"\n=== iter28 deepfm+refined_features hidden={hidden} use_deep={not a.no_deep} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
    def _clean(d):
        return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}
    print(json.dumps({'seed': a.seed, 'features': list(feature_set), 'hidden': list(hidden),
                       'use_deep': not a.no_deep,
                       'valid': _clean(res['valid']), 'test': _clean(res['test'])}))

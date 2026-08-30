"""iter36: multi-task learning v2 -- per-task linear head, sharing ONLY the
FM embedding matrix V, on top of iter27's winning triple-fusion config
(iter24 features + iter23 decay-aware BPR sampling + iter25 formula
constants). This is the fix iter31 diagnosed as missing: iter31 shared the
LITERAL raw score z between the rank-invariant BPR loss and 5 base-rate-
calibrated pointwise BCE losses, and regressed monotonically at every
nonzero weight (worse the higher the weight) because the shared score can't
simultaneously satisfy both objectives' different scales/calibration.

Design: each auxiliary task t gets its OWN linear weight row Waux[t] (shape
(dim,)) and bias baux[t], updated by its OWN Adam moments -- so its
calibration is fully decoupled from the main task's b/W. The auxiliary
tasks share ONLY the pairwise FM interaction term `inter = 0.5*((S**2).sum
- (E**2).sum)`, which is built entirely from the shared embedding matrix V
(same E, S already computed for the main BPR forward pass on Xpos/Xneg --
no extra forward pass needed). Backward: each aux task's per-row BCE
gradient backprops into V through that shared `inter` term (same (S-E)
factor as the main task, scaled and averaged over tasks, then summed with
the main task's gradient before the ONE shared V update -- still a single
joint Adam step per training iteration), but backprops into its OWN
Waux[t]/baux[t] only, NOT into the main W/b.

Everything else (build_pos_neg_index/sample_pairs/bpr_step/run_bpr_ext's
epoch loop/sampling_mode='decay' BPR user-sampling weight/alpha,n_buckets
formula constants) is an unmodified, line-for-line copy of
iter27_triple_fusion/train.py. bpr_step is kept verbatim (used whenever
aux_weight=0, for the harness-fidelity check).

Leakage argument: identical to iter31's (aux_pos/aux_neg are read from
load_aux_labels(...)['train'], indexed by the SAME Xpos_rows/Xneg_rows
already sampled for the main BPR pair -- indices exclusively into
enc['train'], never into valid/test; the aux arrays are never concatenated
into any feature matrix and never read anywhere in evaluate()'s call path).
"""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate           # noqa: E402  official eval, unmodified
from baseline import FM, sigmoid        # noqa: E402  same FM class as iter3/iter9

from data_ext import (load_ext, encode_ext, compute_final_decayed_pos,     # noqa: E402
                       load_aux_labels, AUX_LABELS,
                       BASE_FIELDS, HALFLIVES, TAB_HALFLIVES, ALPHA)


def build_pos_neg_index(y, users):
    """Identical to iter27's train.py -- copied verbatim."""
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
    """Identical to iter27's train.py -- copied verbatim."""
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
    """Identical to iter27's train.py -- copied verbatim, UNCHANGED. Used
    whenever aux_weight=0 for the bit-exact harness-fidelity check."""
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
    return float(-np.mean(np.log(sigmoid(d) + 1e-9))), 0.0


def init_aux_heads(n_aux, dim):
    Waux = np.zeros((n_aux, dim), dtype=np.float32)
    baux = np.zeros(n_aux, dtype=np.float32)
    mWaux = np.zeros_like(Waux); vWaux = np.zeros_like(Waux)
    mbaux = np.zeros_like(baux); vbaux = np.zeros_like(baux)
    return [Waux, baux, mWaux, vWaux, mbaux, vbaux]


def mtl2_step(m, aux_state, Xpos, Xneg, aux_pos, aux_neg, aux_weight):
    """Per-task-linear-head multi-task step (see module docstring). aux_pos/
    aux_neg: (B, n_aux) float32 arrays, the auxiliary labels for the same
    Xpos/Xneg rows sampled for the main BPR pair."""
    Waux, baux, mWaux, vWaux, mbaux, vbaux = aux_state
    B, n_aux = len(Xpos), Waux.shape[0]
    zpos, Epos, Spos = m.logits(Xpos)
    zneg, Eneg, Sneg = m.logits(Xneg)
    d = zpos - zneg
    gpos_bpr = ((sigmoid(d) - 1) / B).astype(np.float32)
    gneg_bpr = -gpos_bpr

    # shared pairwise interaction term (same E,S as the main forward pass --
    # no extra forward pass needed), used by every aux task's own head.
    inter_pos = 0.5 * ((Spos ** 2).sum(1) - (Epos ** 2).sum((1, 2)))
    inter_neg = 0.5 * ((Sneg ** 2).sum(1) - (Eneg ** 2).sum((1, 2)))
    zaux_pos = baux[None, :] + Waux[:, Xpos].sum(2).T + inter_pos[:, None]   # (B, n_aux)
    zaux_neg = baux[None, :] + Waux[:, Xneg].sum(2).T + inter_neg[:, None]
    saux_pos = sigmoid(zaux_pos); saux_neg = sigmoid(zaux_neg)

    gaux_pos = aux_weight * (saux_pos - aux_pos) / B    # (B, n_aux)
    gaux_neg = aux_weight * (saux_neg - aux_neg) / B

    # NOTE: divide by n_aux here so gWaux/gbaux are the true gradient of
    # L_aux = (1/n_aux)*sum_t mean_rows(BCE_t) w.r.t. that task's OWN head
    # -- verified by finite-difference check (scratchpad/grad_check_iter36.py,
    # max abs err ~1e-10) before this was trusted; an earlier version without
    # this factor was off by up to 0.19 against the numerical gradient.
    gWaux = np.zeros_like(Waux); gbaux = np.zeros_like(baux)
    for t in range(n_aux):
        np.add.at(gWaux[t], Xpos, gaux_pos[:, t:t + 1] / n_aux)
        np.add.at(gWaux[t], Xneg, gaux_neg[:, t:t + 1] / n_aux)
    gbaux = (gaux_pos.sum(0) + gaux_neg.sum(0)) / n_aux

    # V's gradient = main-task contribution + mean-over-aux-tasks contribution
    # (mean so aux_weight's scale doesn't depend on n_aux, matching iter31's
    # convention); W/b get ONLY the main-task contribution -- this is the
    # only structural difference from bpr_step/iter31's shared-score design.
    gpos_v = gpos_bpr + gaux_pos.mean(axis=1)
    gneg_v = gneg_bpr + gaux_neg.mean(axis=1)

    X = np.concatenate([Xpos, Xneg], axis=0)
    E = np.concatenate([Epos, Eneg], axis=0)
    S = np.concatenate([Spos, Sneg], axis=0)
    g_v = np.concatenate([gpos_v, gneg_v], axis=0)
    g_w = np.concatenate([gpos_bpr, gneg_bpr], axis=0)

    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, X, g_w[:, None])
    np.add.at(gV, X, g_v[:, None, None] * (S[:, None, :] - E))
    gV += m.l2 * m.V; gW += m.l2 * m.W
    gWaux += m.l2 * Waux

    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW),
                        (Waux, gWaux, mWaux, vWaux), (baux, gbaux, mbaux, vbaux)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.b -= m.lr * g_w.sum()

    bpr_loss = float(-np.mean(np.log(sigmoid(d) + 1e-9)))
    eps_l = 1e-9
    aux_bce = -(aux_pos * np.log(saux_pos + eps_l) + (1 - aux_pos) * np.log(1 - saux_pos + eps_l))
    aux_loss = float(np.mean(aux_bce))
    return bpr_loss, aux_loss


def run_bpr_ext(data_dir, feature_set=('decay_rate_2.5', 'decay_act_2.5', 'decay_tab_3',
                                        'last1', 'lastk_rate', 'gap'),
                 halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
                 k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
                 steps_mult=1, K=5, splits_cache=None,
                 sampling_mode='flat', sampling_alpha=1.0, decay_halflife=3,
                 alpha=ALPHA, n_buckets=10,
                 aux_weight=0.0, aux_tasks=AUX_LABELS, aux_cache=None):
    """Identical signature/body to iter27's run_bpr_ext, plus aux_weight/
    aux_tasks/aux_cache (iter31's multi-task convention: aux_weight=0.0 uses
    bpr_step unmodified for a bit-exact harness-fidelity check)."""
    splits = splits_cache if splits_cache is not None else \
        load_ext(data_dir, halflives=halflives, tab_halflives=tab_halflives, K=K)
    enc, dim = encode_ext(splits, feature_set=feature_set, halflives=halflives, tab_halflives=tab_halflives,
                           alpha=alpha, n_buckets=n_buckets)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs))) * steps_mult

    if sampling_mode == 'flat':
        weights = pos_len.astype(np.float64) ** sampling_alpha
    elif sampling_mode == 'decay':
        decayed_pos_dict = compute_final_decayed_pos(splits['train'], halflife=decay_halflife)
        decayed_arr = np.array([decayed_pos_dict.get(u, 0.0) for u in eligible], dtype=np.float64)
        weights = decayed_arr ** sampling_alpha
    else:
        raise ValueError(f"unknown sampling_mode: {sampling_mode!r}")
    user_cumw = np.cumsum(weights)
    user_totalw = user_cumw[-1]

    use_mtl = aux_weight != 0.0
    aux_mat, aux_state = None, None
    if use_mtl:
        aux = aux_cache if aux_cache is not None else load_aux_labels(data_dir)
        aux_train = aux['train']
        assert len(aux_train[AUX_LABELS[0]]) == len(ytr), \
            "aux label count must match train row count (see data_ext.load_aux_labels)"
        aux_mat = np.stack([aux_train[t] for t in aux_tasks], axis=1).astype(np.float32)
        aux_state = init_aux_heads(len(aux_tasks), dim)

    if verbose:
        fields = list(BASE_FIELDS) + list(feature_set)
        print(f"  fields={fields} dim={dim}")
        print(f"  BPR-eligible train users: {n_users} (>=1 pos & >=1 neg) | "
              f"{steps_per_epoch} steps/epoch | sampling_mode={sampling_mode} "
              f"sampling_alpha={sampling_alpha}"
              + (f" decay_halflife={decay_halflife}" if sampling_mode == 'decay' else '')
              + f" | alpha(Laplace)={alpha} n_buckets={n_buckets}")
        if use_mtl:
            print(f"  multi-task v2 (per-task linear head): aux_weight={aux_weight} aux_tasks={list(aux_tasks)}")

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        bpr_losses, aux_losses = [], []
        for _ in range(steps_per_epoch):
            Xpos_rows, Xneg_rows = sample_pairs(
                rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                user_cumw=user_cumw, user_totalw=user_totalw)
            if use_mtl:
                bl, al = mtl2_step(m, aux_state, Xtr[Xpos_rows], Xtr[Xneg_rows],
                                    aux_mat[Xpos_rows], aux_mat[Xneg_rows], aux_weight)
            else:
                bl, al = bpr_step(m, Xtr[Xpos_rows], Xtr[Xneg_rows])
            bpr_losses.append(bl); aux_losses.append(al)
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            extra = f" | aux_bce {np.mean(aux_losses):.4f}" if use_mtl else ""
            print(f"  epoch {ep:2d} | bpr_loss {np.mean(bpr_losses):.4f}{extra} | valid GAUC {va['GAUC']:.4f} "
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
    ap.add_argument('--features', default='decay_rate_2.5,decay_act_2.5,decay_tab_3,last1,lastk_rate,gap')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps_mult', type=int, default=1)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--sampling_mode', default='decay', choices=['flat', 'decay'])
    ap.add_argument('--sampling_alpha', type=float, default=0.75)
    ap.add_argument('--decay_halflife', type=float, default=3)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--n_buckets', type=int, default=20)
    ap.add_argument('--aux_weight', type=float, default=0.0)
    ap.add_argument('--aux_tasks', default=','.join(AUX_LABELS))
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    feature_set = tuple(f for f in a.features.split(',') if f)
    aux_tasks = tuple(f for f in a.aux_tasks.split(',') if f)
    print(f"loading {a.data_dir} ... extra features={feature_set} aux_weight={a.aux_weight}")
    res = run_bpr_ext(a.data_dir, feature_set=feature_set, k=a.k, lr=a.lr, epochs=a.epochs,
                       bs=a.bs, patience=a.patience, seed=a.seed, verbose=not a.quiet,
                       steps_mult=a.steps_mult, K=a.K,
                       sampling_mode=a.sampling_mode, sampling_alpha=a.sampling_alpha,
                       decay_halflife=a.decay_halflife, alpha=a.alpha, n_buckets=a.n_buckets,
                       aux_weight=a.aux_weight, aux_tasks=aux_tasks)
    print(f"\n=== iter36 fm+bpr_{a.sampling_mode}sampling+{'+'.join(feature_set)}+mtl2(w={a.aux_weight}) "
          f"(seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
    def _clean(d):
        return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}
    print(json.dumps({'seed': a.seed, 'features': list(feature_set), 'aux_weight': a.aux_weight,
                       'valid': _clean(res['valid']), 'test': _clean(res['test'])}))

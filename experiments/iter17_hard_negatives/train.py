"""iter17: FM + activity-weighted BPR + iter9's causal history features
{activity, tab_pos, rate}, testing INFORMED / HARD negative-video sampling
instead of the negative-sampling scheme used unchanged since iter2/iter3/iter9.

--- What "negative" actually meant before this iteration (read from the code,
not assumed) ---
iter2/iter3/iter7/iter9 all build, per user u, two lists of TRAIN ROW INDICES:
  by_user_pos[u] = rows where user==u and label==1
  by_user_neg[u] = rows where user==u and label==0
Each BPR step samples a user (uniform in iter2, activity/pos_len-weighted in
iter3+), then samples ONE ROW uniformly from that SAME user's own pos list
and ONE ROW uniformly from that SAME user's own neg list. So the "negative"
has NEVER been "a uniform random video from the full vocabulary" as this
iteration's brief assumed -- it has always been "a video this exact user was
actually exposed to and did NOT long-view", drawn uniformly from that user's
own negative history. This is already a mildly "hard"/relevant negative (a
real impression, not an arbitrary unrelated video) but it is uninformed by
tab or popularity, and for users with few negatives the pool is small/noisy.
This module's 'uniform' mode reproduces that exact mechanism unchanged, to
serve as this run's own-harness parity baseline. The other three modes keep
user-side context (user_id + causal features, i.e. the picked user's state
"as of" the sampled positive row) but replace the ITEM-side columns
(video_id, author_id, tab, dur_bucket) with a video candidate drawn from a
richer, precomputed, GLOBAL pool -- not restricted to this user's own history.

--- Negative-sampling modes ---
'uniform'     : baseline, see above -- unchanged iter3/iter9 mechanism.
'same_tab'    : negative video sampled UNIFORMLY among the unique videos
                that were shown (to any user, anywhere in train) under the
                SAME `tab` as the sampled positive row. One representative
                row per unique video is precomputed per tab so every unique
                video has equal selection probability (not popularity
                weighted). Falls back to a global uniform-over-all-unique-
                videos pool when a tab's candidate pool is smaller than
                `min_tab_pool` unique videos (rare; see RESULT.md for the
                measured fallback rate).
'pop_weighted': negative video sampled with probability proportional to its
                exposure count in train. Implemented with ZERO extra
                bookkeeping: sampling a TRAIN ROW uniformly at random makes
                the row's video selection probability exactly proportional
                to that video's row-count (= its popularity), because a
                video with N rows is N times more likely to be hit by a
                uniform row draw. So this mode is just `rng.integers(0,
                n_train)`.
'same_tab_pop_weighted' (optional 4th variant): same idea restricted to the
                positive row's tab -- sample a train ROW uniformly among all
                rows sharing that tab (not deduplicated to unique videos),
                which makes video selection popularity-weighted *within*
                the tab. Same fallback mechanism as 'same_tab'.

For all three non-uniform modes, the negative example's user-side columns
(user_id + activity/tab_pos/rate causal-feature columns) are copied from the
POSITIVE row, i.e. "this same user, at this same point in time, paired with
a different candidate video" -- only the item-side columns [video_id,
author_id, tab, dur_bucket] are swapped in from the candidate row. NOTE this
means the `tab_pos` causal feature (which is itself defined per (user,tab)
pair) is carried over from the positive row's own tab even when the
candidate's tab differs from the positive's (this only happens in
'pop_weighted' mode, since both same_tab modes match tabs by construction) --
a known, documented approximation rather than a full per-(user,tab,date)
re-derivation, made for efficiency; see RESULT.md.

Everything else (FM, Adam, activity-weighted USER sampling, epochs/patience/
lr/k/bs) is a line-for-line copy of experiments/iter9_history_dense/train.py.
"""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from evaluate import evaluate           # noqa: E402  official eval, unmodified
from baseline import FM, sigmoid        # noqa: E402  same FM class as iter9/iter3

from data_ext import load_ext, encode_ext, BASE_FIELDS  # noqa: E402

# column layout: BASE_FIELDS (user_id,video_id,author_id,tab,dur_bucket) + feature_set
ITEM_COLS = [1, 2, 3, 4]   # video_id, author_id, tab, dur_bucket
TAB_COL = 3


def build_pos_neg_index(y, users):
    """Identical to iter3/iter9's train.py -- copied verbatim."""
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


def build_negative_pools(Xtr, min_tab_pool=20):
    """Precompute (ONCE, from train split) the candidate row-index pools used
    by the 'same_tab', 'pop_weighted', 'same_tab_pop_weighted' modes.
    Returns a dict with:
      n_train              : int
      tab_rows[tab_val]    : np.array of ALL row idx with that tab (for
                              same_tab_pop_weighted + tab-pool-size check)
      tab_uniq_rep[tab_val]: np.array of one representative row idx per
                              UNIQUE video shown under that tab (for same_tab)
      global_uniq_rep      : np.array of one representative row idx per
                              UNIQUE video globally (fallback pool)
    """
    n_train = Xtr.shape[0]
    video_col = Xtr[:, 1]
    tab_col = Xtr[:, TAB_COL]

    tab_rows = {}
    tab_uniq_rep = {}
    for t in np.unique(tab_col):
        rows_t = np.nonzero(tab_col == t)[0]
        tab_rows[int(t)] = rows_t
        vids_t = video_col[rows_t]
        _, first_local = np.unique(vids_t, return_index=True)
        tab_uniq_rep[int(t)] = rows_t[first_local]

    _, first_global = np.unique(video_col, return_index=True)
    global_uniq_rep = first_global.astype(np.int64)

    return {'n_train': n_train, 'tab_rows': tab_rows, 'tab_uniq_rep': tab_uniq_rep,
            'global_uniq_rep': global_uniq_rep, 'min_tab_pool': min_tab_pool}


def sample_pairs(rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                  user_cumw=None, user_totalw=None):
    """Identical to iter3/iter9's train.py -- copied verbatim. Returns
    (pos_rows, neg_rows_uniform) -- the SECOND is only actually used when
    negative_mode=='uniform'; other modes discard it and resample negatives
    via sample_negatives() below (they still need pos_rows)."""
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


def sample_negatives(mode, rng, pos_rows, Xtr, pools, neg_rows_uniform, fb_counter):
    """Returns Xneg (B,F) built according to `mode`. `fb_counter` is a
    length-2 mutable list [fallback_events, total_tab_draws] updated in
    place for same_tab* modes (used to report the fallback rate)."""
    if mode == 'uniform':
        return Xtr[neg_rows_uniform]

    Xneg = Xtr[pos_rows].copy()          # user-side cols correct by construction
    B = len(pos_rows)

    if mode == 'pop_weighted':
        cand_rows = rng.integers(0, pools['n_train'], size=B)
        Xneg[:, ITEM_COLS] = Xtr[cand_rows][:, ITEM_COLS]
        return Xneg

    if mode in ('same_tab', 'same_tab_pop_weighted'):
        pos_tabs = Xtr[pos_rows, TAB_COL]
        cand_rows = np.empty(B, dtype=np.int64)
        pool_dict = pools['tab_rows'] if mode == 'same_tab_pop_weighted' else pools['tab_uniq_rep']
        min_pool = pools['min_tab_pool']
        for t in np.unique(pos_tabs):
            mask = pos_tabs == t
            cnt = int(mask.sum())
            fb_counter[1] += cnt
            pool = pool_dict.get(int(t))
            if pool is None or len(pool) < min_pool:
                fb_counter[0] += cnt
                pool = pools['global_uniq_rep']
            idx = rng.integers(0, len(pool), size=cnt)
            cand_rows[mask] = pool[idx]
        Xneg[:, ITEM_COLS] = Xtr[cand_rows][:, ITEM_COLS]
        return Xneg

    raise ValueError(mode)


def bpr_step(m, Xpos, Xneg):
    """Identical to iter3/iter9's train.py -- copied verbatim."""
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


def run_bpr_hardneg(data_dir, feature_set=('activity', 'tab', 'rate'), negative_mode='uniform',
                     k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
                     min_tab_pool=20, steps_mult=1):
    splits = load_ext(data_dir)
    enc, dim = encode_ext(splits, feature_set=feature_set)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs))) * steps_mult

    user_cumw = np.cumsum(pos_len.astype(np.float64))
    user_totalw = user_cumw[-1]

    pools = build_negative_pools(Xtr, min_tab_pool=min_tab_pool) if negative_mode != 'uniform' else None
    fb_counter = [0, 0]   # [fallback_events, total_tab_draws]

    if verbose:
        fields = list(BASE_FIELDS) + list(feature_set)
        print(f"  fields={fields} dim={dim} negative_mode={negative_mode}")
        print(f"  BPR-eligible train users: {n_users} (>=1 pos & >=1 neg) | "
              f"{steps_per_epoch} steps/epoch | activity-weighted user sampling")

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(steps_per_epoch):
            pos_rows, neg_rows_uniform = sample_pairs(
                rng, n_users, bs, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len,
                user_cumw=user_cumw, user_totalw=user_totalw)
            Xpos = Xtr[pos_rows]
            Xneg = sample_negatives(negative_mode, rng, pos_rows, Xtr, pools, neg_rows_uniform, fb_counter)
            losses.append(bpr_step(m, Xpos, Xneg))
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
    fallback_rate = (fb_counter[0] / fb_counter[1]) if fb_counter[1] > 0 else None
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte)),
            'fallback_rate': fallback_rate}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='../../KuaiRand-Pure/data')
    ap.add_argument('--features', default='activity,tab,rate',
                     help="comma-separated subset/order of {activity,tab,rate}")
    ap.add_argument('--negative_mode', default='uniform',
                     choices=['uniform', 'same_tab', 'pop_weighted', 'same_tab_pop_weighted'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min_tab_pool', type=int, default=20)
    ap.add_argument('--steps_mult', type=int, default=1)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    feature_set = tuple(f for f in a.features.split(',') if f)
    print(f"loading {a.data_dir} ... extra features={feature_set} negative_mode={a.negative_mode}")
    res = run_bpr_hardneg(a.data_dir, feature_set=feature_set, negative_mode=a.negative_mode,
                           k=a.k, lr=a.lr, epochs=a.epochs, bs=a.bs, patience=a.patience,
                           seed=a.seed, verbose=not a.quiet, min_tab_pool=a.min_tab_pool,
                           steps_mult=a.steps_mult)
    print(f"\n=== iter17 fm+bpr+{'+'.join(feature_set)}+neg={a.negative_mode} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
    if res['fallback_rate'] is not None:
        print(f"  same-tab fallback rate: {res['fallback_rate']*100:.3f}%")

    def _clean(d):
        return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}
    print(json.dumps({'seed': a.seed, 'features': list(feature_set), 'negative_mode': a.negative_mode,
                       'valid': _clean(res['valid']), 'test': _clean(res['test']),
                       'fallback_rate': res['fallback_rate']}))

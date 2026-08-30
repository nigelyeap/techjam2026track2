"""iter32: LIGHTWEIGHT USER-HISTORY SEQUENCE / TARGET-ATTENTION feature
(scoped-down DIN/SIM-style interest modeling), stacked on top of iter24's
exact standing-best feature set (decay_rate_2.5, decay_act_2.5, decay_tab_3,
last1, lastk_rate, gap). This targets the README's #2 unexplored-headroom
item and the ledger's Round 9 roadmap item "lightweight sequence/attention
modeling" -- zero behavioral sequence modeling has been tried across the
prior 30 iterations in this repo; every existing history feature is a
scalar aggregate (count/rate/decay), never conditioned on WHICH item is
being scored.

Core idea (DIN's central mechanism, simplified to fit a one-session,
numpy-only, no-autodiff scope):
  for each row (a candidate video_id being scored for a user at time t),
  look at that SAME user's PAST interactions strictly before t (same
  causality discipline as iter18/iter19/iter24's momentum: time_ms-level
  total order, orig_idx tiebreak, capped history window), compute a
  compatibility score between the candidate item's embedding and each past
  item's embedding, softmax-normalize over the window, and pool the past
  interactions' LABELS with those attention weights into a single scalar
  "attention-weighted interest score" (`attn_rate_W`) -- a target-attention
  analogue of iter18's `lastk_rate` (which pools uniformly, blind to item
  identity) and iter16's `decay_rate` (which pools by time alone, also
  blind to item identity). A second family, `attn_decay_rate_H`, is the
  dispatch-prompt's suggested fallback: same pooling but with an added
  recency-decay term in the attention logits (weight = decay(gap) *
  dot(candidate_emb, past_item_emb)), in case pure similarity-only
  attention proves too noisy on a dataset this size.

Item embeddings are NOT differentiable/jointly-trained end-to-end (that
would need a real autodiff attention layer -- explicitly out of scope per
the dispatch prompt's "do not build a full transformer" and "keep scope
realistic" guidance). Instead they are PRE-TRAINED ONCE via a small
user_id x video_id matrix-factorization FM (`pretrain_item_embeddings`,
k=8, a handful of pointwise-logloss epochs), fit on the TRAIN split ONLY
(same causal boundary as how the main FM's own video_id/author_id
embeddings are fit -- see RESULT.md's "Causality boundary" section for the
explicit scope decision this implies: item embeddings are a fixed,
non-causal-per-row structural representation, exactly like the main model's
raw ID embeddings; only the PER-ROW HISTORY RETRIEVAL that the attention
mechanism reads is causally constrained, and that is what is brute-force
verified below).

Architecture: FOUR independent causal traversals over the same flat
per-row data, joined by row index (iter19/iter24's established pattern,
kept here to avoid cross-family contamination):
  1. compute_causal_features    -- iter9/iter16/iter20's flat date-grouped
                                    activity/tab_pos/rate. Copied verbatim
                                    from iter24's data_ext.py.
  2. compute_decay_features     -- iter16/iter20's exponential-decay
                                    rate/act (fine HALFLIVES grid). Copied
                                    verbatim from iter24's data_ext.py.
  3. compute_decay_tab_features -- iter20's decayed tab_pos. Copied
                                    verbatim from iter24's data_ext.py.
  4. compute_momentum_features  -- iter18's time_ms-level last1/lastk/gap.
                                    Imported via importlib, exactly as
                                    iter19/iter24 did.
  5. compute_attention_features -- THIS ITERATION's new target-attention
                                    traversal (see docstring below).

Row tuple layout (see IDX / _halflife_col / _tab_halflife_col / _attn_col
for the authoritative index map):
  0 date, 1 user_id, 2 video_id, 3 author_id, 4 tab, 5 duration_ms, 6 label,
  7 hourmin, 8 time_ms, 9 orig_idx,
  10 activity, 11 tab_pos, 12 prior_pos, 13 prior_total,        <- flat (iter9)
  14 last1, 15 lastk_sum, 16 lastk_cnt, 17 gap_ms,               <- momentum (iter18)
  18.. decay_pos_h0, decay_total_h0, decay_pos_h1, decay_total_h1, ...  <- decay (iter16/iter20 fine grid)
       (2*len(HALFLIVES) columns, h order follows HALFLIVES)
  then decay_tab_h0, decay_tab_h1, ...                           <- decayed tab_pos (iter20)
       (len(TAB_HALFLIVES) columns, h order follows TAB_HALFLIVES)
  then attn_rate_w0, attn_rate_w1, ...                           <- target attention (iter32 NEW)
       (len(WINDOWS) columns, order follows WINDOWS)
  then attn_decay_rate_h0, attn_decay_rate_h1, ...               <- decay-weighted attention (iter32 NEW)
       (len(ATTN_DECAY_HALFLIVES) columns, order follows ATTN_DECAY_HALFLIVES)

Does NOT modify data.py, baseline.py, or any iterN/data_ext.py file it
reuses/imports.
"""
import os, sys, csv, collections, datetime, importlib.util, pickle
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from data import SPLITS  # noqa: E402  (date ranges only, kept in sync w/ data.py)
from baseline import FM, sigmoid  # noqa: E402  reused unmodified for item-embedding pretraining

LABEL = 'long_view'
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
ALPHA = 1.0
K_DEFAULT = 5

# Axis A grid: iter24's winning fine-grid halflife (kept fixed here -- iter24's
# feature set is the base we stack attention on top of, not re-swept).
HALFLIVES = [2, 2.5, 3, 3.5]  # days
# Axis B grid: candidate halflives for decayed tab_pos (iter20's grid).
TAB_HALFLIVES = [3, 7]  # days

# --- iter32 NEW: target-attention hyperparameters ---
K_EMB = 8            # item-embedding dimension (pretrained, fixed; not swept --
                      # README/iter7/iter14/iter25 all found extra capacity
                      # doesn't help this dataset, so a small k is a defensible
                      # scope decision, not a shortcut).
EMB_EPOCHS = 8        # pretrain epochs (pointwise logloss FM, train split only)
EMB_LR = 0.005
EMB_SEED = 0          # fixed regardless of the main model's seed -- this is a
                      # shared, cached, precomputed artifact like load_ext's cache.
WINDOWS = (10, 20, 40)            # history-window length sweep (rows, capped)
ATTN_DECAY_HALFLIVES = (3.0, 7.0)  # days, fallback recency-decayed-similarity variant


def _load_module(name, rel_path):
    path = os.path.join(_THIS_DIR, *rel_path.split('/'))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# iter18's momentum function is imported (not copied) -- same pattern iter19
# used. Loaded under a distinct module name since every iterN dir has its own
# data_ext.py (a plain `import data_ext` would collide).
_iter18_de = _load_module('iter18_data_ext', '../iter18_momentum/data_ext.py')
compute_momentum_features = _iter18_de.compute_momentum_features


def _date_to_ordinal(d):
    y, m, day = d // 10000, (d // 100) % 100, d % 100
    return datetime.date(y, m, day).toordinal()


def compute_causal_features(rows):
    """Identical to iter9/iter16/iter20's compute_causal_features -- copied
    verbatim (flat activity/tab_pos/prior_pos/prior_total)."""
    n = len(rows)
    order = sorted(range(n), key=lambda i: rows[i][0])
    activity = [0] * n
    tab_pos = [0] * n
    prior_pos = [0] * n

    user_total = collections.defaultdict(int)
    user_pos = collections.defaultdict(int)
    user_tab_pos = collections.defaultdict(int)

    i = 0
    while i < n:
        j = i
        d = rows[order[i]][0]
        while j < n and rows[order[j]][0] == d:
            j += 1
        day_idx = order[i:j]
        for idx in day_idx:
            r = rows[idx]
            u, tab = r[1], r[4]
            activity[idx] = user_total[u]
            prior_pos[idx] = user_pos[u]
            tab_pos[idx] = user_tab_pos[(u, tab)]
        for idx in day_idx:
            r = rows[idx]
            u, tab, label = r[1], r[4], r[6]
            user_total[u] += 1
            if label == 1:
                user_pos[u] += 1
                user_tab_pos[(u, tab)] += 1
        i = j
    return {'activity': activity, 'tab_pos': tab_pos, 'prior_pos': prior_pos,
            'prior_total': activity}


def compute_decay_features(rows, halflives=HALFLIVES):
    """Identical mechanism to iter16/iter20's compute_decay_features --
    copied verbatim (lazy-decay running state per user, exact not
    approximate). Returns (decayed_pos, decayed_total): each an (n, H)
    float64 array."""
    n = len(rows)
    H = len(halflives)
    order = sorted(range(n), key=lambda i: rows[i][0])
    day_mult = [0.5 ** (1.0 / h) for h in halflives]

    decayed_pos = np.zeros((n, H), dtype=np.float64)
    decayed_total = np.zeros((n, H), dtype=np.float64)

    ord_cache = {}
    def ordf(d):
        v = ord_cache.get(d)
        if v is None:
            v = _date_to_ordinal(d)
            ord_cache[d] = v
        return v

    user_last_ord = {}
    user_pos_state = {}
    user_total_state = {}

    i = 0
    while i < n:
        j = i
        d = rows[order[i]][0]
        while j < n and rows[order[j]][0] == d:
            j += 1
        day_idx = order[i:j]
        d_ord = ordf(d)

        for idx in day_idx:
            r = rows[idx]
            u = r[1]
            last = user_last_ord.get(u)
            if last is not None:
                gap = d_ord - last
                pstate = user_pos_state[u]
                tstate = user_total_state[u]
                for h in range(H):
                    f = day_mult[h] ** gap
                    decayed_pos[idx, h] = pstate[h] * f
                    decayed_total[idx, h] = tstate[h] * f

        day_pos_count = collections.defaultdict(int)
        day_total_count = collections.defaultdict(int)
        for idx in day_idx:
            r = rows[idx]
            u, label = r[1], r[6]
            day_total_count[u] += 1
            if label == 1:
                day_pos_count[u] += 1
        touched_users = set(day_total_count.keys())
        for u in touched_users:
            last = user_last_ord.get(u)
            if last is not None:
                gap = d_ord - last
                pstate = user_pos_state[u]
                tstate = user_total_state[u]
                new_p = [pstate[h] * (day_mult[h] ** gap) for h in range(H)]
                new_t = [tstate[h] * (day_mult[h] ** gap) for h in range(H)]
            else:
                new_p = [0.0] * H
                new_t = [0.0] * H
            pc = day_pos_count.get(u, 0)
            tc = day_total_count[u]
            for h in range(H):
                new_p[h] += pc
                new_t[h] += tc
            user_pos_state[u] = new_p
            user_total_state[u] = new_t
            user_last_ord[u] = d_ord
        i = j
    return decayed_pos, decayed_total


def compute_decay_tab_features(rows, halflives=TAB_HALFLIVES):
    """Identical to iter20's compute_decay_tab_features -- copied verbatim.
    Decayed analogue of flat `tab_pos` (count of user's prior POSITIVE rows
    in the SAME tab), keyed by (user, tab) instead of user. Two-phase
    date-grouped traversal, same causal guarantee as compute_decay_features:
    same-date rows never see each other, no future leakage.
    Returns decayed_tab_pos: (n, H) float64 array, column order matching
    `halflives`."""
    n = len(rows)
    H = len(halflives)
    order = sorted(range(n), key=lambda i: rows[i][0])
    day_mult = [0.5 ** (1.0 / h) for h in halflives]

    decayed_tab_pos = np.zeros((n, H), dtype=np.float64)

    ord_cache = {}
    def ordf(d):
        v = ord_cache.get(d)
        if v is None:
            v = _date_to_ordinal(d)
            ord_cache[d] = v
        return v

    key_last_ord = {}
    key_state = {}

    i = 0
    while i < n:
        j = i
        d = rows[order[i]][0]
        while j < n and rows[order[j]][0] == d:
            j += 1
        day_idx = order[i:j]
        d_ord = ordf(d)

        for idx in day_idx:
            r = rows[idx]
            u, tab = r[1], r[4]
            key = (u, tab)
            last = key_last_ord.get(key)
            if last is not None:
                gap = d_ord - last
                state = key_state[key]
                for h in range(H):
                    decayed_tab_pos[idx, h] = state[h] * (day_mult[h] ** gap)

        day_pos_count = collections.defaultdict(int)
        for idx in day_idx:
            r = rows[idx]
            u, tab, label = r[1], r[4], r[6]
            if label == 1:
                day_pos_count[(u, tab)] += 1
        for key, pc in day_pos_count.items():
            last = key_last_ord.get(key)
            if last is not None:
                gap = d_ord - last
                state = key_state[key]
                new_s = [state[h] * (day_mult[h] ** gap) for h in range(H)]
            else:
                new_s = [0.0] * H
            for h in range(H):
                new_s[h] += pc
            key_state[key] = new_s
            key_last_ord[key] = d_ord
        i = j
    return decayed_tab_pos


def pretrain_item_embeddings(train_rows, k=K_EMB, epochs=EMB_EPOCHS, lr=EMB_LR, seed=EMB_SEED,
                              bs=8192):
    """iter32 NEW. Fits a tiny 2-field (user_id, video_id) matrix-factorization
    FM via plain pointwise logloss (reuses baseline.FM UNMODIFIED), on the
    TRAIN split ONLY, and returns the learned video_id embedding table.

    CAUSALITY BOUNDARY (explicit, documented, not hidden): this pretraining
    step is NOT causally constrained per-row -- it is a single batch fit over
    the whole train split, exactly analogous to how the main FM's own
    video_id/author_id embeddings are fit (every iteration since iter0 has
    done this; raw ID embeddings have never been treated as needing per-row
    causal ordering in this repo, only aggregated LABEL-derived history
    counters like activity/rate/decay/momentum have). It uses ONLY the train
    split (dates 20220408-20220421) -- it never touches any valid/test label,
    so there is zero risk of the leakage this experiment's verification is
    actually guarding against (a row seeing its own or a future row's label
    via the ATTENTION mechanism's history retrieval, which IS per-row causal
    and IS brute-force verified in __main__ PART E below). The item vectors
    this function returns are a FIXED, non-differentiable lookup table used
    only to compute similarity scores in compute_attention_features -- they
    are never updated again after this call.

    Returns: dict video_id_str -> np.float64 array shape (k,). video_ids
    never seen in train are simply absent (attention code treats missing
    keys as a zero vector, degrading gracefully -- see compute_attention_features).
    """
    uvocab, vvocab = {}, {}
    for r in train_rows:
        u, v = r[1], r[2]
        if u not in uvocab:
            uvocab[u] = len(uvocab)
        if v not in vvocab:
            vvocab[v] = len(vvocab)
    n_u, n_v = len(uvocab), len(vvocab)
    dim = n_u + 1 + n_v + 1  # +1 UNK slot each (unused at fit time, harmless)
    v_offset = n_u + 1

    X = np.empty((len(train_rows), 2), dtype=np.int32)
    y = np.empty(len(train_rows), dtype=np.float32)
    for i, r in enumerate(train_rows):
        X[i, 0] = uvocab[r[1]]
        X[i, 1] = v_offset + vvocab[r[2]]
        y[i] = r[6]

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(len(y))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            m.step(X[b], y[b])

    item_emb = {}
    for vid, pos in vvocab.items():
        item_emb[vid] = m.V[v_offset + pos].astype(np.float64).copy()
    return item_emb


def compute_attention_features(rows, item_emb, k_emb, windows=WINDOWS,
                                decay_halflives=ATTN_DECAY_HALFLIVES):
    """iter32 NEW. Scoped-down DIN-style TARGET ATTENTION over each user's
    causal history, using the fixed pretrained item embeddings above.

    For each row (candidate video at time t, user u): gather u's past
    interactions strictly before t (time_ms-level total order, orig_idx
    tiebreak -- IDENTICAL causal discipline to iter18/iter19/iter24's
    compute_momentum_features, verified there and re-verified independently
    here in __main__ PART E), capped to the most recent max(windows) rows.
    For each window length W in `windows`: take the most recent W of those,
    compute scaled dot-product compatibility between the candidate's
    embedding and each history item's embedding, softmax-normalize over the
    window, and pool the window's LABELS with those weights into a single
    scalar `attn_rate_W` -- the "attention-weighted pooled" interest score.
    For each halflife H in `decay_halflives`: same pooling but over the FULL
    capped history, with a recency-decay term ADDED to the attention logits
    in log-space (logit = scaled_dot + gap_days * ln(0.5^(1/H))), i.e. the
    dispatch prompt's suggested fallback ("weight = decay(gap_time) *
    dot(candidate_emb, past_item_emb)", here combined multiplicatively via
    the softmax rather than as a separate un-normalized weight, since a
    softmax-normalized pooled label needs weights that sum to 1).

    Strict causality by construction (mirrors compute_momentum_features):
    for each user, rows are visited in (time_ms, orig_idx) order; a row's
    attention features are computed from the history deque AS IT STOOD
    BEFORE this row, and the row is appended to the deque only AFTER its own
    features have been read. A row can never see its own label or a later
    row's label, including same-time_ms ties (orig_idx makes the order
    strict and total, identical to iter18).

    Zero-history rows (a user's first interaction in the combined
    train+valid+test timeline): sentinel -1.0 for every attn_rate_W /
    attn_decay_rate_H column (never a fabricated 0.5 or 0.0 -- 'UNK'
    downstream in encode_ext, same convention as `gap`/`last1`).

    Unseen-item degrade-gracefully behavior: if the candidate and/or history
    items are missing from `item_emb` (zero vector), their dot products are
    0 -- softmax over all-zero logits is uniform, so attn_rate_W reduces
    exactly to a plain (unweighted) mean of the window's labels, i.e. it
    never crashes and never silently produces a nonsensical value; it just
    degrades to the same shape of estimator as iter18's `lastk_rate`.

    Returns: dict column_name -> np.ndarray shape (n,), aligned with `rows`.
    """
    n = len(rows)
    w_max = max(windows)
    zero_vec = np.zeros(k_emb, dtype=np.float64)
    scale = 1.0 / np.sqrt(k_emb)
    ln_half = np.log(0.5)
    day_ms = 86400000.0

    out = {}
    for w in windows:
        out[f'attn_rate_{w}'] = np.full(n, -1.0, dtype=np.float64)
    for h in decay_halflives:
        out[f'attn_decay_rate_{h}'] = np.full(n, -1.0, dtype=np.float64)

    by_user = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_user[r[1]].append(i)

    for u, idxs in by_user.items():
        idxs.sort(key=lambda i: (rows[i][8], rows[i][9]))  # (time_ms, orig_idx)
        hist = collections.deque(maxlen=w_max)  # (emb, label, time_ms), chronological
        for i in idxs:
            r = rows[i]
            cand_emb = item_emb.get(r[2], zero_vec)

            if hist:
                h_embs = np.stack([hh[0] for hh in hist])       # (H, k_emb), oldest first
                h_labels = np.array([hh[1] for hh in hist], dtype=np.float64)
                h_times = np.array([hh[2] for hh in hist], dtype=np.float64)
                sims = h_embs.dot(cand_emb) * scale               # (H,)

                for w in windows:
                    sw = sims[-w:]
                    lw = h_labels[-w:]
                    m_ = sw.max()
                    ex = np.exp(sw - m_)
                    wts = ex / ex.sum()
                    out[f'attn_rate_{w}'][i] = float(np.dot(wts, lw))

                cur_t = float(r[8])
                gap_days = (cur_t - h_times) / day_ms
                for h in decay_halflives:
                    logits = sims + gap_days * (ln_half / h)
                    m_ = logits.max()
                    ex = np.exp(logits - m_)
                    wts = ex / ex.sum()
                    out[f'attn_decay_rate_{h}'][i] = float(np.dot(wts, h_labels))

            # only now fold this row's own (embedding, label, time) into history
            hist.append((cand_emb, float(r[6]), float(r[8])))

    return out


def _load_raw_time(data_dir):
    """Verbatim copy of iter18/iter19's _load_raw_time (same files, same row
    order, same vid2author join, same date-range filtering, same orig_idx
    assignment) -- needed because momentum requires time_ms/orig_idx which
    data.py's plain load() doesn't expose."""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             int(r['hourmin']), int(r['time_ms']), len(rows)))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


IDX = dict(date=0, user_id=1, video_id=2, author_id=3, tab=4, duration_ms=5, label=6,
           hourmin=7, time_ms=8, orig_idx=9,
           activity=10, tab_pos=11, prior_pos=12, prior_total=13,
           last1=14, lastk_sum=15, lastk_cnt=16, gap_ms=17)
DECAY_BASE = 18  # decay_pos_h0 starts here


def _halflife_col(h, halflives=HALFLIVES):
    pos = halflives.index(h)
    return DECAY_BASE + 2 * pos, DECAY_BASE + 2 * pos + 1


def _tab_halflife_col(h, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES):
    base = DECAY_BASE + 2 * len(halflives)
    return base + tab_halflives.index(h)


def _attn_base(halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES):
    return DECAY_BASE + 2 * len(halflives) + len(tab_halflives)


def _attn_col(w, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES, windows=WINDOWS):
    return _attn_base(halflives, tab_halflives) + windows.index(w)


def _attn_decay_col(h, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES, windows=WINDOWS,
                     decay_halflives=ATTN_DECAY_HALFLIVES):
    base = _attn_base(halflives, tab_halflives) + len(windows)
    return base + decay_halflives.index(h)


_CACHE_VERSION = 2  # bumped from iter24's v1: row tuples now carry attention columns too


def _cache_path(halflives, tab_halflives, windows, decay_halflives, k_emb, emb_epochs):
    key = ('-'.join(str(h) for h in halflives) + '__tab_' + '-'.join(str(h) for h in tab_halflives)
           + '__w_' + '-'.join(str(w) for w in windows)
           + '__adh_' + '-'.join(str(h) for h in decay_halflives)
           + f'__k{k_emb}e{emb_epochs}')
    return os.path.join(_THIS_DIR, f'.cache_v{_CACHE_VERSION}_{key}.pkl')


def _emb_cache_path(k_emb, emb_epochs, emb_seed):
    return os.path.join(_THIS_DIR, f'.itememb_v1_k{k_emb}_e{emb_epochs}_s{emb_seed}.pkl')


def get_item_embeddings(train_rows, k=K_EMB, epochs=EMB_EPOCHS, lr=EMB_LR, seed=EMB_SEED,
                         use_cache=True):
    """Cached wrapper around pretrain_item_embeddings -- the pretrain pass is
    a fixed, shared artifact reused across every sweep config/seed of the
    MAIN model, so it is cached separately from the (config-dependent)
    load_ext cache."""
    epath = _emb_cache_path(k, epochs, seed)
    if use_cache and os.path.exists(epath):
        with open(epath, 'rb') as fh:
            return pickle.load(fh)
    item_emb = pretrain_item_embeddings(train_rows, k=k, epochs=epochs, lr=lr, seed=seed)
    if use_cache:
        with open(epath, 'wb') as fh:
            pickle.dump(item_emb, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return item_emb


def load_ext(data_dir, halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES, K=K_DEFAULT,
             windows=WINDOWS, decay_halflives=ATTN_DECAY_HALFLIVES, k_emb=K_EMB,
             emb_epochs=EMB_EPOCHS, use_cache=True):
    """Returns dict split -> list of extended rows (see IDX / DECAY_BASE /
    _tab_halflife_col / _attn_col / _attn_decay_col for layout). Runs FIVE
    independent causal traversals over the same flat (train+valid+test, in
    that order) row list and joins their outputs by row index:
      1. compute_causal_features    (date-grouped flat activity/tab_pos/rate)
      2. compute_momentum_features  (time_ms-level last1/lastk/gap, iter18)
      3. compute_decay_features     (date-grouped exponential decay, fine grid)
      4. compute_decay_tab_features (date-grouped exponential decay of tab_pos)
      5. compute_attention_features (time_ms-level target attention, iter32 NEW)
    Each traversal only reads columns it documents needing and is causally
    self-contained -- combining them is a pure join, not a shared mutable
    pass, so no cross-family leakage is possible by construction.
    """
    cpath = _cache_path(halflives, tab_halflives, windows, decay_halflives, k_emb, emb_epochs)
    if use_cache and os.path.exists(cpath):
        with open(cpath, 'rb') as fh:
            return pickle.load(fh)

    splits = _load_raw_time(data_dir)
    order = ('train', 'valid', 'test')
    flat, owner = [], []
    for name in order:
        for r in splits[name]:
            flat.append(r)
            owner.append(name)

    day_feats = compute_causal_features(flat)
    mom_feats = compute_momentum_features(flat, K=K)
    decayed_pos, decayed_total = compute_decay_features(flat, halflives)
    decayed_tab_pos = compute_decay_tab_features(flat, tab_halflives)

    item_emb = get_item_embeddings(splits['train'], k=k_emb, epochs=emb_epochs, use_cache=use_cache)
    attn_feats = compute_attention_features(flat, item_emb, k_emb, windows=windows,
                                             decay_halflives=decay_halflives)

    ext = {name: [] for name in order}
    for i, (r, name) in enumerate(zip(flat, owner)):
        extra = [day_feats['activity'][i], day_feats['tab_pos'][i],
                 day_feats['prior_pos'][i], day_feats['prior_total'][i],
                 mom_feats['last1'][i], mom_feats['lastk_sum'][i],
                 mom_feats['lastk_cnt'][i], mom_feats['gap_ms'][i]]
        for h in range(len(halflives)):
            extra.append(decayed_pos[i, h])
            extra.append(decayed_total[i, h])
        for h in range(len(tab_halflives)):
            extra.append(decayed_tab_pos[i, h])
        for w in windows:
            extra.append(attn_feats[f'attn_rate_{w}'][i])
        for h in decay_halflives:
            extra.append(attn_feats[f'attn_decay_rate_{h}'][i])
        ext[name].append(r + tuple(extra))

    if use_cache:
        with open(cpath, 'wb') as fh:
            pickle.dump(ext, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ext


def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def encode_ext(splits, feature_set=('decay_rate_3', 'decay_act_3', 'tab', 'last1', 'lastk_rate', 'gap'),
               halflives=HALFLIVES, tab_halflives=TAB_HALFLIVES,
               windows=WINDOWS, decay_halflives=ATTN_DECAY_HALFLIVES):
    """splits: dict from load_ext(). feature_set: subset/order of:
      'activity','tab','rate'         flat features (bucketed)
      'decay_rate_H','decay_act_H'    decayed rate/act, H in halflives (bucketed)
      'decay_tab_H'                   decayed tab_pos, H in tab_halflives (bucketed)
      'last1'                         categorical ('0'/'1'/'UNK')
      'lastk_rate'                    continuous, bucketed
      'gap'                           continuous (ms), bucketed, 'UNK' for first row
      'attn_rate_W'                   DIN-style target-attention pooled rate, window W
                                       in `windows` (bucketed, 'UNK' for no-history rows)
      'attn_decay_rate_H'             recency-decayed-similarity attention pooled rate,
                                       halflife H in `decay_halflives` (bucketed, 'UNK' for
                                       no-history rows)
    Returns (enc, field_dims_sum) with enc[name] = (X, y, users).
    """
    tr = splits['train']
    dur_edges = _bucket_edges([x[IDX['duration_ms']] for x in tr])

    def raw_dur(x):
        return str(int(np.searchsorted(dur_edges, x[IDX['duration_ms']])))

    def parse_feat(name):
        if name in ('activity', 'tab', 'rate', 'last1', 'lastk_rate', 'gap'):
            return name, None
        if name.startswith('decay_rate_'):
            return 'decay_rate', float(name.rsplit('_', 1)[1])
        if name.startswith('decay_act_'):
            return 'decay_act', float(name.rsplit('_', 1)[1])
        if name.startswith('decay_tab_'):
            return 'decay_tab', float(name.rsplit('_', 1)[1])
        if name.startswith('attn_decay_rate_'):
            return 'attn_decay_rate', float(name.rsplit('_', 1)[1])
        if name.startswith('attn_rate_'):
            return 'attn_rate', int(name.rsplit('_', 1)[1])
        raise ValueError(name)

    extra_edges = {}
    for name in feature_set:
        kind, h = parse_feat(name)
        if kind == 'activity':
            extra_edges[name] = _bucket_edges([x[IDX['activity']] for x in tr])
        elif kind == 'tab':
            extra_edges[name] = _bucket_edges([x[IDX['tab_pos']] for x in tr])
        elif kind == 'rate':
            extra_edges[name] = _bucket_edges(
                [(x[IDX['prior_pos']] + ALPHA) / (x[IDX['prior_total']] + 2 * ALPHA) for x in tr])
        elif kind == 'decay_rate':
            pcol, tcol = _halflife_col(h, halflives)
            extra_edges[name] = _bucket_edges([(x[pcol] + ALPHA) / (x[tcol] + 2 * ALPHA) for x in tr])
        elif kind == 'decay_act':
            pcol, tcol = _halflife_col(h, halflives)
            extra_edges[name] = _bucket_edges([x[tcol] for x in tr])
        elif kind == 'decay_tab':
            tcol = _tab_halflife_col(h, halflives, tab_halflives)
            extra_edges[name] = _bucket_edges([x[tcol] for x in tr])
        elif kind == 'lastk_rate':
            extra_edges[name] = _bucket_edges(
                [(x[IDX['lastk_sum']] + ALPHA) / (x[IDX['lastk_cnt']] + 2 * ALPHA) for x in tr])
        elif kind == 'gap':
            gaps = [x[IDX['gap_ms']] for x in tr if x[IDX['gap_ms']] >= 0]
            extra_edges[name] = _bucket_edges(gaps)
        elif kind == 'attn_rate':
            acol = _attn_col(h, halflives, tab_halflives, windows)
            vals = [x[acol] for x in tr if x[acol] != -1.0]
            extra_edges[name] = _bucket_edges(vals)
        elif kind == 'attn_decay_rate':
            acol = _attn_decay_col(h, halflives, tab_halflives, windows, decay_halflives)
            vals = [x[acol] for x in tr if x[acol] != -1.0]
            extra_edges[name] = _bucket_edges(vals)
        # 'last1' needs no edges (raw categorical)

    def extra_val(x, name):
        kind, h = parse_feat(name)
        if kind == 'activity':
            return str(int(np.searchsorted(extra_edges[name], x[IDX['activity']])))
        elif kind == 'tab':
            return str(int(np.searchsorted(extra_edges[name], x[IDX['tab_pos']])))
        elif kind == 'rate':
            r = (x[IDX['prior_pos']] + ALPHA) / (x[IDX['prior_total']] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges[name], r)))
        elif kind == 'decay_rate':
            pcol, tcol = _halflife_col(h, halflives)
            r = (x[pcol] + ALPHA) / (x[tcol] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges[name], r)))
        elif kind == 'decay_act':
            pcol, tcol = _halflife_col(h, halflives)
            return str(int(np.searchsorted(extra_edges[name], x[tcol])))
        elif kind == 'decay_tab':
            tcol = _tab_halflife_col(h, halflives, tab_halflives)
            return str(int(np.searchsorted(extra_edges[name], x[tcol])))
        elif kind == 'last1':
            v = x[IDX['last1']]
            return 'UNK' if v == -1 else str(int(v))
        elif kind == 'lastk_rate':
            r = (x[IDX['lastk_sum']] + ALPHA) / (x[IDX['lastk_cnt']] + 2 * ALPHA)
            return str(int(np.searchsorted(extra_edges[name], r)))
        elif kind == 'gap':
            g = x[IDX['gap_ms']]
            if g < 0:
                return 'UNK'
            return str(int(np.searchsorted(extra_edges[name], g)))
        elif kind == 'attn_rate':
            acol = _attn_col(h, halflives, tab_halflives, windows)
            v = x[acol]
            if v == -1.0:
                return 'UNK'
            return str(int(np.searchsorted(extra_edges[name], v)))
        elif kind == 'attn_decay_rate':
            acol = _attn_decay_col(h, halflives, tab_halflives, windows, decay_halflives)
            v = x[acol]
            if v == -1.0:
                return 'UNK'
            return str(int(np.searchsorted(extra_edges[name], v)))
        raise ValueError(name)

    def raw(x):
        return [x[1], x[2], x[3], x[4], raw_dur(x)] + [extra_val(x, nm) for nm in feature_set]

    fields = BASE_FIELDS + list(feature_set)
    vocabs = [dict() for _ in fields]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(fields)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))


if __name__ == '__main__':
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(_THIS_DIR, '..', '..',
                                                         'KuaiRand-Pure', 'data'))
    ap.add_argument('--no_cache', action='store_true')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    t0 = time.time()
    ext = load_ext(a.data_dir, use_cache=not a.no_cache)
    print({k: len(v) for k, v in ext.items()}, f"  ({time.time()-t0:.1f}s)")

    flat = ext['train'] + ext['valid'] + ext['test']
    n = len(flat)

    # ================================================================
    # PART A: decay-family (rate/act, fine grid) causal spot-checks
    # ================================================================
    print("\n=== PART A: decay-feature (rate/act) causal spot-checks (brute force) ===")
    for h in HALFLIVES:
        pcol, tcol = _halflife_col(h)
        tot = np.array([r[tcol] for r in flat])
        print(f"halflife={h:4.1f}d  decayed_total>0 coverage: {np.mean(tot > 0)*100:.2f}%  "
              f"(mean={tot.mean():.3f}, max={tot.max():.3f})")

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=25, replace=False)
    max_err = 0.0
    for idx in sample_idx:
        r = flat[idx]
        date, uid = r[0], r[1]
        d_ord = _date_to_ordinal(date)
        earlier = [rr for rr in flat if rr[1] == uid and rr[0] < date]
        for h in HALFLIVES:
            pcol, tcol = _halflife_col(h)
            manual_pos = sum(0.5 ** ((d_ord - _date_to_ordinal(rr[0])) / h)
                              for rr in earlier if rr[6] == 1)
            manual_tot = sum(0.5 ** ((d_ord - _date_to_ordinal(rr[0])) / h) for rr in earlier)
            err_p = abs(manual_pos - r[pcol])
            err_t = abs(manual_tot - r[tcol])
            max_err = max(max_err, err_p, err_t)
            assert err_p < 1e-6, f"CAUSALITY BUG: decayed_pos mismatch h={h} idx={idx}"
            assert err_t < 1e-6, f"CAUSALITY BUG: decayed_total mismatch h={h} idx={idx}"
    print(f"25 random rows x {len(HALFLIVES)} halflives: decayed_pos/decayed_total match brute "
          f"force (max abs err {max_err:.2e}). No leakage detected.")

    zero_examples = [r for r in flat if r[IDX['activity']] == 0][:5]
    for r in zero_examples:
        for h in HALFLIVES:
            pcol, tcol = _halflife_col(h)
            assert r[pcol] == 0.0 and r[tcol] == 0.0, "CAUSALITY BUG: zero-activity row has nonzero decay!"
        for h in TAB_HALFLIVES:
            tcol = _tab_halflife_col(h)
            assert r[tcol] == 0.0, "CAUSALITY BUG: zero-activity row has nonzero decay_tab!"
    print(f"zero-activity rows ({len(zero_examples)} checked): decayed_pos/total/tab correctly 0.0.")

    by_u_date = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        if r[6] == 1:
            by_u_date[(r[1], r[0])].append(idx)
    same_date_case = next((v for v in by_u_date.values() if len(v) >= 2), None)
    if same_date_case:
        h = HALFLIVES[0]
        pcol, tcol = _halflife_col(h)
        vals = [(flat[idx][pcol], flat[idx][tcol]) for idx in same_date_case]
        dp0, dt0 = vals[0]
        for dp, dt in vals:
            assert abs(dp - dp0) < 1e-9 and abs(dt - dt0) < 1e-9, \
                "CAUSALITY BUG: same-date pair should show identical decay state!"
        print(f"same-date-pair edge case ({len(same_date_case)} rows): decay values identical "
              "across the pair, as expected (same-date rows never see each other). PASSED.")
    else:
        print("(no same-date-pair found -- skipping, not a failure)")

    # ================================================================
    # PART B: decayed tab_pos (decay_tab, iter20-style) causal spot-checks
    # ================================================================
    print("\n=== PART B: decayed tab_pos causal spot-checks (brute force) ===")
    for h in TAB_HALFLIVES:
        tcol = _tab_halflife_col(h)
        tot = np.array([r[tcol] for r in flat])
        print(f"tab halflife={h:4.1f}d  decayed_tab_pos>0 coverage: {np.mean(tot > 0)*100:.2f}%  "
              f"(mean={tot.mean():.3f}, max={tot.max():.3f})")

    max_err_tab = 0.0
    sample_idx2 = rng.choice(n, size=30, replace=False)
    for idx in sample_idx2:
        r = flat[idx]
        date, uid, tab = r[0], r[1], r[4]
        d_ord = _date_to_ordinal(date)
        earlier_same_tab = [rr for rr in flat if rr[1] == uid and rr[4] == tab and rr[0] < date and rr[6] == 1]
        for h in TAB_HALFLIVES:
            tcol = _tab_halflife_col(h)
            manual = sum(0.5 ** ((d_ord - _date_to_ordinal(rr[0])) / h) for rr in earlier_same_tab)
            err = abs(manual - r[tcol])
            max_err_tab = max(max_err_tab, err)
            assert err < 1e-6, f"CAUSALITY BUG: decayed_tab_pos mismatch h={h} idx={idx} manual={manual} got={r[tcol]}"
    print(f"30 random rows x {len(TAB_HALFLIVES)} tab-halflives: all decayed_tab_pos match brute force "
          f"(max abs err {max_err_tab:.2e}). No leakage detected.")

    zero_tab_examples = [r for r in flat if r[IDX['tab_pos']] == 0 and r[IDX['activity']] > 0][:5]
    for r in zero_tab_examples:
        for h in TAB_HALFLIVES:
            tcol = _tab_halflife_col(h)
            assert r[tcol] == 0.0, "CAUSALITY BUG: zero-tab_pos row (with other activity) has nonzero decay_tab!"
    if zero_tab_examples:
        print(f"zero-tab_pos-but-nonzero-activity rows ({len(zero_tab_examples)} checked): "
              f"decayed_tab_pos correctly 0.0 despite nonzero decay_rate/act.")

    by_u_tab_date = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        if r[6] == 1:
            by_u_tab_date[(r[1], r[4], r[0])].append(idx)
    same_date_tab_case = next((v for v in by_u_tab_date.values() if len(v) >= 2), None)
    if same_date_tab_case:
        h = TAB_HALFLIVES[0]
        tcol = _tab_halflife_col(h)
        vals = [(flat[idx][1], flat[idx][4], flat[idx][0], flat[idx][tcol]) for idx in same_date_tab_case]
        dv0 = vals[0][3]
        for uid, tab, date, dv in vals:
            assert abs(dv - dv0) < 1e-9, \
                "CAUSALITY BUG: same-date pair should show identical (mutually-exclusive) decay_tab state!"
        print(f"same-date-pair edge case (decay_tab, {len(same_date_tab_case)} rows): decayed_tab_pos "
              "identical across the pair, as expected. PASSED.")
    else:
        print("(no same-user/same-tab/same-date positive pair found -- skipping, not a failure)")

    # ================================================================
    # PART C: momentum-family causal spot-checks (adapted from iter18/
    # iter19's __main__), re-run against the COMBINED row tuple.
    # ================================================================
    print("\n=== PART C: momentum-feature causal spot-checks (brute force) ===")
    last1 = np.array([r[IDX['last1']] for r in flat])
    gap_ms = np.array([r[IDX['gap_ms']] for r in flat])
    print(f"last1 coverage (not user's first row): {np.mean(last1 >= 0)*100:.2f}%")
    print(f"gap coverage (not user's first row): {np.mean(gap_ms >= 0)*100:.2f}%")

    by_user = collections.defaultdict(list)
    for idx, r in enumerate(flat):
        by_user[r[IDX['user_id']]].append(idx)
    candidate_users = [u for u, idxs in by_user.items() if 8 <= len(idxs) <= 14][:3]

    def manual_check(u):
        idxs = sorted(by_user[u], key=lambda i: (flat[i][IDX['time_ms']], flat[i][IDX['orig_idx']]))
        print(f"\nuser={u}  ({len(idxs)} rows, chronological order)")
        window = []
        prev_t = None
        for pos, i in enumerate(idxs):
            r = flat[i]
            t = r[IDX['time_ms']]
            manual_last1 = window[-1] if window else -1
            manual_lastk_sum = sum(window[-5:])
            manual_lastk_cnt = len(window[-5:])
            manual_gap = (t - prev_t) if prev_t is not None else -1
            ok = (manual_last1 == r[IDX['last1']] and manual_lastk_sum == r[IDX['lastk_sum']]
                  and manual_lastk_cnt == r[IDX['lastk_cnt']] and manual_gap == r[IDX['gap_ms']])
            flag = "OK" if ok else "MISMATCH!!"
            print(f"  pos={pos:2d} date={r[IDX['date']]} time_ms={t} label={r[IDX['label']]} "
                  f"| last1 got={r[IDX['last1']]} manual={manual_last1} "
                  f"| lastk_sum got={r[IDX['lastk_sum']]} manual={manual_lastk_sum} "
                  f"| gap_ms got={r[IDX['gap_ms']]} manual={manual_gap}  [{flag}]")
            assert ok, f"CAUSALITY BUG for user {u} at pos {pos}!"
            window.append(r[IDX['label']])
            prev_t = t

    for u in candidate_users:
        manual_check(u)

    print("\n--- synthetic same-time_ms tie stress test (momentum) ---")
    fake_rows = [
        (20220410, 'TIEUSER', 'v1', 'a1', '0', 1000.0, 1, 1000, 5000, 100),
        (20220410, 'TIEUSER', 'v2', 'a1', '0', 1000.0, 0, 1000, 5000, 101),
        (20220410, 'TIEUSER', 'v3', 'a1', '0', 1000.0, 1, 1005, 5500, 99),
    ]
    feats = compute_momentum_features(fake_rows, K=5)
    assert feats['last1'][0] == -1
    assert feats['last1'][1] == 1
    assert feats['last1'][2] == 0
    assert feats['gap_ms'][0] == -1
    assert feats['gap_ms'][1] == 0
    assert feats['gap_ms'][2] == 500
    print("tie stress test: all assertions passed.")

    # ================================================================
    # PART D: cross-family joint edge case -- a same-user, same-date pair
    # with distinct time_ms. Now THREE families (decay, decay_tab,
    # momentum) must be checked pairwise on the same rows: decay/decay_tab
    # must be IDENTICAL across the pair (date-level, blind to intra-date
    # order); momentum must DIFFER and correctly resolve the true time_ms
    # order. This directly checks all pairwise interactions the dispatch
    # prompt calls out: decay vs decay_tab (should agree: both date-blind),
    # decay vs momentum (should disagree: date-blind vs time_ms-aware),
    # decay_tab vs momentum (same).
    # ================================================================
    print("\n=== PART D: cross-family joint edge case (same-date, different time_ms pair) ===")
    joint_case = None
    for (uid, date), idxs in by_u_date.items():
        if len(idxs) >= 2:
            times = sorted(set(flat[i][IDX['time_ms']] for i in idxs))
            if len(times) >= 2:
                joint_case = (uid, date, idxs)
                break
    if joint_case:
        uid, date, idxs = joint_case
        idxs_sorted = sorted(idxs, key=lambda i: (flat[i][IDX['time_ms']], flat[i][IDX['orig_idx']]))
        h = HALFLIVES[0]
        pcol, tcol = _halflife_col(h)
        th = TAB_HALFLIVES[0]
        ttcol = _tab_halflife_col(th)
        print(f"user={uid} date={date}: {len(idxs_sorted)} rows, same calendar date, distinct time_ms")
        decay_vals, decay_tab_vals = set(), set()
        for rank, i in enumerate(idxs_sorted):
            r = flat[i]
            decay_vals.add((round(r[pcol], 9), round(r[tcol], 9)))
            decay_tab_vals.add(round(r[ttcol], 9))
            print(f"  rank={rank} time_ms={r[IDX['time_ms']]} label={r[IDX['label']]} "
                  f"decay_pos={r[pcol]:.4f} decay_total={r[tcol]:.4f} decay_tab={r[ttcol]:.4f} "
                  f"last1={r[IDX['last1']]} gap_ms={r[IDX['gap_ms']]}")
        assert len(decay_vals) == 1, \
            "CAUSALITY BUG: decay (rate/act) feature should be IDENTICAL across a same-date pair"
        assert len(decay_tab_vals) == 1, \
            "CAUSALITY BUG: decay_tab feature should be IDENTICAL across a same-date pair"
        last1_vals = [flat[i][IDX['last1']] for i in idxs_sorted]
        assert last1_vals[1] == flat[idxs_sorted[0]][IDX['label']], \
            "CAUSALITY BUG: momentum last1 for later-time_ms row should equal earlier row's label"
        print("  -> decay AND decay_tab features are IDENTICAL across the pair (both date-level, "
              "correctly blind to intra-date order); momentum last1 correctly DIFFERS and resolves "
              "the true time_ms order. All three families verified independently correct on the "
              "same rows -- no cross-contamination from the join (decay<->decay_tab agree by "
              "construction, both disagree with momentum by construction).")
    else:
        print("(no same-user/same-date pair with >=2 distinct time_ms found -- skipping)")

    # ================================================================
    # PART E: target-attention (iter32 NEW) causal spot-checks. This is
    # the most leakage-prone feature family in this repo (per the dispatch
    # prompt): it jointly depends on per-row history retrieval AND item
    # embeddings, so it gets the same brute-force treatment as PART A/B/C
    # plus two extra checks specific to attention (zero-history sentinel,
    # unseen-item degrade-to-uniform-mean).
    # ================================================================
    print("\n=== PART E: target-attention causal spot-checks (brute force) ===")
    raw_splits = _load_raw_time(a.data_dir)
    item_emb = get_item_embeddings(raw_splits['train'], k=K_EMB, epochs=EMB_EPOCHS, seed=EMB_SEED,
                                    use_cache=not a.no_cache)
    zero_vec = np.zeros(K_EMB, dtype=np.float64)
    scale = 1.0 / np.sqrt(K_EMB)
    ln_half = np.log(0.5)
    day_ms = 86400000.0

    for w in WINDOWS:
        acol = _attn_col(w)
        vals = np.array([r[acol] for r in flat])
        cov = np.mean(vals != -1.0) * 100
        print(f"window={w:3d}  attn_rate coverage: {cov:.2f}%  "
              f"(mean over covered={vals[vals != -1.0].mean():.4f})")
    for h in ATTN_DECAY_HALFLIVES:
        acol = _attn_decay_col(h)
        vals = np.array([r[acol] for r in flat])
        cov = np.mean(vals != -1.0) * 100
        print(f"decay halflife={h:4.1f}d  attn_decay_rate coverage: {cov:.2f}%  "
              f"(mean over covered={vals[vals != -1.0].mean():.4f})")

    def manual_attn(uid, cand_time, cand_orig_idx, cand_vid):
        earlier = [rr for rr in flat if rr[1] == uid and (rr[8], rr[9]) < (cand_time, cand_orig_idx)]
        earlier.sort(key=lambda rr: (rr[8], rr[9]))
        earlier = earlier[-max(WINDOWS):]
        if not earlier:
            return None
        h_embs = np.stack([item_emb.get(rr[2], zero_vec) for rr in earlier])
        h_labels = np.array([float(rr[6]) for rr in earlier])
        h_times = np.array([float(rr[8]) for rr in earlier])
        cand_emb = item_emb.get(cand_vid, zero_vec)
        sims = h_embs.dot(cand_emb) * scale
        out_rate = {}
        for w in WINDOWS:
            sw = sims[-w:]; lw = h_labels[-w:]
            m_ = sw.max(); ex = np.exp(sw - m_); wts = ex / ex.sum()
            out_rate[w] = float(np.dot(wts, lw))
        out_decay = {}
        gap_days = (float(cand_time) - h_times) / day_ms
        for h in ATTN_DECAY_HALFLIVES:
            logits = sims + gap_days * (ln_half / h)
            m_ = logits.max(); ex = np.exp(logits - m_); wts = ex / ex.sum()
            out_decay[h] = float(np.dot(wts, h_labels))
        return out_rate, out_decay

    rng2 = np.random.default_rng(1)
    sample_idx3 = rng2.choice(n, size=25, replace=False)
    max_err_attn = 0.0
    checked = 0
    for idx in sample_idx3:
        r = flat[idx]
        uid, vid, t, oi = r[1], r[2], r[8], r[9]
        manual = manual_attn(uid, t, oi, vid)
        if manual is None:
            for w in WINDOWS:
                assert r[_attn_col(w)] == -1.0, \
                    f"CAUSALITY BUG: zero-history row idx={idx} has non-sentinel attn_rate_{w}"
            for h in ATTN_DECAY_HALFLIVES:
                assert r[_attn_decay_col(h)] == -1.0, \
                    f"CAUSALITY BUG: zero-history row idx={idx} has non-sentinel attn_decay_rate_{h}"
            continue
        out_rate, out_decay = manual
        checked += 1
        for w in WINDOWS:
            err = abs(out_rate[w] - r[_attn_col(w)])
            max_err_attn = max(max_err_attn, err)
            assert err < 1e-6, \
                f"CAUSALITY BUG: attn_rate_{w} mismatch idx={idx} manual={out_rate[w]} got={r[_attn_col(w)]}"
        for h in ATTN_DECAY_HALFLIVES:
            err = abs(out_decay[h] - r[_attn_decay_col(h)])
            max_err_attn = max(max_err_attn, err)
            assert err < 1e-6, (f"CAUSALITY BUG: attn_decay_rate_{h} mismatch idx={idx} "
                                f"manual={out_decay[h]} got={r[_attn_decay_col(h)]}")
    print(f"{len(sample_idx3)} random rows ({checked} with nonzero history, "
          f"{len(sample_idx3) - checked} zero-history sentinel-checked) x {len(WINDOWS)} windows x "
          f"{len(ATTN_DECAY_HALFLIVES)} decay-halflives: attn_rate/attn_decay_rate match brute force "
          f"(max abs err {max_err_attn:.2e}). No leakage detected.")

    print("\n--- synthetic same-time_ms tie stress test (attention) ---")
    fake_rows_attn = [
        (20220410, 'TIEUSER2', 'x1', 'a1', '0', 1000.0, 1, 1000, 5000, 200),
        (20220410, 'TIEUSER2', 'x2', 'a1', '0', 1000.0, 0, 1000, 5000, 201),
        (20220410, 'TIEUSER2', 'x3', 'a1', '0', 1000.0, 1, 1000, 5000, 202),
    ]
    fake_item_emb = {'x1': np.array([1.0, 0.0]), 'x2': np.array([0.0, 1.0]), 'x3': np.array([1.0, 1.0])}
    feats_fake = compute_attention_features(fake_rows_attn, fake_item_emb, k_emb=2, windows=(2,),
                                             decay_halflives=(3.0,))
    assert feats_fake['attn_rate_2'][0] == -1.0
    assert abs(feats_fake['attn_rate_2'][1] - 1.0) < 1e-9
    assert abs(feats_fake['attn_rate_2'][2] - 0.5) < 1e-9
    assert feats_fake['attn_decay_rate_3.0'][0] == -1.0
    assert abs(feats_fake['attn_decay_rate_3.0'][1] - 1.0) < 1e-9
    assert abs(feats_fake['attn_decay_rate_3.0'][2] - 0.5) < 1e-9
    print("all three rows share time_ms=5000, differ only by orig_idx -- row 3's attention correctly "
          "sees only rows 1&2 (never itself, never a 'future' same-time_ms row with higher orig_idx). "
          "tie stress test: all assertions passed.")

    print("\n--- unseen-item / missing-embedding degrade-gracefully check (attention) ---")
    fake_rows_unseen = [
        (20220410, 'TIEUSER3', 'u1', 'a1', '0', 1000.0, 1, 1000, 6000, 300),
        (20220410, 'TIEUSER3', 'u2', 'a1', '0', 1000.0, 0, 1000, 6000, 301),
        (20220410, 'TIEUSER3', 'u3', 'a1', '0', 1000.0, 1, 1000, 6000, 302),
    ]
    feats_unseen = compute_attention_features(fake_rows_unseen, {}, k_emb=2, windows=(2,),
                                               decay_halflives=(3.0,))
    assert feats_unseen['attn_rate_2'][0] == -1.0
    assert abs(feats_unseen['attn_rate_2'][1] - 1.0) < 1e-9
    assert abs(feats_unseen['attn_rate_2'][2] - 0.5) < 1e-9
    print("with all item embeddings missing (empty item_emb dict), attention logits are all-zero, "
          "softmax is uniform, and attn_rate correctly reduces to a plain unweighted mean of the "
          "window's labels (0.5 for a [1,0] window) -- no crash, no NaN, graceful degradation confirmed.")

    print("\nAll causal spot-checks (decay + decay_tab + momentum + attention + cross-family joint) passed.")

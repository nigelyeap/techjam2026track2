"""Trains iter64's SASRec-style history encoder standalone via pairwise BPR
(same per-user positive/negative sampling infra as the FM's own BPR loop --
`build_pos_neg_index`/`sample_pairs`, imported verbatim from
iter27_triple_fusion/train.py, TRAIN split only) and reports valid/test
primary via evaluate.py, exactly like every other model in this project.

No dependency on data.py's `encode()`/bucketed feature pipeline and no
shared parameters with the FM or GBM -- this model only ever sees
(history item-id sequence, target item-id) pairs from its own vocabulary.
"""
import os, sys, importlib.util, time
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_de = _load_module(os.path.join(_THIS_DIR, 'data_ext.py'), 'iter64_data_ext')
load_ext = _de.load_ext
_md = _load_module(os.path.join(_THIS_DIR, 'model.py'), 'iter64_model')
SASRecScorer = _md.SASRecScorer

# iter27's train.py does `from data_ext import ...` (a plain, sys.path-based
# import) -- with iter64's own data_ext.py loaded above (module name
# 'iter64_data_ext', NOT 'data_ext'), sys.modules has no 'data_ext' entry yet
# to collide with, so temporarily prepending iter27's own directory resolves
# its plain import to iter27's own data_ext.py, not iter64's.
_iter27_dir = os.path.join(_THIS_DIR, '..', 'iter27_triple_fusion')
sys.path.insert(0, _iter27_dir)
try:
    _iter27_train = _load_module(os.path.join(_iter27_dir, 'train.py'), 'iter27_train_for_iter64')
finally:
    sys.path.remove(_iter27_dir)
build_pos_neg_index = _iter27_train.build_pos_neg_index
sample_pairs = _iter27_train.sample_pairs


def run(data_dir, max_len=20, d=32, n_heads=2, dropout=0.2, lr=1e-3, weight_decay=1e-5,
        bs=4096, epochs=25, patience=3, sampling_alpha=0.0, seed=0, verbose=True,
        eval_every_steps=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ext, vocab = load_ext(data_dir, max_len=max_len)
    vocab_size = len(vocab) + 2
    hist_tr, len_tr, item_tr, y_tr, u_tr = ext['train']
    hist_va, len_va, item_va, y_va, u_va = ext['valid']
    hist_te, len_te, item_te, y_te, u_te = ext['test']

    hist_tr_t = torch.from_numpy(hist_tr)
    item_tr_t = torch.from_numpy(item_tr)
    hist_va_t = torch.from_numpy(hist_va)
    item_va_t = torch.from_numpy(item_va)
    hist_te_t = torch.from_numpy(hist_te)
    item_te_t = torch.from_numpy(item_te)

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = \
        build_pos_neg_index(y_tr, u_tr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs)))

    model = SASRecScorer(vocab_size, d=d, n_heads=n_heads, dropout=dropout, max_len=max_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    def eval_split(hist_t, item_t, y, u, bs_eval=32768):
        model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(y), bs_eval):
                h = hist_t[i:i + bs_eval]
                it = item_t[i:i + bs_eval]
                s = model.score(h, it)
                scores.append(torch.sigmoid(s).numpy())
        scores = np.concatenate(scores)
        return evaluate(u, y, scores), scores

    best_va, best_state, bad = -1, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            pos_rows, neg_rows = sample_pairs(rng, n_users, bs, pos_flat, pos_start, pos_len,
                                               neg_flat, neg_start, neg_len)
            pos_rows_t = torch.from_numpy(pos_rows.astype(np.int64))
            neg_rows_t = torch.from_numpy(neg_rows.astype(np.int64))

            s_pos = model.score(hist_tr_t[pos_rows_t], item_tr_t[pos_rows_t])
            s_neg = model.score(hist_tr_t[neg_rows_t], item_tr_t[neg_rows_t])
            loss = -torch.nn.functional.logsigmoid(s_pos - s_neg).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        va_metrics, _ = eval_split(hist_va_t, item_va_t, y_va, u_va)
        if verbose:
            print(f"  epoch {ep}: loss={ep_loss/steps_per_epoch:.4f} valid_primary={va_metrics['primary']:.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if va_metrics['primary'] > best_va + 1e-5:
            best_va, bad = va_metrics['primary'], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    va_metrics, va_scores = eval_split(hist_va_t, item_va_t, y_va, u_va)
    te_metrics, te_scores = eval_split(hist_te_t, item_te_t, y_te, u_te)
    if verbose:
        print(f"[iter64 SASRec] best valid={va_metrics['primary']:.5f} test={te_metrics['primary']:.5f}", flush=True)
    return model, va_metrics, te_metrics, (va_scores, te_scores, y_va, u_va, y_te, u_te)


if __name__ == '__main__':
    DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data')
    run(DATA_DIR, verbose=True)

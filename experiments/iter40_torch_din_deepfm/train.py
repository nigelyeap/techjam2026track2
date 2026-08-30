"""iter40 training loop: FM(+DIN attention +deep MLP), BPR loss, the same
decay-weighted user sampling scheme proven in iter22/23/27 (imported
verbatim, not reimplemented). PyTorch/autodiff; MPS (Apple GPU) if available.
"""
import os, sys, time, importlib.util
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))
from evaluate import evaluate  # noqa: E402

import data_prep  # noqa: E402
from model import FMDeepDIN  # noqa: E402


def _load_iter27_train():
    iter27_dir = os.path.join(_THIS_DIR, '..', 'iter27_triple_fusion')
    sys.path.insert(0, iter27_dir)
    path = os.path.join(iter27_dir, 'train.py')
    spec = importlib.util.spec_from_file_location('iter27_train', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_t27 = _load_iter27_train()
build_pos_neg_index = _t27.build_pos_neg_index
sample_pairs = _t27.sample_pairs


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def to_device(arr, device, dtype=None):
    t = torch.as_tensor(arr)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(device)


@torch.no_grad()
def batched_predict(model, X, hv, ha, hl, hm, device, batch=32768):
    model.eval()
    out = []
    n = X.shape[0]
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        z = model(X[sl], hv[sl], ha[sl], hl[sl], hm[sl])
        out.append(z.cpu().numpy())
    return np.concatenate(out)


def run(data_dir, k=16, lr=0.001, weight_decay=1e-6, epochs=25, bs=8192,
        patience=4, seed=0, sampling_alpha=0.75, decay_halflife=3,
        use_attention=True, use_deep=True, use_attn_direct=False, deep_hidden=(128, 64),
        dropout=0.1, verbose=True,
        _cache=None):
    device = get_device()
    if _cache is None:
        enc, dim, hist, decayed_pos_dict, PAD_IDX = data_prep.prepare(data_dir)
        _cache = (enc, dim, hist, decayed_pos_dict, PAD_IDX)
    else:
        enc, dim, hist, decayed_pos_dict, PAD_IDX = _cache

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    Xtr_np, ytr, utr = enc['train']
    Xva_np, yva, uva = enc['valid']
    Xte_np, yte, ute = enc['test']

    Xtr = to_device(Xtr_np, device, torch.long)
    Xva = to_device(Xva_np, device, torch.long)
    Xte = to_device(Xte_np, device, torch.long)

    hv_tr, ha_tr, hl_tr, hm_tr = hist['train']
    hv_va, ha_va, hl_va, hm_va = hist['valid']
    hv_te, ha_te, hl_te, hm_te = hist['test']
    Htr = tuple(to_device(a, device, dt) for a, dt in
                zip((hv_tr, ha_tr, hl_tr, hm_tr), (torch.long, torch.long, torch.float32, torch.float32)))
    Hva = tuple(to_device(a, device, dt) for a, dt in
                zip((hv_va, ha_va, hl_va, hm_va), (torch.long, torch.long, torch.float32, torch.float32)))
    Hte = tuple(to_device(a, device, dt) for a, dt in
                zip((hv_te, ha_te, hl_te, hm_te), (torch.long, torch.long, torch.float32, torch.float32)))

    eligible, pos_flat, pos_start, pos_len, neg_flat, neg_start, neg_len = build_pos_neg_index(ytr, utr)
    n_users = len(eligible)
    steps_per_epoch = max(1, int(np.ceil(pos_len.sum() / bs)))

    decayed_arr = np.array([decayed_pos_dict.get(u, 0.0) for u in eligible], dtype=np.float64)
    weights = decayed_arr ** sampling_alpha
    user_cumw = np.cumsum(weights); user_totalw = user_cumw[-1]

    model = FMDeepDIN(dim, k=k, n_fields=Xtr.shape[1], use_attention=use_attention, use_deep=use_deep,
                      use_attn_direct=use_attn_direct, deep_hidden=deep_hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            pos_rows, neg_rows = sample_pairs(rng, n_users, bs, pos_flat, pos_start, pos_len,
                                               neg_flat, neg_start, neg_len,
                                               user_cumw=user_cumw, user_totalw=user_totalw)
            pos_rows_t = torch.as_tensor(pos_rows, device=device, dtype=torch.long)
            neg_rows_t = torch.as_tensor(neg_rows, device=device, dtype=torch.long)

            zpos = model(Xtr[pos_rows_t], Htr[0][pos_rows_t], Htr[1][pos_rows_t],
                         Htr[2][pos_rows_t], Htr[3][pos_rows_t])
            zneg = model(Xtr[neg_rows_t], Htr[0][neg_rows_t], Htr[1][neg_rows_t],
                         Htr[2][neg_rows_t], Htr[3][neg_rows_t])
            loss = -torch.nn.functional.logsigmoid(zpos - zneg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        va_scores = batched_predict(model, Xva, *Hva, device)
        va = evaluate(uva, yva, va_scores)
        if verbose:
            print(f"    epoch {ep:2d} | loss {total_loss/steps_per_epoch:.4f} | "
                  f"valid primary {va['primary']:.4f} | {time.time()-t0:.1f}s", flush=True)
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop at epoch {ep}", flush=True)
                break

    model.load_state_dict(best_state)
    va_scores = batched_predict(model, Xva, *Hva, device)
    te_scores = batched_predict(model, Xte, *Hte, device)
    va_final = evaluate(uva, yva, va_scores)
    te_final = evaluate(ute, yte, te_scores)
    return model, va_final, te_final, _cache


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=os.path.join(_THIS_DIR, '..', '..', 'KuaiRand-Pure', 'data'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--no_attention', action='store_true')
    p.add_argument('--no_deep', action='store_true')
    args = p.parse_args()
    print(f"device: {get_device()}")
    model, va, te, _ = run(args.data_dir, k=args.k, lr=args.lr, epochs=args.epochs, seed=args.seed,
                            use_attention=not args.no_attention, use_deep=not args.no_deep)
    print("valid:", va)
    print("test:", te)

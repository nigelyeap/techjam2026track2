"""iter40: FM (wide) + DIN-style causal target-attention over user history
(deep tower input) + small MLP (deep), trained end-to-end with PyTorch
autodiff -- unlike iter32's frozen-pretrained-embedding attention feature,
gradients here flow from the BPR loss straight into the shared embedding
table that also serves the FM's own video_id/author_id fields.

`use_attention` / `use_deep` flags let the harness-fidelity check run this
exact model with both new mechanisms switched off, reducing it to plain
FM (+ engineered categorical fields), for comparison against the proven
numpy FM+BPR numbers before trusting anything new.
"""
import torch
import torch.nn as nn


class FMDeepDIN(nn.Module):
    def __init__(self, dim, k=16, n_fields=11, deep_hidden=(128, 64),
                 attn_hidden=64, dropout=0.1, use_attention=True, use_deep=True,
                 use_attn_direct=False):
        super().__init__()
        self.dim = dim          # PAD_IDX == dim, embedding table sized dim+1
        self.k = k
        self.n_fields = n_fields  # total X columns: 5 base fields + engineered feature_set
        self.use_attention = use_attention
        self.use_deep = use_deep
        self.use_attn_direct = use_attn_direct

        self.V = nn.Embedding(dim + 1, k, padding_idx=dim)
        self.W = nn.Embedding(dim + 1, 1, padding_idx=dim)
        self.b = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.V.weight, 0, 0.01)
        nn.init.zeros_(self.W.weight)
        with torch.no_grad():
            self.V.weight[dim].zero_()
            self.W.weight[dim].zero_()

        if use_attention:
            attn_in = 4 * (2 * k) + 1  # [hist, target, hist*target, hist-target, label]
            self.attn_mlp = nn.Sequential(
                nn.Linear(attn_in, attn_hidden), nn.ReLU(),
                nn.Linear(attn_hidden, 1),
            )

        if use_deep:
            deep_in = self.n_fields * k + (2 * k if use_attention else 0) + (2 * k if use_attention else 0)
            layers = []
            prev = deep_in
            for h in deep_hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers += [nn.Linear(prev, 1)]
            self.deep_mlp = nn.Sequential(*layers)

        if use_attn_direct:
            # low-capacity bilinear-style head: learned weighted dot product of
            # (user_interest * target_repr), added straight to the FM logit --
            # no dense MLP, so it composes with the FM's own pairwise structure
            # instead of competing with it for the shared embedding table.
            self.attn_direct_head = nn.Linear(2 * k, 1, bias=False)
            nn.init.zeros_(self.attn_direct_head.weight)

    def fm_forward(self, X):
        """X: (B,F) int64, all fields (base + engineered categorical).
        Returns (fm_logit, E) where E = V[X] (B,F,k) for reuse by the deep tower."""
        E = self.V(X)                       # (B,F,k)
        S = E.sum(1)                        # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        lin = self.W(X).squeeze(-1).sum(1)  # (B,)
        return self.b.squeeze() + lin + inter, E

    def attention_forward(self, X, hist_video, hist_author, hist_label, hist_mask):
        """Returns (user_interest (B,2k), target_repr (B,2k))."""
        target_video = X[:, 1]   # BASE_FIELDS[1] = video_id
        target_author = X[:, 2]  # BASE_FIELDS[2] = author_id
        v_t = self.V(target_video)          # (B,k)
        a_t = self.V(target_author)         # (B,k)
        target_repr = torch.cat([v_t, a_t], dim=-1)              # (B,2k)

        hv = self.V(hist_video)             # (B,L,k)
        ha = self.V(hist_author)            # (B,L,k)
        hist_repr = torch.cat([hv, ha], dim=-1)                  # (B,L,2k)

        target_bcast = target_repr.unsqueeze(1).expand_as(hist_repr)  # (B,L,2k)
        attn_in = torch.cat([
            hist_repr, target_bcast, hist_repr * target_bcast,
            hist_repr - target_bcast, hist_label.unsqueeze(-1),
        ], dim=-1)                                                # (B,L,4*2k+1)
        logits = self.attn_mlp(attn_in).squeeze(-1)               # (B,L)
        logits = logits.masked_fill(hist_mask == 0, float('-inf'))
        has_hist = hist_mask.sum(1) > 0
        # rows with zero real history: avoid softmax(all -inf) NaN, force weights to 0
        safe_logits = torch.where(has_hist.unsqueeze(1), logits, torch.zeros_like(logits))
        safe_mask = torch.where(has_hist.unsqueeze(1), hist_mask, torch.zeros_like(hist_mask))
        weights = torch.softmax(safe_logits, dim=1) * safe_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        weights = torch.where(has_hist.unsqueeze(1), weights, torch.zeros_like(weights))
        user_interest = (weights.unsqueeze(-1) * hist_repr).sum(dim=1)  # (B,2k)
        return user_interest, target_repr

    def forward(self, X, hist_video=None, hist_author=None, hist_label=None, hist_mask=None):
        fm_logit, E = self.fm_forward(X)
        z = fm_logit
        user_interest = target_repr = None
        if self.use_attention and (self.use_deep or self.use_attn_direct):
            user_interest, target_repr = self.attention_forward(
                X, hist_video, hist_author, hist_label, hist_mask)
        if self.use_attn_direct:
            z = z + self.attn_direct_head(user_interest * target_repr).squeeze(-1)
        if self.use_deep:
            B = X.shape[0]
            deep_parts = [E.reshape(B, -1)]
            if self.use_attention:
                deep_parts += [user_interest, target_repr]
            deep_in = torch.cat(deep_parts, dim=-1)
            z = z + self.deep_mlp(deep_in).squeeze(-1)
        return z

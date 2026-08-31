"""A minimal SASRec-style self-attention user-history encoder, PyTorch,
trained fully independently (own embedding table, own loss, own optimizer)
-- never shares parameters or a gradient path with the FM (baseline.py) or
the GBM (iter63's LightGBM ranker). Its predicted score is combined with the
existing best (iter63) blend only post-hoc via score-blending, matching how
the FM and GBM components are already composed.

Architecture: item embedding (padding_idx=0) + learned positional embedding
-> ONE manual scaled-dot-product self-attention block (multi-head) over the
user's causally-prior history (already strictly-prior by construction, see
data_ext.py, so no additional causal mask is needed within the block itself)
-> masked mean-pool over real (non-pad) positions -> dot product with the
candidate item's own embedding = predicted score. A user with zero prior
history (their first-ever row) gets a well-defined, NaN-free zero-context
vector (see `_self_attend`'s padding-mask handling).
"""
import torch
import torch.nn as nn

PAD = 0


class SASRecScorer(nn.Module):
    def __init__(self, vocab_size, d=32, n_heads=2, dropout=0.2, max_len=20):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.max_len = max_len
        self.item_emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD].zero_()
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _self_attend(self, hist_ids):
        B, L = hist_ids.shape
        mask = (hist_ids != PAD)  # (B, L) True=real item
        positions = torch.arange(L, device=hist_ids.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(hist_ids) + self.pos_emb(positions)  # (B, L, d)

        H, dh = self.n_heads, self.d // self.n_heads
        q = self.q_proj(x).view(B, L, H, dh).transpose(1, 2)  # (B,H,L,dh)
        k = self.k_proj(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, dh).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) / (dh ** 0.5)  # (B,H,L,L)
        key_mask = mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,L), broadcasts over query dim
        logits = logits.masked_fill(~key_mask, -1e9)  # finite, NaN-safe even if a row is all-pad
        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)
        ctx = torch.matmul(attn, v)  # (B,H,L,dh)
        ctx = ctx.transpose(1, 2).reshape(B, L, self.d)
        enc = self.ln(x + self.out_proj(ctx))  # (B,L,d), residual + LN

        m = mask.unsqueeze(-1).float()
        denom = m.sum(1).clamp(min=1.0)
        pooled = (enc * m).sum(1) / denom  # (B,d); zero vector for hist_len==0 rows
        return pooled

    def score(self, hist_ids, target_ids):
        pooled = self._self_attend(hist_ids)
        tgt = self.item_emb(target_ids)
        return (pooled * tgt).sum(-1)

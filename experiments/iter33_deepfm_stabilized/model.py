"""iter33: DeepFM-style model -- byte-for-byte copy of iter28_deepfm_refined_features/
model.py (itself a verbatim copy of iter26_deepfm/model.py), with ONE addition:
an `init_scale_mult` constructor kwarg (default 1.0, i.e. a no-op that exactly
reproduces iter28/iter26's original init) that multiplies the MLP's He/linear-
readout init scales, following iter30_deepfm_variance_reduction/model.py's
`DeepFMVR` mechanism exactly (same formula: `scale = base_scale * init_scale_mult`).
iter30 found `init_scale_mult=0.5` was the only lever of four tested that
reduced deep_h32's seed-to-seed variance on BOTH valid and test simultaneously
without hurting the mean -- this file lets that lever be applied to iter28's
setup (DeepFM on iter24's refined feature set) instead of iter30's own
(DeepFM on iter19's older feature set). Everything else below (forward/
backward math, Adam step) is unmodified from iter26/iter28.

baseline.FM's linear (W) + pairwise (V) terms
reused UNMODIFIED, plus a small hand-rolled MLP ("deep part") consuming the
same per-field embeddings E (already computed inside FM.logits), added to the
FM logit before the sigmoid/BPR loss.

Design decision on gradient flow (documented explicitly, see RESULT.md):
the MLP's backward pass DOES compute dL/dE (gradient w.r.t. the flattened
embedding input), but this gradient is NOT scattered back into V. V and W
receive ONLY the standard FM-part gradient, byte-for-byte identical to
baseline.FM / iter19's bpr_step. This directly satisfies the dispatch
instruction ("keep V/W exactly as-is, computed identically to baseline.FM")
and means enabling/disabling the deep part cannot alter V/W's gradient
formula -- only the forward logit and the MLP's own parameters change. This
is a deliberate stability choice: it isolates the "add a second scoring head
on top of the already-tuned FM embedding" axis without letting a fresh,
randomly-initialized deep tower perturb gradients into the embedding that
iter19 already tuned over 5 seeds.

Forward pass (per row):
    E = V[X]                        (F, k)   -- F fields, k = embed dim
    S = E.sum(axis=0)                (k,)
    z_fm = b + sum(W[X]) + 0.5*((S**2).sum() - (E**2).sum())     [unchanged FM]
    x0 = E.flatten()                 (F*k,)   -- deep part's input
    h_1 = relu(x0 @ W1 + b1)         (H1,)
    h_2 = relu(h_1 @ W2 + b2)        (H2,)    [only if 2 hidden layers]
    z_deep = h_last @ W_out + b_out  scalar   (last "layer" is a plain linear
                                                readout, no activation)
    z = z_fm + z_deep

Backward pass (per batch, standard MLP backprop derived by hand):
  given dL/dz (upstream gradient, identical scalar for z_fm and z_deep since
  z = z_fm + z_deep is a plain sum):
    dL/dW_out = h_last^T @ dL/dz ;  dL/db_out = sum(dL/dz)
    dL/dh_last = dL/dz @ W_out^T
    for each hidden layer i counting backward:
        dL/dz_i = dL/dh_i * 1[z_i > 0]                 (relu mask)
        dL/dW_i = h_{i-1}^T @ dL/dz_i ;  dL/db_i = sum(dL/dz_i, axis=0)
        dL/dh_{i-1} = dL/dz_i @ W_i^T
  (dL/dx0, i.e. dL/dh_0, is also computed by this recursion for
  completeness/inspection, but per the design decision above it is discarded
  -- not added into gV.)

Adam optimizer for the MLP's own parameters mirrors baseline.FM's Adam
exactly (b1=0.9, b2=0.999, eps=1e-8), as a separate parameter group with its
own moment buffers and its own step counter, added alongside the V/W update
in bpr_step (see train.py's deepfm_bpr_step).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from baseline import FM, sigmoid  # noqa: E402


class DeepFM(FM):
    def __init__(self, dim, num_fields, k=16, hidden=(32,), lr=0.001, l2=1e-6,
                 mlp_lr=None, l2_mlp=1e-5, use_deep=True, seed=0, init_scale_mult=1.0):
        super().__init__(dim, k=k, lr=lr, l2=l2, seed=seed)
        self.num_fields = num_fields
        self.hidden = list(hidden)
        self.use_deep = use_deep
        self.mlp_lr = float(mlp_lr) if mlp_lr is not None else lr
        self.l2_mlp = l2_mlp
        self.init_scale_mult = float(init_scale_mult)

        # Separate RNG stream for MLP init so V's random draws stay bit-identical
        # to baseline.FM / iter19's FM for the same seed (verified in the
        # use_deep=False harness-fidelity check).
        rng = np.random.default_rng(seed + 12345)
        dims = [num_fields * k] + self.hidden + [1]
        self.mlp_W, self.mlp_b = [], []
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            # He init (ReLU) for hidden layers; smaller-scale init for the
            # final linear readout to keep the deep branch's initial output
            # near 0 (a fresh untuned MLP head near a well-tuned FM part
            # should start as close to a no-op as possible).
            is_last = (i == len(dims) - 2)
            base_scale = np.sqrt(1.0 / fan_in) if is_last else np.sqrt(2.0 / fan_in)
            # iter30's variance-reduction lever: multiply the init scale by
            # init_scale_mult (default 1.0 -- exact no-op, reproduces iter28/
            # iter26 bit-for-bit since multiplying by 1.0 is exact for floats).
            scale = base_scale * self.init_scale_mult
            W = rng.normal(0, scale, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self.mlp_W.append(W)
            self.mlp_b.append(b)

        self.mlp_mW = [np.zeros_like(W) for W in self.mlp_W]
        self.mlp_vW = [np.zeros_like(W) for W in self.mlp_W]
        self.mlp_mb = [np.zeros_like(b) for b in self.mlp_b]
        self.mlp_vb = [np.zeros_like(b) for b in self.mlp_b]
        self.mlp_t = 0

    def deep_forward(self, E):
        """E: (N, F, k). Returns (out (N,), inputs list, preacts list)."""
        N = E.shape[0]
        if not self.use_deep:
            return np.zeros(N, dtype=np.float32), None, None
        x = E.reshape(N, -1).astype(np.float32)
        inputs = [x]
        preacts = []
        h = x
        L = len(self.mlp_W)
        for i in range(L - 1):
            z = h @ self.mlp_W[i] + self.mlp_b[i]
            preacts.append(z)
            h = np.maximum(z, 0).astype(np.float32)
            inputs.append(h)
        out = h @ self.mlp_W[-1] + self.mlp_b[-1]     # (N,1) linear readout
        return out[:, 0], inputs, preacts

    def deep_backward(self, g, inputs, preacts):
        """g: (N,) dL/d(deep_out). Returns (gW list, gb list, dE_flat (N,F*k))."""
        L = len(self.mlp_W)
        dh = g[:, None].astype(np.float32)
        gW = [None] * L
        gb = [None] * L
        gW[L - 1] = inputs[L - 1].T @ dh
        gb[L - 1] = dh.sum(0)
        d_in = dh @ self.mlp_W[L - 1].T
        for i in range(L - 2, -1, -1):
            mask = (preacts[i] > 0).astype(np.float32)
            dz = d_in * mask
            gW[i] = inputs[i].T @ dz
            gb[i] = dz.sum(0)
            d_in = dz @ self.mlp_W[i].T
        return gW, gb, d_in   # d_in ends up == dL/dx0 (discarded by caller, see module docstring)

    def mlp_adam_step(self, gW, gb):
        self.mlp_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.mlp_W)):
            gWi = gW[i] + self.l2_mlp * self.mlp_W[i]
            gbi = gb[i]
            for P, G, M, Vv in ((self.mlp_W[i], gWi, self.mlp_mW[i], self.mlp_vW[i]),
                                 (self.mlp_b[i], gbi, self.mlp_mb[i], self.mlp_vb[i])):
                M *= b1; M += (1 - b1) * G
                Vv *= b2; Vv += (1 - b2) * (G * G)
                P -= self.mlp_lr * (M / (1 - b1 ** self.mlp_t)) / (np.sqrt(Vv / (1 - b2 ** self.mlp_t)) + eps)

    def logits_full(self, X):
        """Full DeepFM logit: FM part (unchanged) + deep part. Returns
        (z, E, S, deep_inputs, deep_preacts) -- the last two are needed only
        for the backward pass and are None when use_deep=False."""
        z_fm, E, S = self.logits(X)          # baseline.FM.logits, unmodified
        z_deep, inputs, preacts = self.deep_forward(E)
        return z_fm + z_deep, E, S, inputs, preacts

    def predict(self, X, bs=200_000):
        out = []
        for i in range(0, len(X), bs):
            z, *_ = self.logits_full(X[i:i + bs])
            out.append(z)
        return np.concatenate(out)

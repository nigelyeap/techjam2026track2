"""iter30: variance-reduction study on iter26's DeepFM deep part.

This file does NOT re-derive any forward/backward math. It imports
`DeepFM` from `experiments/iter26_deepfm/model.py` unmodified and
subclasses it with two extra, purely-constructor-level knobs needed for
levers 3 and 4 of the variance-reduction study (see this dir's RESULT.md):

  1. `init_scale_mult` -- multiplies iter26's existing init scales
     (He-scaled `sqrt(2/fan_in)` for hidden layers, `sqrt(1/fan_in)` for
     the linear readout) by a constant < 1, so the deep part can start
     even closer to a no-op than iter26's default. `init_scale_mult=1.0`
     exactly reproduces iter26's init (verified below).
  2. `mlp_seed` -- decouples the RNG seed used for the deep part's own
     weight init from `seed` (which iter26 hard-wired as `seed + 12345`).
     `seed` still controls the FM part's V/W init (via `FM.__init__`,
     unchanged) AND the BPR sampling order (via train.py's `rng =
     np.random.default_rng(seed)`, unchanged) -- only the MLP's initial
     weights are decoupled. This is what lever 4 (ensembling multiple
     deep-part inits at fixed data/FM seed) needs.

`deep_forward`, `deep_backward`, `mlp_adam_step`, `logits_full`, and
`predict` are all inherited UNMODIFIED from iter26_deepfm.model.DeepFM.
Only `__init__` is overridden, and only to redraw `mlp_W`/`mlp_b`/the
Adam moment buffers with a possibly-different seed and/or scale after
iter26's own `__init__` has already set up the FM part identically to
iter19/iter26.
"""
import os, sys
import importlib.util
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, abs_path):
    # NOTE: iter26_deepfm/model.py and this file share the basename "model.py".
    # Loading via importlib under a private module name (instead of a bare
    # `sys.path.insert` + `import model`) avoids a sys.modules['model']
    # collision between the two same-named files.
    spec = importlib.util.spec_from_file_location(name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter26_model = _load_module('iter26_deepfm_model', os.path.join(_THIS_DIR, '..', 'iter26_deepfm', 'model.py'))
DeepFM26 = _iter26_model.DeepFM  # unmodified iter26 forward/backward math


class DeepFMVR(DeepFM26):
    def __init__(self, dim, num_fields, k=16, hidden=(32,), lr=0.001, l2=1e-6,
                 mlp_lr=None, l2_mlp=1e-5, use_deep=True, seed=0,
                 init_scale_mult=1.0, mlp_seed=None):
        # Sets up FM part (V/W/b, Adam buffers) exactly as iter26/iter19, plus
        # a default mlp init at seed+12345 / scale_mult=1.0 that we may
        # immediately overwrite below.
        super().__init__(dim, num_fields, k=k, hidden=hidden, lr=lr, l2=l2,
                          mlp_lr=mlp_lr, l2_mlp=l2_mlp, use_deep=use_deep, seed=seed)
        self.init_scale_mult = float(init_scale_mult)
        self.mlp_seed = mlp_seed
        if init_scale_mult != 1.0 or mlp_seed is not None:
            eff_seed = mlp_seed if mlp_seed is not None else (seed + 12345)
            rng = np.random.default_rng(eff_seed)
            dims = [num_fields * k] + list(self.hidden) + [1]
            self.mlp_W, self.mlp_b = [], []
            for i in range(len(dims) - 1):
                fan_in, fan_out = dims[i], dims[i + 1]
                is_last = (i == len(dims) - 2)
                base_scale = np.sqrt(1.0 / fan_in) if is_last else np.sqrt(2.0 / fan_in)
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

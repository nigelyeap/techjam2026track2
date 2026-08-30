"""iter17: thin re-export of iter9_history_dense/data_ext.py's feature
computation (compute_causal_features / load_ext / encode_ext / BASE_FIELDS),
completely unmodified. This iteration only changes BPR negative sampling
(see train.py); the feature set is exactly iter9's {activity,tab,rate}.
"""
import os, importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_iter9_path = os.path.join(_here, '..', 'iter9_history_dense', 'data_ext.py')
_spec = importlib.util.spec_from_file_location('_iter9_data_ext', _iter9_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_causal_features = _mod.compute_causal_features
load_ext = _mod.load_ext
encode_ext = _mod.encode_ext
BASE_FIELDS = _mod.BASE_FIELDS
ALPHA = _mod.ALPHA

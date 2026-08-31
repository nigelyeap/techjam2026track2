"""Thin, immutable reuse of the promoted YIXI10 feature representation."""

from __future__ import annotations

import importlib.util
import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SOURCE_PATH = os.path.join(
    REPO_ROOT, "experiments", "iterYIXI10_video_metadata", "features.py"
)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = load_module(SOURCE_PATH, "yixi10_features_for_yixi11")
load_frames = source.load_frames


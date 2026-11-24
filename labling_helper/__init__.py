"""Labling Helper package for Medanta Hospital Counter Monitor.
Automates dataset annotation for AnyLabeling.
"""

from .auto_labeler import AutoLabeler, auto_label_dataset
from .config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    STATIC_OBJECT_ANCHORS,
)

__all__ = [
    "AutoLabeler",
    "auto_label_dataset",
    "DEFAULT_INPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "STATIC_OBJECT_ANCHORS",
]

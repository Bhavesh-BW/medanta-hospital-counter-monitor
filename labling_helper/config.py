"""Configuration and default parameters for Auto-Labeling.
"""

from pathlib import Path

# Paths to models
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_COCO_MODEL = ROOT_DIR / "backend" / "models" / "yolo11s.pt"
DEFAULT_PHONE_MODEL = ROOT_DIR / "backend" / "models" / "best.pt"

# Default directories
DEFAULT_INPUT_DIR = ROOT_DIR / "backend" / "training_samples"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "labbled_images"

# Static counter objects (coordinates based on fixed CCTV camera perspective)
# Format: [xmin, ymin, xmax, ymax]
STATIC_OBJECT_ANCHORS = [
    {
        "label": "table_phone",
        "box": [682.5, 1212.7, 877.6, 1397.0],
        "description": "Reception desk landline telephone",
    },
    {
        "label": "cash_machine",
        "box": [880.0, 1220.0, 1005.0, 1340.0],
        "description": "Counter 1 swipe / POS card machine",
    },
    {
        "label": "cash_machine",
        "box": [1525.0, 960.0, 1580.0, 1065.0],
        "description": "Counter 2 swipe / POS card machine",
    },
]

# Staff counter zones (pixel bounds to filter out waiting room / hallway people)
# Using bounding box center_x divider at x=1550
COUNTER_1_ZONE = {
    "name": "Receptionist 1 (Left Desk)",
    "center_x_max": 1550,
    "y_bottom_min": 950,
    "default_label": "not_using_phone",
}

COUNTER_2_ZONE = {
    "name": "Receptionist 2 (Right Desk)",
    "center_x_min": 1550,
    "y_bottom_min": 950,
    "default_label": "not_using_phone",
}

# AnyLabeling JSON metadata
ANYLABELING_VERSION = "0.4.43"

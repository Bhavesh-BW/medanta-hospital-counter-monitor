"""Dataset converter: Converts AnyLabeling JSON annotations to standard YOLO format.
Creates a ready-to-train dataset with train/val split and a zip archive for Google Colab.
"""

import json
import logging
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("labling_helper.export")

# Standard class mapping
CLASS_MAP = {
    "using_phone": 0,
    "not_using_phone": 1,
    "table_phone": 2,
    "cash_machine": 3,
}

# Label normalization (handling case variations)
LABEL_ALIASES = {
    "table_phone": "table_phone",
    "Table_phone": "table_phone",
    "cash_machine": "cash_machine",
    "using_phone": "using_phone",
    "not_using_phone": "not_using_phone",
}


def convert_box_to_yolo(
    points: List[List[float]], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert [[x1, y1], [x2, y2]] into normalized (x_center, y_center, w, h)."""
    (x1, y1), (x2, y2) = points[0], points[1]
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)

    # Clamp to image boundaries
    xmin = max(0.0, min(float(img_w), xmin))
    xmax = max(0.0, min(float(img_w), xmax))
    ymin = max(0.0, min(float(img_h), ymin))
    ymax = max(0.0, min(float(img_h), ymax))

    w = xmax - xmin
    h = ymax - ymin
    x_center = xmin + w / 2.0
    y_center = ymin + h / 2.0

    # Normalize by image width and height
    return (
        x_center / img_w,
        y_center / img_h,
        w / img_w,
        h / img_h,
    )


def export_to_yolo(
    labeled_dir: str = "labbled_images",
    output_dataset_dir: str = "dataset",
    val_split: float = 0.15,
    seed: int = 42,
    create_zip: bool = True,
) -> Dict[str, any]:
    """Converts labbled_images into a YOLO dataset directory and dataset.zip."""
    labeled_path = Path(labeled_dir).resolve()
    dataset_path = Path(output_dataset_dir).resolve()

    if not labeled_path.exists():
        raise FileNotFoundError(f"Source folder '{labeled_path}' not found.")

    # Prepare dataset directory structure
    for split in ["train", "val"]:
        (dataset_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collect all image and JSON pairs
    json_files = sorted(labeled_path.glob("*.json"))
    valid_pairs = []

    for jf in json_files:
        stem = jf.stem
        # Look for matching image
        img_candidates = list(labeled_path.glob(f"{stem}.*"))
        img_file = next(
            (p for p in img_candidates if p.suffix.lower() in [".jpg", ".jpeg", ".png"]),
            None,
        )
        if img_file and img_file.exists():
            valid_pairs.append((img_file, jf))

    total_samples = len(valid_pairs)
    if total_samples == 0:
        raise ValueError(f"No matching image + JSON pairs found in {labeled_path}")

    # Random split
    random.seed(seed)
    shuffled_pairs = valid_pairs.copy()
    random.shuffle(shuffled_pairs)

    val_count = max(1, int(total_samples * val_split))
    val_pairs = set(shuffled_pairs[:val_count])

    counts = {"train": 0, "val": 0, "class_counts": {k: 0 for k in CLASS_MAP}}

    for img_file, json_file in valid_pairs:
        split = "val" if (img_file, json_file) in val_pairs else "train"
        counts[split] += 1

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data.get("imageWidth") or 2560
        img_h = data.get("imageHeight") or 1440

        yolo_lines = []
        for s in data.get("shapes", []):
            raw_label = s.get("label", "")
            norm_label = LABEL_ALIASES.get(raw_label)
            if not norm_label or norm_label not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[norm_label]
            counts["class_counts"][norm_label] += 1

            pts = s.get("points", [])
            if len(pts) >= 2:
                xc, yc, w, h = convert_box_to_yolo(pts, img_w, img_h)
                # Keep within 6 decimal precision
                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # Destination paths
        dst_img = dataset_path / "images" / split / img_file.name
        dst_txt = dataset_path / "labels" / split / f"{img_file.stem}.txt"

        shutil.copy2(img_file, dst_img)
        with open(dst_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

    # Generate data.yaml
    yaml_content = f"""# Medanta Hospital Counter & Phone Monitoring Dataset
path: /content/dataset
train: images/train
val: images/val

names:
  0: using_phone
  1: not_using_phone
  2: table_phone
  3: cash_machine
"""
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Optional zip creation for easy Colab upload (including existing best.pt)
    zip_path = None
    if create_zip:
        zip_file = labeled_path.parent / "dataset.zip"
        best_pt_path = labeled_path.parent / "backend" / "models" / "best.pt"
        print(f"Creating zip file: {zip_file} ...")
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(dataset_path)
                    zf.write(full_p, f"dataset/{rel_p.as_posix()}")
            if best_pt_path.exists():
                print(f"Adding existing best.pt ({best_pt_path}) to zip...")
                zf.write(best_pt_path, "dataset/best.pt")
                zf.write(best_pt_path, "best.pt")
        zip_path = str(zip_file)

    summary = {
        "total_samples": total_samples,
        "train_samples": counts["train"],
        "val_samples": counts["val"],
        "class_counts": counts["class_counts"],
        "dataset_dir": str(dataset_path),
        "zip_path": zip_path,
    }
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = export_to_yolo()
    print("Export Summary:", res)

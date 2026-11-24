"""Automated labeling module for hospital counter CCTV images.
Generates AnyLabeling compatible JSON annotation files.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from ultralytics import YOLO

from .config import (
    ANYLABELING_VERSION,
    COUNTER_1_ZONE,
    COUNTER_2_ZONE,
    DEFAULT_COCO_MODEL,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PHONE_MODEL,
    STATIC_OBJECT_ANCHORS,
)

logger = logging.getLogger("labling_helper")


class AutoLabeler:
    """Auto-labeling engine combining fixed geometry anchors with YOLO detections."""

    def __init__(
        self,
        coco_model_path: Optional[Path] = None,
        phone_model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # 1. Person detector
        coco_path = coco_model_path or DEFAULT_COCO_MODEL
        if not coco_path.exists():
            # Fallback to yolo11n.pt if yolo11s.pt not found
            alt_path = coco_path.parent / "yolo11n.pt"
            if alt_path.exists():
                coco_path = alt_path

        logger.info(f"Loading person detector from: {coco_path} on {self.device}")
        self.person_model = YOLO(str(coco_path))

        # 2. Phone detector
        phone_path = phone_model_path or DEFAULT_PHONE_MODEL
        if phone_path.exists():
            logger.info(f"Loading phone detector from: {phone_path} on {self.device}")
            self.phone_model = YOLO(str(phone_path))
        else:
            logger.warning(f"Phone detector {phone_path} not found. Using heuristic.")
            self.phone_model = None

    def _create_shape_dict(
        self, label: str, points: List[List[float]], shape_type: str = "rectangle"
    ) -> Dict[str, Any]:
        """Format shape dict according to AnyLabeling specification."""
        return {
            "label": label,
            "text": "",
            "points": points,
            "group_id": None,
            "shape_type": shape_type,
            "flags": {},
        }

    def detect_static_objects(self) -> List[Dict[str, Any]]:
        """Generate annotations for fixed counter objects (table phone, cash machines)."""
        shapes = []
        for obj in STATIC_OBJECT_ANCHORS:
            xmin, ymin, xmax, ymax = obj["box"]
            shapes.append(
                self._create_shape_dict(
                    label=obj["label"],
                    points=[[round(xmin, 2), round(ymin, 2)], [round(xmax, 2), round(ymax, 2)]],
                )
            )
        return shapes

    def detect_staff_people(
        self, img_path: Path, image: cv2.Mat
    ) -> List[Dict[str, Any]]:
        """Detect receptionists sitting at Counter 1 and Counter 2 and classify phone usage."""
        shapes = []
        h, w = image.shape[:2]

        # Run YOLO COCO person detector (class 0)
        results = self.person_model.predict(
            str(img_path),
            classes=[0],
            conf=0.40,
            verbose=False,
            device=self.device,
        )

        detected_staff_1 = None
        detected_staff_2 = None

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # Filter: must be sitting at counter (bottom of bbox near desk bottom)
                if y2 < COUNTER_1_ZONE["y_bottom_min"] and y2 < COUNTER_2_ZONE["y_bottom_min"]:
                    continue

                cx = (x1 + x2) / 2.0

                # Counter 1 zone (left receptionist: center_x < 1550)
                if cx < COUNTER_1_ZONE["center_x_max"]:
                    if detected_staff_1 is None or conf > detected_staff_1["conf"]:
                        detected_staff_1 = {"box": [x1, y1, x2, y2], "conf": conf}

                # Counter 2 zone (right receptionist: center_x >= 1550)
                else:
                    if detected_staff_2 is None or conf > detected_staff_2["conf"]:
                        detected_staff_2 = {"box": [x1, y1, x2, y2], "conf": conf}

        # Classify Staff 1 (Left Desk)
        if detected_staff_1:
            x1, y1, x2, y2 = detected_staff_1["box"]
            lbl = self._classify_phone_usage(image, [x1, y1, x2, y2], default="not_using_phone")
            shapes.append(
                self._create_shape_dict(
                    label=lbl,
                    points=[[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
                )
            )

        # Classify Staff 2 (Right Desk)
        if detected_staff_2:
            x1, y1, x2, y2 = detected_staff_2["box"]
            lbl = self._classify_phone_usage(image, [x1, y1, x2, y2], default="using_phone")
            shapes.append(
                self._create_shape_dict(
                    label=lbl,
                    points=[[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
                )
            )

        return shapes

    def _classify_phone_usage(
        self, image: cv2.Mat, box: List[float], default: str
    ) -> str:
        """Classify person box as using_phone or not_using_phone using best.pt or default."""
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = image.shape[:2]
        crop = image[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]

        if crop.size == 0 or self.phone_model is None:
            return default

        # If phone model is available, check detection
        try:
            res = self.phone_model.predict(crop, conf=0.45, verbose=False, device=self.device)
            for r in res:
                if r.boxes is not None and len(r.boxes) > 0:
                    cls_idx = int(r.boxes.cls[0])
                    cls_name = self.phone_model.names.get(cls_idx, "")
                    if cls_name == "using_phone":
                        return "using_phone"
        except Exception:
            pass

        return default

    def label_image(self, img_path: Path) -> Dict[str, Any]:
        """Generate complete AnyLabeling annotation dict for a single image."""
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")

        h, w = image.shape[:2]

        shapes = []
        # 1. Fixed static objects
        shapes.extend(self.detect_static_objects())

        # 2. Dynamic counter staff
        shapes.extend(self.detect_staff_people(img_path, image))

        return {
            "version": ANYLABELING_VERSION,
            "flags": {},
            "shapes": shapes,
            "imagePath": img_path.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
            "text": "",
        }

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        skip_existing: bool = True,
        copy_images: bool = True,
    ) -> Dict[str, Any]:
        """Process all images in input_dir, outputting JSON annotations and images to output_dir.

        If skip_existing is True:
          - If a matching .json already exists in input_dir, it is treated as already
            labeled (e.g. human verified). It is copied to output_dir without re-running AI.
          - If a .json already exists in output_dir, it is skipped.
        """
        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted(
            [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
        )

        total_images = len(image_files)
        skipped_existing = 0
        newly_labeled = 0
        errors = 0

        logger.info(f"Found {total_images} images in {input_dir}")
        logger.info(f"Target output directory: {output_dir}")

        for idx, img_path in enumerate(image_files, 1):
            stem = img_path.stem
            out_json = output_dir / f"{stem}.json"
            out_img = output_dir / img_path.name
            src_json = input_dir / f"{stem}.json"

            # 1. Always preserve human-labeled annotations from input_dir
            if src_json.exists():
                logger.info(f"[{idx}/{total_images}] Preserving existing human-labeled: {img_path.name}")
                shutil.copy2(src_json, out_json)
                if copy_images and not out_img.exists():
                    shutil.copy2(img_path, out_img)
                skipped_existing += 1
                continue

            # 2. Check if output JSON already exists
            if skip_existing and out_json.exists():
                logger.info(f"[{idx}/{total_images}] Skipping already generated: {out_json.name}")
                if copy_images and not out_img.exists():
                    shutil.copy2(img_path, out_img)
                skipped_existing += 1
                continue

            # Auto-label new image
            try:
                annotation = self.label_image(img_path)
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(annotation, f, indent=2)

                if copy_images and not out_img.exists():
                    shutil.copy2(img_path, out_img)

                newly_labeled += 1
                logger.info(
                    f"[{idx}/{total_images}] Auto-labeled: {img_path.name} "
                    f"({len(annotation['shapes'])} objects)"
                )
            except Exception as e:
                logger.error(f"[{idx}/{total_images}] Error processing {img_path.name}: {e}")
                errors += 1

        summary = {
            "total_images": total_images,
            "skipped_existing": skipped_existing,
            "newly_labeled": newly_labeled,
            "errors": errors,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        }
        logger.info(f"Processing Complete! Summary: {summary}")
        return summary


def auto_label_dataset(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    skip_existing: bool = True,
    copy_images: bool = True,
) -> Dict[str, Any]:
    """Convenience function to run auto-labeling pipeline on a dataset folder."""
    in_dir = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    labeler = AutoLabeler()
    return labeler.process_directory(
        input_dir=in_dir,
        output_dir=out_dir,
        skip_existing=skip_existing,
        copy_images=copy_images,
    )

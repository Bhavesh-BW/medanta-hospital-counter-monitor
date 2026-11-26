"""Customer Anger Detection Plugin.

Self-contained, pluggable module for Medanta Hospital Counter Monitor:
1. Runs YuNet Face Detector (OpenCV DNN FaceDetectorYN) on customer queue persons only.
2. Evaluates facial expressions using AffectNet-7 (DDAMFN) TorchScript model.
3. Flags anger when anger score exceeds threshold or is dominant expression.
4. Gracefully degrades / self-disables if model weights are not present.
"""

import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger("customer_anger_detector")

EMOTION_LABELS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]


class CustomerAngerDetector:
    """Pluggable detector for customer anger in counter queue zones."""

    def __init__(
        self,
        yunet_path: Optional[os.PathLike] = None,
        fer_path: Optional[os.PathLike] = None,
        anger_threshold: Optional[float] = None,
        face_confidence_threshold: float = 0.55,
        enabled: Optional[bool] = None,
    ):
        if enabled is None:
            self.enabled = os.environ.get("ENABLE_ANGER_DETECTION", "true").lower() in ("true", "1", "yes")
        else:
            self.enabled = bool(enabled)

        if anger_threshold is None:
            self.anger_threshold = float(os.environ.get("ANGER_CONFIDENCE_THRESHOLD", "0.45"))
        else:
            self.anger_threshold = float(anger_threshold)

        self.face_confidence_threshold = face_confidence_threshold
        self.yunet_path = pathlib.Path(yunet_path) if yunet_path else None
        self.fer_path = pathlib.Path(fer_path) if fer_path else None

        self._face_detector = None
        self._fer_model = None
        self._is_ready = False

        if self.enabled:
            self._load_models()

    def _load_models(self) -> None:
        """Loads YuNet and FER TorchScript models safely."""
        # 1. Load YuNet Face Detector
        if self.yunet_path and self.yunet_path.exists():
            try:
                self._face_detector = cv2.FaceDetectorYN.create(
                    model=str(self.yunet_path),
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.face_confidence_threshold,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                logger.info("✅ CustomerAngerDetector: YuNet face detector loaded.")
            except Exception as e:
                logger.warning("⚠️ Failed to load YuNet face detector: %s", e)
                self._face_detector = None
        else:
            logger.info("Notice: YuNet model not found at %s. Anger detection disabled.", self.yunet_path)

        # 2. Load AffectNet-7 (DDAMFN) TorchScript Model
        if self.fer_path and self.fer_path.exists():
            try:
                self._fer_model = torch.jit.load(str(self.fer_path), map_location="cpu")
                self._fer_model.eval()
                logger.info("✅ CustomerAngerDetector: AffectNet-7 FER model loaded.")
            except Exception as e:
                logger.warning("⚠️ Failed to load FER model: %s", e)
                self._fer_model = None
        else:
            logger.info("Notice: FER model not found at %s. Anger detection disabled.", self.fer_path)

        self._is_ready = (self._face_detector is not None) and (self._fer_model is not None)
        if self._is_ready:
            print(f"[CustomerAngerDetector] Initialized successfully (Anger threshold: {self.anger_threshold:.2f})")
        else:
            print("[CustomerAngerDetector] Standby: Required models not fully available.")

    @property
    def is_ready(self) -> bool:
        """Returns True if both models are loaded and ready for inference."""
        return self.enabled and self._is_ready

    def analyze_person_crop(
        self, frame: np.ndarray, bbox: List[int]
    ) -> Optional[Dict[str, Any]]:
        """Analyzes upper-body of a person bbox to detect face and check for anger.

        Args:
            frame: Full BGR frame (H, W, 3).
            bbox: Person bounding box [x1, y1, x2, y2].

        Returns:
            Dictionary with anger detection result, or None if no face detected.
        """
        if not self.is_ready or frame is None:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        pw = x2 - x1
        ph = y2 - y1
        if pw < 25 or ph < 35:
            return None

        # Search face in upper 70% of person bounding box
        face_search_h = max(20, int(ph * 0.70))
        upper_crop = frame[y1 : min(h, y1 + face_search_h), x1:x2]
        ch, cw = upper_crop.shape[:2]
        if ch < 20 or cw < 20:
            return None

        # Detect face with YuNet
        try:
            self._face_detector.setInputSize((cw, ch))
            _, faces = self._face_detector.detect(upper_crop)
        except Exception as e:
            logger.debug("YuNet face detection exception: %s", e)
            return None

        if faces is None or len(faces) == 0:
            return None

        # Pick face with highest confidence
        best_face = max(faces, key=lambda f: float(f[-1]))
        conf = float(best_face[-1])
        if conf < self.face_confidence_threshold:
            return None

        fx, fy, fw, fh = best_face[0:4].astype(int)
        fx1 = max(0, x1 + fx)
        fy1 = max(0, y1 + fy)
        fx2 = min(w, x1 + fx + fw)
        fy2 = min(h, y1 + fy + fh)

        face_crop = upper_crop[max(0, fy) : min(ch, fy + fh), max(0, fx) : min(cw, fx + fw)]
        if face_crop.size == 0 or face_crop.shape[0] < 12 or face_crop.shape[1] < 12:
            return None

        # Preprocess for AffectNet-7 (112x112, normalized)
        try:
            resized = cv2.resize(face_crop, (112, 112))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
            tensor = (tensor / 255.0 - 0.5) / 0.5
            tensor = tensor.unsqueeze(0)

            with torch.no_grad():
                logits = self._fer_model(tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            emotion_scores = {
                EMOTION_LABELS[i]: round(float(probs[i]), 3)
                for i in range(len(EMOTION_LABELS))
            }
            dominant_idx = int(np.argmax(probs))
            dominant_label = EMOTION_LABELS[dominant_idx]
            dominant_conf = round(float(probs[dominant_idx]), 3)

            anger_score = emotion_scores.get("anger", 0.0)

            # Strictly detect only the angry emotion
            is_anger = (dominant_label == "anger") or (anger_score >= self.anger_threshold)

            return {
                "detected": True,
                "face_bbox": [fx1, fy1, fx2, fy2],
                "face_confidence": round(conf, 3),
                "face_crop": face_crop,
                "expression": dominant_label,
                "expression_confidence": dominant_conf,
                "anger_score": anger_score,
                "is_anger": is_anger,
                "emotions": emotion_scores,
            }
        except Exception as e:
            logger.warning("Facial emotion evaluation failed: %s", e)
            return None

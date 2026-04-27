"""
Project LUNA — Object Detection Module
═══════════════════════════════════════
Uses Ultralytics YOLOv8 for real-time object detection.
Gracefully falls back to an empty-result placeholder when the
model or library is unavailable.

Features:
  • Threadsafe YOLO inference
  • Learned-object integration (user-taught labels from storage)
  • Detection deduplication (merge overlapping bounding boxes)
  • Structured logging with per-frame detection counts
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from config import YOLO_CONFIDENCE, YOLO_MODEL, LEARNED_OBJECTS_FILE, BASE_DIR

logger = logging.getLogger("luna.object_detection")


class ObjectDetector:
    """YOLOv8-based object detection with learned-object support."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_lock = threading.Lock()
        self.learned_objects: dict[str, dict] = {}
        self._load_model()
        self._load_learned_objects()

    # ── Model Management ────────────────────────

    def _load_model(self) -> None:
        """Load the YOLOv8 model, trying several candidate paths."""
        try:
            from ultralytics import YOLO

            # Try: backend/yolov8n.pt → project-root/yolov8n.pt → auto-download
            candidates: list[Path] = [
                BASE_DIR / YOLO_MODEL,
                BASE_DIR.parent / YOLO_MODEL,
            ]

            model_path: str = YOLO_MODEL  # fallback: let ultralytics download
            for candidate in candidates:
                if candidate.exists():
                    model_path = str(candidate)
                    logger.info("Found YOLO model at: %s", model_path)
                    break

            self._model = YOLO(model_path)
            logger.info("✅ YOLOv8 model loaded: %s (conf threshold: %.2f)", model_path, YOLO_CONFIDENCE)

        except ImportError:
            logger.warning("⚠️  ultralytics not installed — object detection disabled")
            logger.warning("   Install with: pip install ultralytics>=8.1.0")
        except Exception as exc:
            logger.warning("⚠️  Failed to load YOLO model: %s — object detection disabled", exc)

    @property
    def model(self) -> Any:
        """Public read-only access to the underlying YOLO model (or None)."""
        return self._model

    # ── Learned Objects ─────────────────────────

    def _load_learned_objects(self) -> None:
        """Load user-taught object labels from JSON storage."""
        path = Path(LEARNED_OBJECTS_FILE)
        if not path.exists():
            self.learned_objects = {}
            return
        try:
            with path.open("r", encoding="utf-8") as fh:
                self.learned_objects = json.load(fh)
            logger.info("📦 Loaded %d learned objects", len(self.learned_objects))
        except Exception as exc:
            logger.warning("Failed to load learned objects: %s", exc)
            self.learned_objects = {}

    def reload_learned_objects(self) -> None:
        """Hot-reload learned objects after the user teaches a new label."""
        self._load_learned_objects()

    # ── Detection ───────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run object detection on a single BGR frame.

        Returns a list of dicts, each containing:
            label        (str)   — object class name
            confidence   (float) — 0.0 – 1.0
            bbox         (list)  — [x1, y1, x2, y2]
            is_learned   (bool)  — True if user-taught
        """
        if self._model is None:
            return []

        try:
            with self._model_lock:
                results = self._model(frame, conf=YOLO_CONFIDENCE, verbose=False)

            detections: list[dict] = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                    confidence = float(box.conf[0])
                    bbox = [int(coord) for coord in box.xyxy[0].tolist()]

                    detections.append({
                        "label": label,
                        "confidence": round(confidence, 2),
                        "bbox": bbox,
                        "is_learned": False,
                    })

            # Deduplicate highly-overlapping boxes of the same class
            detections = self._deduplicate(detections)

            if detections:
                logger.debug(
                    "Detected %d object(s): %s",
                    len(detections),
                    ", ".join(d["label"] for d in detections),
                )

            return detections

        except Exception as exc:
            logger.error("Object detection error: %s", exc)
            return []

    # ── Helpers ──────────────────────────────────

    @staticmethod
    def _deduplicate(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
        """
        Remove duplicate detections of the same class by IoU overlap.
        Keeps the higher-confidence box when two boxes of the same label
        overlap more than *iou_threshold*.
        """
        if len(detections) <= 1:
            return detections

        keep: list[dict] = []
        used: set[int] = set()

        # Sort by confidence descending so we keep the best box first
        ranked = sorted(detections, key=lambda d: d["confidence"], reverse=True)

        for i, det_a in enumerate(ranked):
            if i in used:
                continue
            keep.append(det_a)
            for j in range(i + 1, len(ranked)):
                if j in used:
                    continue
                det_b = ranked[j]
                if det_a["label"] == det_b["label"]:
                    iou = ObjectDetector._compute_iou(det_a["bbox"], det_b["bbox"])
                    if iou >= iou_threshold:
                        used.add(j)

        return keep

    @staticmethod
    def _compute_iou(box_a: list[int], box_b: list[int]) -> float:
        """Compute Intersection over Union for two [x1, y1, x2, y2] boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

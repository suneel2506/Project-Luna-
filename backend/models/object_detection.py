"""
Project LUNA — Object Detection Module
Uses Ultralytics YOLOv8 for real-time object detection.
Falls back to placeholder if model is unavailable.
"""

import logging
import json
import os
import sys

logger = logging.getLogger(__name__)

# Add parent directory to path for config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YOLO_CONFIDENCE, YOLO_MODEL, LEARNED_OBJECTS_FILE


class ObjectDetector:
    """YOLOv8-based object detection with learned object support."""

    def __init__(self):
        self.model = None
        self.learned_objects = {}
        self._load_model()
        self._load_learned_objects()

    def _load_model(self):
        """Load YOLOv8 model. Falls back gracefully if unavailable."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(YOLO_MODEL)
            logger.info("✅ YOLOv8 model loaded successfully")
        except ImportError:
            logger.warning("⚠️ ultralytics not installed. Using placeholder detection.")
            self.model = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to load YOLO model: {e}. Using placeholder.")
            self.model = None

    def _load_learned_objects(self):
        """Load user-taught object labels from storage."""
        try:
            if os.path.exists(LEARNED_OBJECTS_FILE):
                with open(LEARNED_OBJECTS_FILE, "r") as f:
                    self.learned_objects = json.load(f)
                logger.info(f"📦 Loaded {len(self.learned_objects)} learned objects")
        except Exception as e:
            logger.warning(f"Failed to load learned objects: {e}")
            self.learned_objects = {}

    def detect(self, frame):
        """
        Detect objects in a frame.

        Args:
            frame: numpy array (BGR image from OpenCV)

        Returns:
            list of dicts with keys:
                - label (str): Object class name
                - confidence (float): Detection confidence 0-1
                - bbox (list): [x1, y1, x2, y2] bounding box
                - is_learned (bool): Whether this is a user-taught object
        """
        if self.model is None:
            return self._placeholder_detect(frame)

        try:
            results = self.model(frame, conf=YOLO_CONFIDENCE, verbose=False)
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    detections.append({
                        "label": label,
                        "confidence": round(confidence, 2),
                        "bbox": [int(b) for b in bbox],
                        "is_learned": False
                    })

            return detections

        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return self._placeholder_detect(frame)

    def _placeholder_detect(self, frame):
        """
        Placeholder when YOLO is unavailable.
        Returns empty list — no fake detections.
        """
        return []

    def reload_learned_objects(self):
        """Reload learned objects from storage (called after learning)."""
        self._load_learned_objects()

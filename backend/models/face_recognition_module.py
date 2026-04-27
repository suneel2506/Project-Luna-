"""
Project LUNA — Face Recognition Module (LBPH-based)
════════════════════════════════════════════════════
Detects faces via OpenCV Haar Cascade and recognizes them using
OpenCV's LBPH (Local Binary Pattern Histograms) face recognizer.

NO external dependencies beyond opencv-contrib-python.

Features:
  • LBPH-based face recognition — no dlib required
  • Face crops saved to storage/faces/images/{name}/ for persistence
  • Automatic LBPH model retraining when new faces are learned
  • Haar-cascade detection with minimum face-size filtering
  • Thread-safe known-faces access
  • Stale-pending cleanup with configurable TTL
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import (
    FACES_DIR,
    FACE_IMAGES_DIR,
    LBPH_MODEL_FILE,
    FACE_LABELS_FILE,
    LEARNED_FACES_FILE,
    LBPH_CONFIDENCE_THRESHOLD,
    PENDING_TTL_SECONDS,
)

logger = logging.getLogger("luna.face_recognition")

# Minimum face size in pixels (width & height) to consider valid
_MIN_FACE_PX: int = 60
# Size to normalize face crops for LBPH training
_FACE_SIZE: tuple[int, int] = (160, 160)


class FaceRecognizer:
    """Face detection + LBPH-based recognition with learning support."""

    def __init__(self) -> None:
        self.face_cascade: cv2.CascadeClassifier | None = None
        self._recognizer: Any = None
        self._label_to_name: dict[int, str] = {}   # LBPH label ID → name
        self._name_to_label: dict[str, int] = {}   # name → LBPH label ID
        self._next_label: int = 0
        self.face_rec_available: bool = False

        # Pending unknowns: face_id → {crop_gray, crop_b64, timestamp}
        self._pending_unknowns: dict[str, dict] = {}
        self._lock = threading.Lock()

        # Cooldown: don't flag the same region as unknown too frequently
        self._last_unknown_time: float = 0.0
        self._UNKNOWN_COOLDOWN: float = 5.0  # seconds between unknown flags

        self._load_cascade()
        self._init_recognizer()

    # ════════════════════════════════════════════
    # Initialisation
    # ════════════════════════════════════════════

    def _load_cascade(self) -> None:
        """Load the OpenCV Haar Cascade for frontal face detection."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise ValueError("Cascade classifier loaded but is empty")
            logger.info("✅ OpenCV face cascade loaded")
        except Exception as exc:
            logger.error("❌ Failed to load face cascade: %s", exc)
            self.face_cascade = None

    def _init_recognizer(self) -> None:
        """Initialize LBPH recognizer and load existing model if available."""
        try:
            self._recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1,
                neighbors=8,
                grid_x=8,
                grid_y=8,
                threshold=LBPH_CONFIDENCE_THRESHOLD,
            )
            # Load existing labels mapping
            self._load_labels()
            # Load existing trained model
            if LBPH_MODEL_FILE.exists() and len(self._label_to_name) > 0:
                self._recognizer.read(str(LBPH_MODEL_FILE))
                self.face_rec_available = True
                logger.info(
                    "✅ LBPH model loaded with %d known face(s): %s",
                    len(self._label_to_name),
                    list(self._label_to_name.values()),
                )
            else:
                # Try training from any stored images
                self._retrain_from_disk()
                logger.info("✅ LBPH recognizer initialized (no pre-trained model)")
        except Exception as exc:
            logger.error("❌ Failed to init LBPH recognizer: %s", exc)
            self._recognizer = None

    def _load_labels(self) -> None:
        """Load the label-to-name mapping from JSON."""
        if not FACE_LABELS_FILE.exists():
            return
        try:
            with FACE_LABELS_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            # JSON keys are strings, convert to int
            self._label_to_name = {int(k): v for k, v in data.items()}
            self._name_to_label = {v: int(k) for k, v in data.items()}
            if self._label_to_name:
                self._next_label = max(self._label_to_name.keys()) + 1
            logger.info("📋 Loaded %d face label(s)", len(self._label_to_name))
        except Exception as exc:
            logger.warning("Failed to load face labels: %s", exc)

    def _save_labels(self) -> None:
        """Persist label-to-name mapping to JSON."""
        try:
            FACE_LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with FACE_LABELS_FILE.open("w", encoding="utf-8") as fh:
                json.dump(
                    {str(k): v for k, v in self._label_to_name.items()},
                    fh, indent=2,
                )
        except Exception as exc:
            logger.error("Failed to save face labels: %s", exc)

    # ════════════════════════════════════════════
    # Public Detection API
    # ════════════════════════════════════════════

    def detect(self, frame: np.ndarray | None) -> dict[str, list]:
        """
        Detect and recognize faces in *frame*.

        Returns:
            {
              "faces": [{name, location, is_unknown, face_id}, ...],
              "unknown_faces": [{face_id, crop_base64}, ...]
            }
        """
        if frame is None or frame.size == 0:
            return {"faces": [], "unknown_faces": []}
        if self.face_cascade is None:
            return {"faces": [], "unknown_faces": []}

        # Clean up stale pending unknowns
        self._cleanup_stale_pending()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Histogram equalization for better detection in varying light
        gray = cv2.equalizeHist(gray)

        detected = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(_MIN_FACE_PX, _MIN_FACE_PX),
        )

        faces: list[dict] = []
        unknown_faces: list[dict] = []
        now = datetime.now().timestamp()

        for (x, y, w, h) in detected:
            location = [int(x), int(y), int(x + w), int(y + h)]

            # Extract and normalize face crop for recognition
            face_crop = gray[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
            face_resized = cv2.resize(face_crop, _FACE_SIZE)

            # Try to recognize
            name = None
            confidence = 0.0
            if self.face_rec_available and self._recognizer is not None:
                try:
                    label, conf = self._recognizer.predict(face_resized)
                    # LBPH confidence: lower = better match
                    # conf < threshold means recognized
                    if conf < LBPH_CONFIDENCE_THRESHOLD and label in self._label_to_name:
                        name = self._label_to_name[label]
                        confidence = max(0, 1.0 - conf / LBPH_CONFIDENCE_THRESHOLD)
                except Exception as exc:
                    logger.debug("LBPH predict failed: %s", exc)

            if name is not None:
                faces.append({
                    "name": name,
                    "location": location,
                    "is_unknown": False,
                    "face_id": None,
                    "confidence": round(confidence, 2),
                })
            else:
                # Unknown face — flag for learning (with cooldown)
                if (now - self._last_unknown_time) >= self._UNKNOWN_COOLDOWN:
                    face_id = f"face_{uuid.uuid4().hex[:8]}"
                    crop_b64 = self._crop_face_b64(frame, y, x + w, y + h, x)

                    with self._lock:
                        self._pending_unknowns[face_id] = {
                            "crop_gray": face_resized,
                            "crop_b64": crop_b64,
                            "timestamp": datetime.now(),
                        }

                    self._last_unknown_time = now

                    faces.append({
                        "name": "Unknown",
                        "location": location,
                        "is_unknown": True,
                        "face_id": face_id,
                    })
                    unknown_faces.append({
                        "face_id": face_id,
                        "crop_base64": crop_b64,
                    })
                else:
                    # Still report detection but don't re-flag
                    faces.append({
                        "name": "Unknown",
                        "location": location,
                        "is_unknown": False,
                        "face_id": None,
                    })

        return {"faces": faces, "unknown_faces": unknown_faces}

    # ════════════════════════════════════════════
    # Learning API
    # ════════════════════════════════════════════

    def add_known_face(self, name: str, encoding=None) -> bool:
        """
        Learn a face from the pending unknowns or from a provided encoding.
        Called by the learning flow when a user teaches a name.

        For LBPH, we need the actual face crop image, not an encoding.
        The crop is retrieved from pending_unknowns using the face_id
        passed through the unknown_handler flow.
        """
        # This is called from app.py after unknown_handler.learn_face()
        # The actual learning is done in learn_face_from_pending()
        # This method is kept for API compatibility
        return True

    def learn_face_from_pending(self, face_id: str, name: str) -> bool:
        """
        Learn a face using the stored pending crop.
        Saves the crop image and retrains the LBPH model.
        """
        name = name.strip()
        if not name:
            return False

        with self._lock:
            pending = self._pending_unknowns.pop(face_id, None)

        if pending is None:
            logger.warning("Face ID %s not found in pending", face_id)
            return False

        face_gray = pending.get("crop_gray")
        if face_gray is None:
            logger.warning("No face crop data for %s", face_id)
            return False

        # Save face image to disk
        person_dir = FACE_IMAGES_DIR / name.lower().replace(" ", "_")
        person_dir.mkdir(parents=True, exist_ok=True)
        img_count = len(list(person_dir.glob("*.jpg")))
        img_path = person_dir / f"{img_count + 1:03d}.jpg"
        cv2.imwrite(str(img_path), face_gray)
        logger.info("💾 Saved face image: %s", img_path)

        # Update label mapping
        if name not in self._name_to_label:
            label_id = self._next_label
            self._next_label += 1
            self._name_to_label[name] = label_id
            self._label_to_name[label_id] = name
            self._save_labels()

        # Retrain model with all stored images
        self._retrain_from_disk()

        return True

    def _retrain_from_disk(self) -> None:
        """Retrain LBPH model using all face images stored on disk."""
        if self._recognizer is None:
            return

        images: list[np.ndarray] = []
        labels: list[int] = []

        if not FACE_IMAGES_DIR.exists():
            return

        for person_dir in FACE_IMAGES_DIR.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            # Find or create label for this person
            if person_name not in self._name_to_label:
                label_id = self._next_label
                self._next_label += 1
                self._name_to_label[person_name] = label_id
                self._label_to_name[label_id] = person_name

            label_id = self._name_to_label[person_name]

            for img_file in person_dir.glob("*.jpg"):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, _FACE_SIZE)
                images.append(img)
                labels.append(label_id)

        if len(images) == 0:
            logger.info("No face images found for training")
            return

        logger.info(
            "🧠 Training LBPH on %d image(s) across %d person(s)...",
            len(images), len(set(labels)),
        )

        try:
            self._recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8,
                threshold=LBPH_CONFIDENCE_THRESHOLD,
            )
            self._recognizer.train(images, np.array(labels, dtype=np.int32))
            self._recognizer.write(str(LBPH_MODEL_FILE))
            self._save_labels()
            self.face_rec_available = True
            logger.info("✅ LBPH model trained and saved")
        except Exception as exc:
            logger.error("❌ LBPH training failed: %s", exc)

    # ════════════════════════════════════════════
    # Pending Management
    # ════════════════════════════════════════════

    def get_pending_encoding(self, face_id: str) -> list | None:
        """Return a dummy encoding for API compatibility."""
        with self._lock:
            if face_id in self._pending_unknowns:
                return [1.0]  # Dummy — LBPH doesn't use encodings
        return None

    def get_pending_count(self) -> int:
        """Number of pending unknown faces awaiting labels."""
        return len(self._pending_unknowns)

    def get_known_face_names(self) -> list[str]:
        """All learned face names."""
        return list(self._name_to_label.keys())

    def _cleanup_stale_pending(self) -> None:
        """Remove pending unknowns older than PENDING_TTL_SECONDS."""
        cutoff = datetime.now() - timedelta(seconds=PENDING_TTL_SECONDS)
        stale: list[str] = []

        with self._lock:
            for fid, data in self._pending_unknowns.items():
                ts = data.get("timestamp")
                if ts and ts < cutoff:
                    stale.append(fid)
            for fid in stale:
                self._pending_unknowns.pop(fid, None)

        if stale:
            logger.debug("🧹 Cleaned %d stale pending face(s)", len(stale))

    # ════════════════════════════════════════════
    # Utilities
    # ════════════════════════════════════════════

    @staticmethod
    def _crop_face_b64(
        frame: np.ndarray,
        top: int, right: int, bottom: int, left: int,
        padding: int = 10,
    ) -> str:
        """Crop a face region with optional padding and encode as base64 JPEG."""
        h, w = frame.shape[:2]
        t = max(0, top - padding)
        b = min(h, bottom + padding)
        l = max(0, left - padding)
        r = min(w, right + padding)

        crop = frame[t:b, l:r]
        if crop.size == 0:
            return ""

        _, buffer = cv2.imencode(".jpg", crop)
        return base64.b64encode(buffer).decode("utf-8")

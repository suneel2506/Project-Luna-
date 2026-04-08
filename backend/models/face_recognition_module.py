"""
Project LUNA — Face Recognition Module
═══════════════════════════════════════
Detects faces via OpenCV Haar Cascade and (optionally) recognizes them
using the `face_recognition` library's 128-D encoding model.

Features:
  • Encoding-based recognition when face_recognition is installed
  • Haar-cascade fallback for basic detection-only mode
  • Automatic stale-pending cleanup (configurable TTL)
  • Minimum face-size filtering to reduce false positives
  • Thread-safe known-faces access
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import (
    FACES_DIR,
    LEARNED_FACES_FILE,
    FACE_TOLERANCE,
    PENDING_TTL_SECONDS,
)

logger = logging.getLogger("luna.face_recognition")

# Minimum face size in pixels (width & height) to consider valid
_MIN_FACE_PX: int = 40


class FaceRecognizer:
    """Face detection + encoding-based recognition with learning support."""

    def __init__(self) -> None:
        self.known_faces: dict[str, list[np.ndarray]] = {}
        self.face_cascade: cv2.CascadeClassifier | None = None
        self.face_rec_available: bool = False
        self._fr: Any = None  # face_recognition module reference

        # Pending unknowns: face_id → encoding (list of floats)
        self._pending_unknowns: dict[str, list[float]] = {}
        self._pending_timestamps: dict[str, datetime] = {}
        self._lock = threading.Lock()

        self._load_cascade()
        self._try_load_face_recognition()
        self._load_known_faces()

    # ════════════════════════════════════════════
    # Initialisation Helpers
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

    def _try_load_face_recognition(self) -> None:
        """Attempt to import the face_recognition library (requires dlib)."""
        try:
            import face_recognition as fr
            self._fr = fr
            self.face_rec_available = True
            logger.info("✅ face_recognition library available")
        except ImportError:
            self._fr = None
            self.face_rec_available = False
            logger.warning("⚠️  face_recognition not installed — basic detection only")

    def _load_known_faces(self) -> None:
        """Load persisted face encodings from JSON storage."""
        path = Path(LEARNED_FACES_FILE)
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as fh:
                face_data: dict = json.load(fh)

            for name, encodings in face_data.items():
                self.known_faces[name] = [np.array(enc) for enc in encodings]

            if self.known_faces:
                logger.info("👤 Loaded %d known face(s)", len(self.known_faces))
        except Exception as exc:
            logger.warning("Failed to load known faces: %s", exc)
            self.known_faces = {}

    # ════════════════════════════════════════════
    # Public Detection API
    # ════════════════════════════════════════════

    def detect(self, frame: np.ndarray | None) -> dict[str, list]:
        """
        Detect (and optionally recognize) faces in *frame*.

        Returns:
            {
              "faces": [{name, location, is_unknown, face_id}, ...],
              "unknown_faces": [{face_id, crop_base64}, ...]
            }
        """
        if frame is None or frame.size == 0:
            return {"faces": [], "unknown_faces": []}

        # Clean up stale pending unknowns before processing
        self._cleanup_stale_pending()

        if self.face_rec_available:
            return self._detect_with_encoding(frame)
        return self._detect_basic(frame)

    # ════════════════════════════════════════════
    # Encoding-Based Detection (full recognition)
    # ════════════════════════════════════════════

    def _detect_with_encoding(self, frame: np.ndarray) -> dict[str, list]:
        """Use face_recognition for full detect + encode + match."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = self._fr.face_locations(rgb, model="hog")
        face_encodings = self._fr.face_encodings(rgb, face_locations)

        faces: list[dict] = []
        unknown_faces: list[dict] = []

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Skip tiny detections
            if (right - left) < _MIN_FACE_PX or (bottom - top) < _MIN_FACE_PX:
                continue

            location = [int(left), int(top), int(right), int(bottom)]
            name = self._match_face(encoding)

            if name is None:
                face_id = f"face_{uuid.uuid4().hex[:8]}"
                crop_b64 = self._crop_face_b64(frame, top, right, bottom, left)

                with self._lock:
                    self._pending_unknowns[face_id] = encoding.tolist()
                    self._pending_timestamps[face_id] = datetime.now()

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
                faces.append({
                    "name": name,
                    "location": location,
                    "is_unknown": False,
                    "face_id": None,
                })

        return {"faces": faces, "unknown_faces": unknown_faces}

    # ════════════════════════════════════════════
    # Basic Cascade Detection (no recognition)
    # ════════════════════════════════════════════

    def _detect_basic(self, frame: np.ndarray) -> dict[str, list]:
        """Haar-cascade fallback — detects faces but cannot truly recognize."""
        if self.face_cascade is None:
            return {"faces": [], "unknown_faces": []}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(_MIN_FACE_PX, _MIN_FACE_PX),
        )

        faces: list[dict] = []
        unknown_faces: list[dict] = []

        for (x, y, w, h) in detected:
            location = [int(x), int(y), int(x + w), int(y + h)]
            face_id = f"face_{uuid.uuid4().hex[:8]}"

            # In basic mode we can't match encodings — treat all as unknown
            # unless we already have a known face stored (demo shortcut)
            name = self._check_known_placeholder()

            if name is None:
                crop_b64 = self._crop_face_b64(frame, y, x + w, y + h, x)

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
                faces.append({
                    "name": name,
                    "location": location,
                    "is_unknown": False,
                    "face_id": None,
                })

        return {"faces": faces, "unknown_faces": unknown_faces}

    # ════════════════════════════════════════════
    # Face Matching
    # ════════════════════════════════════════════

    def _match_face(self, encoding: np.ndarray) -> str | None:
        """Compare *encoding* against all known faces. Returns name or None."""
        if not self.known_faces:
            return None

        with self._lock:
            for name, known_encs in self.known_faces.items():
                matches = self._fr.compare_faces(
                    known_encs, encoding, tolerance=FACE_TOLERANCE,
                )
                if any(matches):
                    return name

        return None

    def _check_known_placeholder(self) -> str | None:
        """Basic-mode placeholder: return first known face or None."""
        if self.known_faces:
            return next(iter(self.known_faces))
        return None

    # ════════════════════════════════════════════
    # Learning API
    # ════════════════════════════════════════════

    def add_known_face(self, name: str, encoding: list | np.ndarray) -> bool:
        """
        Directly add a face encoding to the known-faces store.
        Called by the learning flow when a user teaches a name.
        """
        try:
            enc_array = np.array(encoding) if isinstance(encoding, list) else encoding

            with self._lock:
                self.known_faces.setdefault(name, []).append(enc_array)

            self._save_known_faces()
            logger.info("✅ Added known face: %s", name)
            return True
        except Exception as exc:
            logger.error("Failed to add known face: %s", exc)
            return False

    # ════════════════════════════════════════════
    # Pending-Unknown Management
    # ════════════════════════════════════════════

    def get_pending_encoding(self, face_id: str) -> list | None:
        """Return the encoding for a pending unknown face (without removing it)."""
        with self._lock:
            return self._pending_unknowns.get(face_id)

    def get_pending_count(self) -> int:
        """Number of pending unknown faces awaiting labels."""
        return len(self._pending_unknowns)

    def get_known_face_names(self) -> list[str]:
        """All learned face names."""
        return list(self.known_faces.keys())

    def _cleanup_stale_pending(self) -> None:
        """Remove pending unknowns older than PENDING_TTL_SECONDS."""
        cutoff = datetime.now() - timedelta(seconds=PENDING_TTL_SECONDS)
        stale: list[str] = []

        with self._lock:
            for fid, ts in self._pending_timestamps.items():
                if ts < cutoff:
                    stale.append(fid)
            for fid in stale:
                self._pending_unknowns.pop(fid, None)
                self._pending_timestamps.pop(fid, None)

        if stale:
            logger.debug("🧹 Cleaned %d stale pending face(s)", len(stale))

    # ════════════════════════════════════════════
    # Persistence
    # ════════════════════════════════════════════

    def _save_known_faces(self) -> None:
        """Persist known face encodings to JSON."""
        try:
            save_data: dict[str, list] = {}
            with self._lock:
                for name, encodings in self.known_faces.items():
                    save_data[name] = [
                        enc.tolist() if isinstance(enc, np.ndarray) else enc
                        for enc in encodings
                    ]

            path = Path(LEARNED_FACES_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(save_data, fh, indent=2)

            logger.info("💾 Saved %d known face(s) to storage", len(save_data))
        except Exception as exc:
            logger.error("Failed to save known faces: %s", exc)

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

"""
Project LUNA — Emotion Detection Module
════════════════════════════════════════
Detects facial emotions using DeepFace when available, otherwise
returns an empty result (no fake placeholder data).

Supported emotions: happy, sad, angry, neutral, surprise, fear, disgust

Features:
  • DeepFace integration with enforce_detection=False for robustness
  • Per-region and full-frame analysis modes
  • Emotion result caching to avoid redundant analysis on similar crops
  • Clean fallback when DeepFace is not installed
"""

from __future__ import annotations

import logging
import hashlib
from typing import Any

import cv2
import numpy as np

from config import EMOTION_CONFIDENCE

logger = logging.getLogger("luna.emotion_detection")

# Simple LRU-ish cache: (frame_hash, region) → result
_CACHE_MAX: int = 32


class EmotionDetector:
    """Facial emotion detection backed by DeepFace."""

    EMOTIONS: list[str] = [
        "happy", "sad", "angry", "neutral", "surprise", "fear", "disgust",
    ]

    def __init__(self) -> None:
        self.deepface_available: bool = False
        self._deepface: Any = None
        self._cache: dict[str, list[dict]] = {}
        self._try_load_deepface()

    # ── Initialisation ──────────────────────────

    def _try_load_deepface(self) -> None:
        """Attempt to import DeepFace for emotion analysis."""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self.deepface_available = True
            logger.info("✅ DeepFace loaded for emotion detection")
        except ImportError:
            logger.warning("⚠️  deepface not installed — emotion detection disabled")

    # ── Public API ──────────────────────────────

    def detect(
        self,
        frame: np.ndarray | None,
        face_locations: list[list[int]] | None = None,
    ) -> list[dict]:
        """
        Detect emotions in faces within *frame*.

        Args:
            frame:          BGR numpy array from OpenCV
            face_locations: Optional list of [x1, y1, x2, y2] bounding boxes.
                            When provided, only those regions are analysed.

        Returns:
            List of dicts with keys:
                emotion       (str)   – dominant emotion name
                confidence    (float) – 0.0 – 1.0
                location      (list)  – [x1, y1, x2, y2]
                all_emotions  (dict)  – {emotion_name: confidence, ...}
        """
        if frame is None or frame.size == 0:
            return []
        if not self.deepface_available:
            return []

        return self._analyse(frame, face_locations)

    # ── DeepFace Analysis ───────────────────────

    def _analyse(
        self,
        frame: np.ndarray,
        face_locations: list[list[int]] | None,
    ) -> list[dict]:
        """Run emotion analysis with DeepFace."""
        results: list[dict] = []

        try:
            if face_locations:
                # Analyse specific face crops
                for loc in face_locations:
                    x1, y1, x2, y2 = loc
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    parsed = self._analyse_crop(crop, loc)
                    if parsed:
                        results.append(parsed)
            else:
                # Full-frame analysis — DeepFace finds faces internally
                results = self._analyse_full_frame(frame)

        except Exception as exc:
            logger.error("Emotion detection error: %s", exc)

        return results

    def _analyse_crop(self, crop: np.ndarray, location: list[int]) -> dict | None:
        """Analyse a single face crop and return structured result."""
        # Check cache
        cache_key = self._frame_hash(crop)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached:
                result = cached[0].copy()
                result["location"] = location
                return result

        try:
            analysis = self._deepface.analyze(
                crop,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            if isinstance(analysis, list):
                analysis = analysis[0]

            dominant: str = analysis.get("dominant_emotion", "neutral")
            emotions: dict = analysis.get("emotion", {})
            confidence = emotions.get(dominant, 0) / 100.0

            if confidence < EMOTION_CONFIDENCE:
                return None

            result = {
                "emotion": dominant,
                "confidence": round(confidence, 2),
                "location": location,
                "all_emotions": {
                    k: round(v / 100.0, 2) for k, v in emotions.items()
                },
            }

            # Update cache
            self._cache_put(cache_key, [result])
            return result

        except Exception as exc:
            logger.debug("Emotion analysis failed for crop: %s", exc)
            return None

    def _analyse_full_frame(self, frame: np.ndarray) -> list[dict]:
        """Analyse the entire frame — DeepFace handles face finding."""
        try:
            analysis = self._deepface.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            if not isinstance(analysis, list):
                analysis = [analysis]

            results: list[dict] = []
            for face in analysis:
                dominant: str = face.get("dominant_emotion", "neutral")
                emotions: dict = face.get("emotion", {})
                region: dict = face.get("region", {})

                confidence = emotions.get(dominant, 0) / 100.0
                if confidence < EMOTION_CONFIDENCE:
                    continue

                results.append({
                    "emotion": dominant,
                    "confidence": round(confidence, 2),
                    "location": [
                        region.get("x", 0),
                        region.get("y", 0),
                        region.get("x", 0) + region.get("w", 0),
                        region.get("y", 0) + region.get("h", 0),
                    ],
                    "all_emotions": {
                        k: round(v / 100.0, 2) for k, v in emotions.items()
                    },
                })

            return results

        except Exception as exc:
            logger.debug("Full-frame emotion analysis failed: %s", exc)
            return []

    # ── Caching Helpers ─────────────────────────

    @staticmethod
    def _frame_hash(img: np.ndarray) -> str:
        """Fast perceptual hash: downsample + md5."""
        small = cv2.resize(img, (16, 16))
        return hashlib.md5(small.tobytes()).hexdigest()

    def _cache_put(self, key: str, value: list[dict]) -> None:
        """Insert into cache, evicting oldest when full."""
        if len(self._cache) >= _CACHE_MAX:
            # Pop the first key (oldest insertion)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value

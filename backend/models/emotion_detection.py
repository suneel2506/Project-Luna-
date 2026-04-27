"""
Project LUNA — Emotion Detection Module
════════════════════════════════════════
Detects facial emotions using DeepFace when available, or falls back
to a lightweight rule-based approach using face geometry analysis.

Supported emotions: happy, sad, angry, neutral, surprise, fear, disgust

Features:
  • DeepFace integration (when available)
  • Lightweight OpenCV-based fallback using face landmark geometry
  • Per-region and full-frame analysis modes
  • Clean fallback chain: DeepFace → Geometry-based → empty
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from config import EMOTION_CONFIDENCE

logger = logging.getLogger("luna.emotion_detection")


class EmotionDetector:
    """Facial emotion detection with multiple backend support."""

    EMOTIONS: list[str] = [
        "happy", "sad", "angry", "neutral", "surprise", "fear", "disgust",
    ]

    def __init__(self) -> None:
        self.deepface_available: bool = False
        self._deepface: Any = None
        self._face_cascade: cv2.CascadeClassifier | None = None
        self._try_load_deepface()
        self._load_face_cascade()
        logger.info(
            "✅ Emotion detector initialized (mode: %s)",
            "deepface" if self.deepface_available else "geometry"
        )

    # ── Initialisation ──────────────────────────

    def _try_load_deepface(self) -> None:
        """Attempt to import DeepFace for emotion analysis."""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self.deepface_available = True
            logger.info("✅ DeepFace loaded for emotion detection")
        except ImportError:
            logger.info("ℹ️  DeepFace not available — using geometry-based emotion detection")

    def _load_face_cascade(self) -> None:
        """Load cascade for geometry-based fallback."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self._face_cascade = None

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

        Returns:
            List of dicts with keys:
                emotion       (str)   – dominant emotion name
                confidence    (float) – 0.0 – 1.0
                location      (list)  – [x1, y1, x2, y2]
                all_emotions  (dict)  – {emotion_name: confidence, ...}
        """
        if frame is None or frame.size == 0:
            return []

        if self.deepface_available:
            return self._analyse_deepface(frame, face_locations)

        # Fallback: geometry-based emotion detection
        return self._analyse_geometry(frame, face_locations)

    # ── DeepFace Analysis ───────────────────────

    def _analyse_deepface(
        self,
        frame: np.ndarray,
        face_locations: list[list[int]] | None,
    ) -> list[dict]:
        """Run emotion analysis with DeepFace."""
        results: list[dict] = []

        try:
            if face_locations:
                for loc in face_locations:
                    x1, y1, x2, y2 = loc
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    parsed = self._deepface_crop(crop, loc)
                    if parsed:
                        results.append(parsed)
            else:
                results = self._deepface_full_frame(frame)

        except Exception as exc:
            logger.error("DeepFace emotion error: %s", exc)

        return results

    def _deepface_crop(self, crop: np.ndarray, location: list[int]) -> dict | None:
        """Analyse a single face crop with DeepFace."""
        try:
            analysis = self._deepface.analyze(
                crop, actions=["emotion"],
                enforce_detection=False, silent=True,
            )
            if isinstance(analysis, list):
                analysis = analysis[0]

            dominant: str = analysis.get("dominant_emotion", "neutral")
            emotions: dict = analysis.get("emotion", {})
            confidence = emotions.get(dominant, 0) / 100.0

            if confidence < EMOTION_CONFIDENCE:
                return None

            return {
                "emotion": dominant,
                "confidence": round(confidence, 2),
                "location": location,
                "all_emotions": {k: round(v / 100.0, 2) for k, v in emotions.items()},
            }
        except Exception as exc:
            logger.debug("DeepFace crop analysis failed: %s", exc)
            return None

    def _deepface_full_frame(self, frame: np.ndarray) -> list[dict]:
        """Analyse entire frame with DeepFace."""
        try:
            analysis = self._deepface.analyze(
                frame, actions=["emotion"],
                enforce_detection=False, silent=True,
            )
            if not isinstance(analysis, list):
                analysis = [analysis]

            results: list[dict] = []
            for face in analysis:
                dominant = face.get("dominant_emotion", "neutral")
                emotions = face.get("emotion", {})
                region = face.get("region", {})
                confidence = emotions.get(dominant, 0) / 100.0

                if confidence < EMOTION_CONFIDENCE:
                    continue

                results.append({
                    "emotion": dominant,
                    "confidence": round(confidence, 2),
                    "location": [
                        region.get("x", 0), region.get("y", 0),
                        region.get("x", 0) + region.get("w", 0),
                        region.get("y", 0) + region.get("h", 0),
                    ],
                    "all_emotions": {k: round(v / 100.0, 2) for k, v in emotions.items()},
                })
            return results
        except Exception:
            return []

    # ── Geometry-Based Emotion Detection ────────

    def _analyse_geometry(
        self,
        frame: np.ndarray,
        face_locations: list[list[int]] | None,
    ) -> list[dict]:
        """
        Lightweight emotion detection using face geometry analysis.

        Analyses the face region for:
        - Brightness distribution (forehead vs lower face)
        - Edge density (wrinkles, expression lines)
        - Symmetry
        - Mouth region darkness (open mouth detection)
        - Eye region analysis
        """
        results: list[dict] = []

        if face_locations:
            for loc in face_locations:
                x1, y1, x2, y2 = loc
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                emotion_result = self._geometry_analyse_crop(crop, loc)
                if emotion_result:
                    results.append(emotion_result)
        else:
            # Detect faces ourselves
            if self._face_cascade is None:
                return []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
            )
            for (x, y, w, h) in faces:
                crop = frame[y:y+h, x:x+w]
                loc = [int(x), int(y), int(x+w), int(y+h)]
                emotion_result = self._geometry_analyse_crop(crop, loc)
                if emotion_result:
                    results.append(emotion_result)

        return results

    def _geometry_analyse_crop(
        self, crop: np.ndarray, location: list[int],
    ) -> dict | None:
        """
        Analyse a face crop using geometric features to estimate emotion.

        Features extracted:
        1. Mouth openness (dark region ratio in lower face)
        2. Edge density (expression lines)
        3. Brightness variance (muscle tension changes)
        4. Eye region brightness
        """
        try:
            h, w = crop.shape[:2]
            if h < 30 or w < 30:
                return None

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            gray = cv2.equalizeHist(gray)

            # Split face into regions
            upper_face = gray[0:h//3, :]              # Forehead + eyes
            mid_face = gray[h//3:2*h//3, :]           # Nose + cheeks
            lower_face = gray[2*h//3:, :]              # Mouth + chin

            # Eye region (upper third, middle horizontal portion)
            eye_region = gray[h//6:h//3, w//6:5*w//6]

            # Mouth region (lower third, center)
            mouth_region = gray[2*h//3:, w//4:3*w//4]

            # Feature 1: Mouth darkness ratio (open mouth = darker)
            mouth_mean = np.mean(mouth_region)
            face_mean = np.mean(gray)
            mouth_dark_ratio = 1.0 - (mouth_mean / max(face_mean, 1))

            # Feature 2: Edge density in lower face (smile lines)
            edges_lower = cv2.Canny(lower_face, 50, 150)
            edge_density_lower = np.sum(edges_lower > 0) / max(edges_lower.size, 1)

            # Feature 3: Edge density in upper face (frown lines)
            edges_upper = cv2.Canny(upper_face, 50, 150)
            edge_density_upper = np.sum(edges_upper > 0) / max(edges_upper.size, 1)

            # Feature 4: Brightness variance (expression = more variance)
            brightness_var = np.var(gray) / 1000.0

            # Feature 5: Eye brightness (wide eyes = more white/bright)
            eye_brightness = np.mean(eye_region) if eye_region.size > 0 else 128

            # ── Rule-based classification ──
            scores: dict[str, float] = {
                "neutral": 0.40,
                "happy": 0.0,
                "surprise": 0.0,
                "sad": 0.0,
                "angry": 0.0,
            }

            # Happy: high edge density in lower face (smile lines) + brighter lower face
            if edge_density_lower > 0.08:
                scores["happy"] += 0.3
            if mouth_dark_ratio > 0.05:
                scores["happy"] += 0.2  # Mouth movement

            # Surprise: mouth open (dark) + eyes wide (bright)
            if mouth_dark_ratio > 0.15:
                scores["surprise"] += 0.4
            if eye_brightness > 140:
                scores["surprise"] += 0.2

            # Angry: high edge density upper (frown) + low brightness variance
            if edge_density_upper > 0.1:
                scores["angry"] += 0.3
            if brightness_var < 2.0:
                scores["angry"] += 0.1

            # Sad: low overall edge density + darker overall
            if edge_density_lower < 0.05 and edge_density_upper < 0.06:
                scores["sad"] += 0.2
            if face_mean < 100:
                scores["sad"] += 0.1

            # Find dominant emotion
            dominant = max(scores, key=scores.get)
            confidence = min(scores[dominant], 0.85)

            if confidence < EMOTION_CONFIDENCE:
                return None

            all_emotions = {
                k: round(v, 2)
                for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            }

            return {
                "emotion": dominant,
                "confidence": round(confidence, 2),
                "location": location,
                "all_emotions": all_emotions,
            }

        except Exception as exc:
            logger.debug("Geometry emotion analysis failed: %s", exc)
            return None

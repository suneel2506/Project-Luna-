"""
Project LUNA — Sign Language Detection Module
══════════════════════════════════════════════
Uses MediaPipe Tasks API for hand landmark detection and rule-based
gesture classification for a limited sign set.

Supports both:
  • MediaPipe Tasks API (mp.tasks) — Python 3.14+ / mediapipe >= 0.10.30
  • Legacy MediaPipe Solutions (mp.solutions) — older Python versions

Supported signs: Hello, Yes, No, Thank You, Help, I Love You

Features:
  • Hand landmark extraction via MediaPipe
  • Improved finger-state detection (left/right aware thumb)
  • Reduced debounce for responsive detection
  • Better gesture classification with margin thresholds
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from config import SUPPORTED_SIGNS, BASE_DIR

logger = logging.getLogger("luna.sign_detection")

# Debounce: minimum seconds between reporting the same sign
_DEBOUNCE_SECONDS: float = 1.5

# Margin for finger extension detection
_EXTENSION_MARGIN: float = 0.02

# Path to the hand landmarker model file
_HAND_MODEL_PATH: Path = BASE_DIR / "models" / "hand_landmarker.task"


class SignDetector:
    """Hand sign / gesture detection using MediaPipe landmarks."""

    def __init__(self) -> None:
        self._detector: Any = None
        self._legacy_hands: Any = None
        self._mp_hands: Any = None
        self.mediapipe_available: bool = False
        self._use_tasks_api: bool = False
        self._last_signs: dict[str, float] = {}
        self._try_load_mediapipe()

    # ── Initialisation ──────────────────────────

    def _try_load_mediapipe(self) -> None:
        """Attempt to load MediaPipe — try Tasks API first, then legacy."""
        # Try new Tasks API (mediapipe >= 0.10.30, required for Python 3.14)
        if self._try_load_tasks_api():
            return
        # Fallback to legacy Solutions API
        self._try_load_legacy()

    def _try_load_tasks_api(self) -> bool:
        """Load MediaPipe using the new Tasks API."""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )

            if not _HAND_MODEL_PATH.exists():
                logger.warning(
                    "Hand landmarker model not found at %s — "
                    "download from https://storage.googleapis.com/mediapipe-models/"
                    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                    _HAND_MODEL_PATH,
                )
                return False

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_HAND_MODEL_PATH)),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.4,
            )
            self._detector = HandLandmarker.create_from_options(options)
            self._use_tasks_api = True
            self.mediapipe_available = True
            logger.info("✅ MediaPipe Tasks API loaded for sign detection")
            return True

        except Exception as exc:
            logger.debug("Tasks API not available: %s", exc)
            return False

    def _try_load_legacy(self) -> bool:
        """Load MediaPipe using the legacy Solutions API."""
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._legacy_hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.4,
            )
            self._use_tasks_api = False
            self.mediapipe_available = True
            logger.info("✅ MediaPipe legacy API loaded for sign detection")
            return True
        except Exception as exc:
            logger.warning("⚠️  MediaPipe not available (%s) — sign detection disabled", exc)
            return False

    # ── Public API ──────────────────────────────

    def detect(self, frame) -> list[dict]:
        """
        Detect hand signs / gestures in *frame*.

        Returns list of dicts:
            sign        (str)   – gesture name
            confidence  (float) – 0.0 – 1.0
            hand        (str)   – 'left' or 'right'
            landmarks   (list)  – [{x, y, z}, ...]
        """
        if frame is None or not self.mediapipe_available:
            return []

        if self._use_tasks_api:
            return self._detect_tasks_api(frame)
        return self._detect_legacy(frame)

    def get_supported_signs(self) -> list[str]:
        """Return the list of recognizable signs."""
        return SUPPORTED_SIGNS

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._detector is not None:
            self._detector.close()
        if self._legacy_hands is not None:
            self._legacy_hands.close()
        logger.debug("MediaPipe resources released")

    # ── Tasks API Detection ─────────────────────

    def _detect_tasks_api(self, frame) -> list[dict]:
        """Run detection using the new MediaPipe Tasks API."""
        import mediapipe as mp
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._detector.detect(mp_image)
        except Exception as exc:
            logger.debug("Tasks API detection failed: %s", exc)
            return []

        detections: list[dict] = []

        if not result.hand_landmarks or not result.handedness:
            return detections

        now = time.time()

        for hand_lms, handedness in zip(result.hand_landmarks, result.handedness):
            hand_label = handedness[0].category_name.lower()

            landmarks: list[dict[str, float]] = [
                {
                    "x": round(lm.x, 4),
                    "y": round(lm.y, 4),
                    "z": round(lm.z, 4),
                }
                for lm in hand_lms
            ]

            sign, confidence = self._classify_gesture(landmarks, hand_label)
            if sign is None:
                continue

            # Debounce
            last_time = self._last_signs.get(sign, 0.0)
            if (now - last_time) < _DEBOUNCE_SECONDS:
                continue

            self._last_signs[sign] = now
            detections.append({
                "sign": sign,
                "confidence": round(confidence, 2),
                "hand": hand_label,
                "landmarks": landmarks,
            })

        return detections

    # ── Legacy API Detection ────────────────────

    def _detect_legacy(self, frame) -> list[dict]:
        """Run detection using the legacy MediaPipe Solutions API."""
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._legacy_hands.process(rgb)

        detections: list[dict] = []

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return detections

        now = time.time()

        for hand_lm, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            hand_label: str = handedness.classification[0].label.lower()

            landmarks: list[dict[str, float]] = [
                {
                    "x": round(lm.x, 4),
                    "y": round(lm.y, 4),
                    "z": round(lm.z, 4),
                }
                for lm in hand_lm.landmark
            ]

            sign, confidence = self._classify_gesture(landmarks, hand_label)
            if sign is None:
                continue

            last_time = self._last_signs.get(sign, 0.0)
            if (now - last_time) < _DEBOUNCE_SECONDS:
                continue

            self._last_signs[sign] = now
            detections.append({
                "sign": sign,
                "confidence": round(confidence, 2),
                "hand": hand_label,
                "landmarks": landmarks,
            })

        return detections

    # ── Gesture Classification ──────────────────

    def _classify_gesture(
        self,
        landmarks: list[dict[str, float]],
        hand: str,
    ) -> tuple[str | None, float]:
        """
        Rule-based gesture classification using finger positions.

        MediaPipe landmark indices:
            0  WRIST
            4  THUMB_TIP    3 THUMB_IP    2 THUMB_MCP
            8  INDEX_TIP    6 INDEX_PIP   5 INDEX_MCP
           12  MIDDLE_TIP  10 MIDDLE_PIP  9 MIDDLE_MCP
           16  RING_TIP    14 RING_PIP   13 RING_MCP
           20  PINKY_TIP   18 PINKY_PIP  17 PINKY_MCP
        """
        fingers = self._get_finger_states(landmarks, hand)
        thumb, index, middle, ring, pinky = fingers
        extended_count = sum(fingers)

        # I LOVE YOU — thumb + index + pinky extended, middle + ring closed
        if thumb and index and not middle and not ring and pinky:
            return "i_love_you", 0.90

        # HELLO — open palm, all five fingers extended
        if extended_count == 5:
            wrist_z = landmarks[0]["z"]
            middle_z = landmarks[12]["z"]
            if middle_z < wrist_z:
                return "hello", 0.85
            else:
                return "thank_you", 0.75

        # YES — fist with thumb up only
        if thumb and not index and not middle and not ring and not pinky:
            if landmarks[4]["y"] < landmarks[3]["y"]:
                return "yes", 0.85

        # NO — index + middle extended (V-sign), others closed
        if not thumb and index and middle and not ring and not pinky:
            return "no", 0.80

        if index and middle and not ring and not pinky:
            return "no", 0.75

        # HELP — closed fist (all fingers down)
        if extended_count == 0:
            return "help", 0.70

        return None, 0.0

    def _get_finger_states(
        self,
        landmarks: list[dict[str, float]],
        hand: str,
    ) -> tuple[bool, bool, bool, bool, bool]:
        """
        Determine which fingers are extended.
        Thumb is orientation-aware. Margin prevents borderline detections.
        """
        # Thumb — orientation-aware
        if hand == "right":
            thumb = landmarks[4]["x"] < (landmarks[3]["x"] - _EXTENSION_MARGIN)
        else:
            thumb = landmarks[4]["x"] > (landmarks[3]["x"] + _EXTENSION_MARGIN)

        # Also check thumb tip above IP joint
        thumb_up = landmarks[4]["y"] < landmarks[3]["y"]
        thumb = thumb or thumb_up

        # Other fingers: tip y < pip y ⟹ extended
        index  = landmarks[8]["y"]  < (landmarks[6]["y"] - _EXTENSION_MARGIN)
        middle = landmarks[12]["y"] < (landmarks[10]["y"] - _EXTENSION_MARGIN)
        ring   = landmarks[16]["y"] < (landmarks[14]["y"] - _EXTENSION_MARGIN)
        pinky  = landmarks[20]["y"] < (landmarks[18]["y"] - _EXTENSION_MARGIN)

        return (thumb, index, middle, ring, pinky)

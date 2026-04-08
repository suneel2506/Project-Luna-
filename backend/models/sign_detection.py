"""
Project LUNA — Sign Language Detection Module
══════════════════════════════════════════════
Uses MediaPipe for hand landmark detection and rule-based
gesture classification for a limited sign set.

Supported signs: Hello, Yes, No, Thank You, Help, I Love You

Features:
  • MediaPipe-backed hand landmark extraction
  • Improved finger-state detection (left/right aware thumb)
  • Gesture debouncing — suppress repeated same-sign detections
  • Proper MediaPipe resource cleanup
"""

from __future__ import annotations

import logging
import time
from typing import Any

from config import SUPPORTED_SIGNS

logger = logging.getLogger("luna.sign_detection")

# Debounce: minimum seconds between reporting the same sign
_DEBOUNCE_SECONDS: float = 3.0


class SignDetector:
    """Hand sign / gesture detection using MediaPipe landmarks."""

    def __init__(self) -> None:
        self._mp_hands: Any = None
        self._hands: Any = None
        self.mediapipe_available: bool = False
        self._last_signs: dict[str, float] = {}   # sign → last_reported_time
        self._try_load_mediapipe()

    # ── Initialisation ──────────────────────────

    def _try_load_mediapipe(self) -> None:
        """Attempt to import and configure MediaPipe hands."""
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mediapipe_available = True
            logger.info("✅ MediaPipe loaded for sign detection")
        except (ImportError, AttributeError, Exception) as exc:
            self.mediapipe_available = False
            logger.warning("⚠️  MediaPipe not available (%s) — sign detection disabled", exc)

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
        if frame is None:
            return []
        if not self.mediapipe_available:
            return []

        return self._detect_with_mediapipe(frame)

    def get_supported_signs(self) -> list[str]:
        """Return the list of recognizable signs."""
        return SUPPORTED_SIGNS

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()
            logger.debug("MediaPipe hands resources released")

    # ── MediaPipe Detection ─────────────────────

    def _detect_with_mediapipe(self, frame) -> list[dict]:
        """Run MediaPipe hand detection + gesture classification."""
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

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

            # Debounce: skip if same sign was reported recently
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

        # I LOVE YOU — thumb + index + pinky extended, middle + ring closed
        if thumb and index and not middle and not ring and pinky:
            return "i_love_you", 0.85

        # HELLO — open palm, all five fingers extended
        if all(fingers):
            # Distinguish from "thank_you" by z-depth (palm orientation)
            if landmarks[12]["z"] < landmarks[0]["z"]:
                return "thank_you", 0.70
            return "hello", 0.85

        # YES — fist with thumb up only
        if thumb and not index and not middle and not ring and not pinky:
            return "yes", 0.80

        # NO — index + middle extended (peace / no sign), others closed
        if index and middle and not ring and not pinky:
            return "no", 0.75

        # HELP — closed fist (all fingers down including thumb)
        if not thumb and not index and not middle and not ring and not pinky:
            return "help", 0.65

        return None, 0.0

    def _get_finger_states(
        self,
        landmarks: list[dict[str, float]],
        hand: str,
    ) -> tuple[bool, bool, bool, bool, bool]:
        """
        Determine which fingers are extended.

        The thumb check is orientation-aware: on the right hand the
        extended thumb tip moves left (smaller x), on the left hand
        it moves right (larger x).
        """
        # Thumb — orientation-aware comparison
        if hand == "right":
            thumb = landmarks[4]["x"] < landmarks[3]["x"]
        else:
            thumb = landmarks[4]["x"] > landmarks[3]["x"]

        # Other fingers: tip y < pip y ⟹ extended (image y increases downward)
        index  = landmarks[8]["y"]  < landmarks[6]["y"]
        middle = landmarks[12]["y"] < landmarks[10]["y"]
        ring   = landmarks[16]["y"] < landmarks[14]["y"]
        pinky  = landmarks[20]["y"] < landmarks[18]["y"]

        return (thumb, index, middle, ring, pinky)

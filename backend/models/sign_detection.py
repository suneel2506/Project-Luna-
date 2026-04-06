"""
Project LUNA — Sign Language Detection Module
Uses MediaPipe for hand landmark detection and rule-based
gesture classification for a limited sign set.

Supported signs: Hello, Yes, No, Thank You, Help, I Love You
"""

import logging
import math
import random
import sys
import os

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUPPORTED_SIGNS


class SignDetector:
    """Hand sign/gesture detection using MediaPipe landmarks."""

    def __init__(self):
        self.mp_hands = None
        self.hands = None
        self.mp_draw = None
        self.mediapipe_available = False
        self._try_load_mediapipe()

    def _try_load_mediapipe(self):
        """Try to load MediaPipe for hand tracking."""
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mediapipe_available = True
            logger.info("✅ MediaPipe loaded for sign detection")
        except ImportError:
            self.mediapipe_available = False
            logger.warning("⚠️ MediaPipe not installed. Using placeholder sign detection.")

    def detect(self, frame):
        """
        Detect hand signs/gestures in a frame.

        Args:
            frame: numpy array (BGR image)

        Returns:
            list of dicts:
                - sign (str): Detected sign name
                - confidence (float): Detection confidence
                - hand (str): 'left' or 'right'
                - landmarks (list): Key landmark positions
        """
        if frame is None:
            return []

        if self.mediapipe_available:
            return self._detect_with_mediapipe(frame)
        else:
            return self._placeholder_detect()

    def _detect_with_mediapipe(self, frame):
        """Use MediaPipe for real hand gesture detection."""
        import cv2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        detections = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Get hand label (left/right)
                hand_label = handedness.classification[0].label.lower()

                # Extract landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append({
                        "x": round(lm.x, 4),
                        "y": round(lm.y, 4),
                        "z": round(lm.z, 4)
                    })

                # Classify the gesture
                sign, confidence = self._classify_gesture(landmarks)

                if sign:
                    detections.append({
                        "sign": sign,
                        "confidence": round(confidence, 2),
                        "hand": hand_label,
                        "landmarks": landmarks
                    })

        return detections

    def _classify_gesture(self, landmarks):
        """
        Rule-based gesture classification using finger positions.

        Landmark indices (MediaPipe):
        0: WRIST
        4: THUMB_TIP, 3: THUMB_IP, 2: THUMB_MCP
        8: INDEX_TIP, 7: INDEX_DIP, 6: INDEX_PIP, 5: INDEX_MCP
        12: MIDDLE_TIP, 11: MIDDLE_DIP, 10: MIDDLE_PIP, 9: MIDDLE_MCP
        16: RING_TIP, 15: RING_DIP, 14: RING_PIP, 13: RING_MCP
        20: PINKY_TIP, 19: PINKY_DIP, 18: PINKY_PIP, 17: PINKY_MCP
        """
        # Get finger states (extended or not)
        fingers = self._get_finger_states(landmarks)
        thumb, index, middle, ring, pinky = fingers

        # ---- HELLO (Open palm, all fingers extended) ----
        if all(fingers):
            return "hello", 0.85

        # ---- YES (Fist with thumb up) ----
        if thumb and not index and not middle and not ring and not pinky:
            return "yes", 0.80

        # ---- NO (Index finger wagging / index + middle extended, others closed) ----
        if index and middle and not ring and not pinky:
            # Peace/No sign
            return "no", 0.75

        # ---- THANK YOU (Flat hand moving from chin — simplified: open palm facing up) ----
        # Approximation: all fingers extended, palm facing camera (z values)
        if all(fingers) and landmarks[12]["z"] < landmarks[0]["z"]:
            return "thank_you", 0.70

        # ---- HELP (Fist on open palm — simplified: one hand fist-like) ----
        if not index and not middle and not ring and not pinky and not thumb:
            return "help", 0.65

        # ---- I LOVE YOU (thumb + index + pinky extended) ----
        if thumb and index and not middle and not ring and pinky:
            return "i_love_you", 0.85

        return None, 0.0

    def _get_finger_states(self, landmarks):
        """
        Determine which fingers are extended.

        Returns:
            tuple of 5 bools: (thumb, index, middle, ring, pinky)
        """
        # Thumb: compare tip x with IP x (depends on hand orientation)
        thumb = landmarks[4]["x"] < landmarks[3]["x"]

        # Other fingers: tip y < pip y means extended (image coords, y increases downward)
        index = landmarks[8]["y"] < landmarks[6]["y"]
        middle = landmarks[12]["y"] < landmarks[10]["y"]
        ring = landmarks[16]["y"] < landmarks[14]["y"]
        pinky = landmarks[20]["y"] < landmarks[18]["y"]

        return (thumb, index, middle, ring, pinky)

    def _placeholder_detect(self):
        """
        Placeholder sign detection for development/testing.
        Randomly returns a sign detection with low probability.
        """
        if random.random() > 0.7:  # 30% chance of detection
            sign = random.choice(SUPPORTED_SIGNS)
            return [{
                "sign": sign,
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "hand": random.choice(["left", "right"]),
                "landmarks": []
            }]
        return []

    def get_supported_signs(self):
        """Return list of supported sign gestures."""
        return SUPPORTED_SIGNS

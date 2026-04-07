"""
Project LUNA — Emotion Detection Module
Detects facial emotions using deepface or a lightweight fallback.
Supported emotions: happy, sad, angry, neutral, surprise, fear, disgust.
"""

import logging
import random
import cv2
import sys
import os

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMOTION_CONFIDENCE


class EmotionDetector:
    """Facial emotion detection with deepface or placeholder fallback."""

    EMOTIONS = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]

    def __init__(self):
        self.deepface_available = False
        self._try_load_deepface()

    def _try_load_deepface(self):
        """Try to load deepface for emotion analysis."""
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.deepface_available = True
            logger.info("✅ DeepFace loaded for emotion detection")
        except ImportError:
            self.DeepFace = None
            self.deepface_available = False
            logger.warning("⚠️ deepface not installed. Using placeholder emotion detection.")

    def detect(self, frame, face_locations=None):
        """
        Detect emotions in faces within the frame.

        Args:
            frame: numpy array (BGR image)
            face_locations: optional list of [x1, y1, x2, y2] face bounding boxes.
                          If provided, analyzes only those regions.

        Returns:
            list of dicts:
                - emotion (str): Dominant emotion
                - confidence (float): Confidence score
                - location (list): Face bounding box [x1, y1, x2, y2]
                - all_emotions (dict): All emotion scores
        """
        if frame is None:
            return []

        if self.deepface_available:
            return self._detect_with_deepface(frame, face_locations)
        else:
            return self._placeholder_detect(frame, face_locations)

    def _detect_with_deepface(self, frame, face_locations=None):
        """Use DeepFace for real emotion detection."""
        results = []

        try:
            if face_locations:
                # Analyze specific face regions
                for loc in face_locations:
                    x1, y1, x2, y2 = loc
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0:
                        continue

                    try:
                        analysis = self.DeepFace.analyze(
                            face_crop,
                            actions=["emotion"],
                            enforce_detection=False,
                            silent=True
                        )

                        if isinstance(analysis, list):
                            analysis = analysis[0]

                        dominant = analysis.get("dominant_emotion", "neutral")
                        emotions = analysis.get("emotion", {})

                        results.append({
                            "emotion": dominant,
                            "confidence": round(emotions.get(dominant, 0) / 100, 2),
                            "location": loc,
                            "all_emotions": {
                                k: round(v / 100, 2) for k, v in emotions.items()
                            }
                        })
                    except Exception as e:
                        logger.debug(f"Emotion analysis failed for face region: {e}")
            else:
                # Analyze entire frame
                try:
                    analysis = self.DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=False,
                        silent=True
                    )

                    if isinstance(analysis, list):
                        for face in analysis:
                            dominant = face.get("dominant_emotion", "neutral")
                            emotions = face.get("emotion", {})
                            region = face.get("region", {})

                            results.append({
                                "emotion": dominant,
                                "confidence": round(emotions.get(dominant, 0) / 100, 2),
                                "location": [
                                    region.get("x", 0),
                                    region.get("y", 0),
                                    region.get("x", 0) + region.get("w", 0),
                                    region.get("y", 0) + region.get("h", 0)
                                ],
                                "all_emotions": {
                                    k: round(v / 100, 2) for k, v in emotions.items()
                                }
                            })
                except Exception as e:
                    logger.debug(f"Full frame emotion analysis failed: {e}")

        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            return self._placeholder_detect(frame, face_locations)

        return results

    def _placeholder_detect(self, frame, face_locations=None):
        """
        Placeholder when deepface is unavailable.
        Returns empty list — no fake detections.
        """
        return []

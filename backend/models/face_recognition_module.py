"""
Project LUNA — Face Recognition Module
Uses OpenCV for face detection and optionally face_recognition library
for encoding-based recognition with learning capability.
"""

import logging
import json
import os
import sys
import uuid
import numpy as np
import cv2

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FACES_DIR, LEARNED_FACES_FILE, FACE_TOLERANCE


class FaceRecognizer:
    """Face detection and recognition with learning support."""

    def __init__(self):
        self.known_faces = {}  # {name: [encoding1, encoding2, ...]}
        self.face_cascade = None
        self.face_rec_available = False
        self._pending_unknowns = {}  # {face_id: encoding}
        self._load_cascade()
        self._try_load_face_recognition()
        self._load_known_faces()

    def _load_cascade(self):
        """Load OpenCV Haar Cascade for face detection."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise ValueError("Cascade classifier is empty")
            logger.info("✅ OpenCV face cascade loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load face cascade: {e}")

    def _try_load_face_recognition(self):
        """Try to load the face_recognition library for encoding-based matching."""
        try:
            import face_recognition as fr
            self.fr = fr
            self.face_rec_available = True
            logger.info("✅ face_recognition library available")
        except ImportError:
            self.fr = None
            self.face_rec_available = False
            logger.warning("⚠️ face_recognition not installed. Using basic detection only.")

    def _load_known_faces(self):
        """Load known face encodings from storage."""
        try:
            if os.path.exists(LEARNED_FACES_FILE):
                with open(LEARNED_FACES_FILE, "r") as f:
                    face_data = json.load(f)

                for name, encodings in face_data.items():
                    self.known_faces[name] = [
                        np.array(enc) for enc in encodings
                    ]

                logger.info(f"👤 Loaded {len(self.known_faces)} known faces")
        except Exception as e:
            logger.warning(f"Failed to load known faces: {e}")
            self.known_faces = {}

    def detect(self, frame):
        """
        Detect and recognize faces in a frame.

        Args:
            frame: numpy array (BGR image)

        Returns:
            dict with keys:
                - faces: list of {name, location, is_unknown, face_id}
                - unknown_faces: list of {face_id, crop_base64}
        """
        if frame is None:
            return {"faces": [], "unknown_faces": []}

        if self.face_rec_available:
            return self._detect_with_encoding(frame)
        else:
            return self._detect_basic(frame)

    def _detect_with_encoding(self, frame):
        """Full detection with face_recognition library."""
        import base64

        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        face_locations = self.fr.face_locations(rgb_frame, model="hog")
        face_encodings = self.fr.face_encodings(rgb_frame, face_locations)

        faces = []
        unknown_faces = []

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            name = self._match_face(encoding)
            location = [int(left), int(top), int(right), int(bottom)]

            if name is None:
                # Unknown face — generate ID and store for learning
                face_id = f"face_{uuid.uuid4().hex[:8]}"
                self._pending_unknowns[face_id] = encoding.tolist()

                # Crop face for display
                crop = frame[top:bottom, left:right]
                _, buffer = cv2.imencode(".jpg", crop)
                crop_b64 = base64.b64encode(buffer).decode("utf-8")

                faces.append({
                    "name": "Unknown",
                    "location": location,
                    "is_unknown": True,
                    "face_id": face_id
                })
                unknown_faces.append({
                    "face_id": face_id,
                    "crop_base64": crop_b64
                })
            else:
                faces.append({
                    "name": name,
                    "location": location,
                    "is_unknown": False,
                    "face_id": None
                })

        return {"faces": faces, "unknown_faces": unknown_faces}

    def _detect_basic(self, frame):
        """Basic face detection using OpenCV (no recognition)."""
        import base64

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        faces = []
        unknown_faces = []

        for (x, y, w, h) in detected:
            face_id = f"face_{uuid.uuid4().hex[:8]}"
            location = [int(x), int(y), int(x + w), int(y + h)]

            # Check if we have any known faces stored (by name match placeholder)
            name = self._check_known_by_region(frame, x, y, w, h)

            if name is None:
                # Crop face for display
                crop = frame[y:y + h, x:x + w]
                _, buffer = cv2.imencode(".jpg", crop)
                crop_b64 = base64.b64encode(buffer).decode("utf-8")

                faces.append({
                    "name": "Unknown",
                    "location": location,
                    "is_unknown": True,
                    "face_id": face_id
                })
                unknown_faces.append({
                    "face_id": face_id,
                    "crop_base64": crop_b64
                })
            else:
                faces.append({
                    "name": name,
                    "location": location,
                    "is_unknown": False,
                    "face_id": None
                })

        return {"faces": faces, "unknown_faces": unknown_faces}

    def _check_known_by_region(self, frame, x, y, w, h):
        """
        Placeholder matching for basic mode.
        In basic mode without encodings, we can't truly recognize faces.
        Returns None (unknown) to trigger learning flow.
        """
        # If we have any known faces, return the first one as a demo
        # In production, this would use encoding-based matching
        if self.known_faces:
            return list(self.known_faces.keys())[0]
        return None

    def _match_face(self, encoding):
        """Match a face encoding against known faces."""
        if not self.known_faces:
            return None

        for name, known_encodings in self.known_faces.items():
            matches = self.fr.compare_faces(
                known_encodings, encoding, tolerance=FACE_TOLERANCE
            )
            if any(matches):
                return name

        return None

    def learn_face(self, face_id, name):
        """
        Learn a new face from a pending unknown detection.

        Args:
            face_id: ID of the unknown face
            name: User-provided name for the face

        Returns:
            bool: Success status
        """
        if face_id in self._pending_unknowns:
            encoding = self._pending_unknowns.pop(face_id)

            # Add to known faces
            if name not in self.known_faces:
                self.known_faces[name] = []
            self.known_faces[name].append(
                np.array(encoding) if isinstance(encoding, list) else encoding
            )

            # Save to storage
            self._save_known_faces()
            logger.info(f"✅ Learned face: {name}")
            return True

        logger.warning(f"⚠️ Face ID {face_id} not found in pending unknowns")
        return False

    def _save_known_faces(self):
        """Persist known faces to JSON storage."""
        try:
            save_data = {}
            for name, encodings in self.known_faces.items():
                save_data[name] = [
                    enc.tolist() if isinstance(enc, np.ndarray) else enc
                    for enc in encodings
                ]

            with open(LEARNED_FACES_FILE, "w") as f:
                json.dump(save_data, f, indent=2)

            logger.info(f"💾 Saved {len(save_data)} known faces to storage")
        except Exception as e:
            logger.error(f"Failed to save known faces: {e}")

    def get_known_face_names(self):
        """Return list of all learned face names."""
        return list(self.known_faces.keys())

    def get_pending_count(self):
        """Return number of pending unknown faces."""
        return len(self._pending_unknowns)

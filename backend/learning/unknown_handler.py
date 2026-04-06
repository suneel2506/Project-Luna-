"""
Project LUNA — Unknown Handler / Learning System
Manages the human-in-the-loop learning flow for unknown faces and objects.
Stores user-provided labels and makes them available for future recognition.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LEARNED_OBJECTS_FILE, LEARNED_FACES_FILE, FACES_DIR


class UnknownHandler:
    """Manages unknown detections and user-driven learning."""

    def __init__(self):
        self.pending_objects = {}  # {obj_id: {crop_b64, timestamp}}
        self.pending_faces = {}   # {face_id: {encoding, crop_b64, timestamp}}
        self.learned_objects = self._load_json(LEARNED_OBJECTS_FILE)
        self.learned_faces_meta = self._load_json(LEARNED_FACES_FILE)

    def _load_json(self, filepath):
        """Load a JSON file, returning empty dict if not found."""
        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
        return {}

    def _save_json(self, filepath, data):
        """Save data to JSON file."""
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")

    # ---- Object Learning ----

    def flag_unknown_object(self, crop_b64, label_suggestion=None):
        """
        Flag an unknown object for user labeling.

        Args:
            crop_b64: Base64 encoded image crop of the unknown object
            label_suggestion: Optional suggested label

        Returns:
            str: Object ID for reference in learning flow
        """
        obj_id = f"obj_{uuid.uuid4().hex[:8]}"
        self.pending_objects[obj_id] = {
            "crop_b64": crop_b64,
            "suggestion": label_suggestion,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"🔍 Flagged unknown object: {obj_id}")
        return obj_id

    def learn_object(self, obj_id, label):
        """
        Learn a user-provided label for an unknown object.

        Args:
            obj_id: ID of the pending unknown object
            label: User-provided label

        Returns:
            bool: Success status
        """
        if obj_id not in self.pending_objects:
            # Also accept learning without pending (direct label)
            self.learned_objects[label] = {
                "learned_at": datetime.now().isoformat(),
                "source": "direct"
            }
            self._save_json(LEARNED_OBJECTS_FILE, self.learned_objects)
            logger.info(f"📦 Learned new object (direct): {label}")
            return True

        pending = self.pending_objects.pop(obj_id)
        self.learned_objects[label] = {
            "learned_at": datetime.now().isoformat(),
            "source": "detection",
            "original_id": obj_id
        }
        self._save_json(LEARNED_OBJECTS_FILE, self.learned_objects)
        logger.info(f"📦 Learned new object: {label} (from {obj_id})")
        return True

    # ---- Face Learning ----

    def flag_unknown_face(self, encoding, crop_b64):
        """
        Flag an unknown face for user labeling.

        Args:
            encoding: Face encoding (list or numpy array)
            crop_b64: Base64 encoded face crop

        Returns:
            str: Face ID for reference in learning flow
        """
        face_id = f"face_{uuid.uuid4().hex[:8]}"
        self.pending_faces[face_id] = {
            "encoding": encoding if isinstance(encoding, list) else encoding.tolist() if hasattr(encoding, 'tolist') else encoding,
            "crop_b64": crop_b64,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"👤 Flagged unknown face: {face_id}")
        return face_id

    def learn_face(self, face_id, name):
        """
        Learn a user-provided name for an unknown face.

        Args:
            face_id: ID of the pending unknown face
            name: User-provided name

        Returns:
            dict: {success: bool, encoding: list or None}
        """
        if face_id not in self.pending_faces:
            logger.warning(f"Face ID {face_id} not found in pending")
            return {"success": False, "encoding": None}

        pending = self.pending_faces.pop(face_id)
        encoding = pending.get("encoding")

        # Update faces metadata
        if name not in self.learned_faces_meta:
            self.learned_faces_meta[name] = {
                "first_seen": datetime.now().isoformat(),
                "times_seen": 0
            }
        self.learned_faces_meta[name]["times_seen"] = \
            self.learned_faces_meta[name].get("times_seen", 0) + 1
        self.learned_faces_meta[name]["last_seen"] = datetime.now().isoformat()

        self._save_json(LEARNED_FACES_FILE, self.learned_faces_meta)
        logger.info(f"👤 Learned face: {name} (from {face_id})")

        return {"success": True, "encoding": encoding}

    # ---- Query Methods ----

    def get_pending_objects(self):
        """Get all pending unknown objects."""
        return self.pending_objects

    def get_pending_faces(self):
        """Get all pending unknown faces."""
        return {
            fid: {"crop_b64": data.get("crop_b64"), "timestamp": data.get("timestamp")}
            for fid, data in self.pending_faces.items()
        }

    def get_learned_summary(self):
        """Get summary of all learned items."""
        return {
            "objects_count": len(self.learned_objects),
            "faces_count": len(self.learned_faces_meta),
            "objects": list(self.learned_objects.keys()),
            "faces": list(self.learned_faces_meta.keys()),
            "pending_objects": len(self.pending_objects),
            "pending_faces": len(self.pending_faces)
        }

    def is_object_known(self, label):
        """Check if an object label has been learned."""
        return label.lower() in [k.lower() for k in self.learned_objects.keys()]

    def clear_stale_pending(self, max_age_seconds=300):
        """Remove pending items older than max_age_seconds."""
        now = datetime.now()
        stale_objects = []
        stale_faces = []

        for oid, data in self.pending_objects.items():
            ts = datetime.fromisoformat(data["timestamp"])
            if (now - ts).total_seconds() > max_age_seconds:
                stale_objects.append(oid)

        for fid, data in self.pending_faces.items():
            ts = datetime.fromisoformat(data["timestamp"])
            if (now - ts).total_seconds() > max_age_seconds:
                stale_faces.append(fid)

        for oid in stale_objects:
            del self.pending_objects[oid]
        for fid in stale_faces:
            del self.pending_faces[fid]

        if stale_objects or stale_faces:
            logger.info(
                f"🧹 Cleared {len(stale_objects)} stale objects, "
                f"{len(stale_faces)} stale faces"
            )

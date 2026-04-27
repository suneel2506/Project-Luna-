"""
Project LUNA — Unknown Handler / Learning System
═════════════════════════════════════════════════
Manages the human-in-the-loop learning flow for unknown faces and objects.
Stores user-provided labels and makes them available for future recognition.

Features:
  • Thread-safe JSON file access
  • Automatic stale-pending cleanup per configurable TTL
  • Learning history with timestamps and metadata
  • Undo / delete learned item support
  • Per-module storage directories
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from config import (
    LEARNED_OBJECTS_FILE,
    LEARNED_FACES_FILE,
    FACE_IMAGES_DIR,
    PENDING_TTL_SECONDS,
)

logger = logging.getLogger("luna.unknown_handler")


class UnknownHandler:
    """Manages unknown detections and user-driven learning."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pending_objects: dict[str, dict] = {}
        self.pending_faces: dict[str, dict] = {}
        self.learned_objects: dict[str, dict] = self._load_json(LEARNED_OBJECTS_FILE)
        self.learned_faces_meta: dict[str, dict] = self._load_json(LEARNED_FACES_FILE)

    # ════════════════════════════════════════════
    # JSON I/O
    # ════════════════════════════════════════════

    @staticmethod
    def _load_json(filepath: Path | str) -> dict:
        """Load a JSON file, returning empty dict on any failure."""
        path = Path(filepath)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("Failed to load %s: %s", filepath, exc)
            return {}

    def _save_json(self, filepath: Path | str, data: dict) -> None:
        """Thread-safe write of *data* to *filepath*."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                with path.open("w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, default=str)
            except Exception as exc:
                logger.error("Failed to save %s: %s", filepath, exc)

    # ════════════════════════════════════════════
    # Object Learning
    # ════════════════════════════════════════════

    def flag_unknown_object(
        self,
        crop_b64: str = "",
        label_suggestion: str | None = None,
    ) -> str:
        """Flag an unknown object for user labeling. Returns object ID."""
        obj_id = f"obj_{uuid.uuid4().hex[:8]}"
        self.pending_objects[obj_id] = {
            "crop_b64": crop_b64,
            "suggestion": label_suggestion,
            "timestamp": datetime.now().isoformat(),
        }
        logger.info("🔍 Flagged unknown object: %s", obj_id)
        return obj_id

    def learn_object(self, obj_id: str | None, label: str) -> bool:
        """Learn a user-provided label for an unknown object."""
        label = label.strip()
        if not label:
            return False

        source = "direct"
        if obj_id and obj_id in self.pending_objects:
            self.pending_objects.pop(obj_id)
            source = "detection"

        self.learned_objects[label] = {
            "learned_at": datetime.now().isoformat(),
            "source": source,
            "original_id": obj_id,
        }
        self._save_json(LEARNED_OBJECTS_FILE, self.learned_objects)
        logger.info("📦 Learned object: %s (source=%s)", label, source)
        return True

    # ════════════════════════════════════════════
    # Face Learning
    # ════════════════════════════════════════════

    def flag_unknown_face(self, encoding=None, crop_b64: str = "") -> str:
        """Flag an unknown face for user labeling. Returns allocated face_id."""
        face_id = f"face_{uuid.uuid4().hex[:8]}"
        self._store_pending_face(face_id, encoding, crop_b64)
        return face_id

    def flag_unknown_face_with_id(
        self,
        face_id: str,
        encoding=None,
        crop_b64: str = "",
    ) -> str:
        """Flag an unknown face reusing the ID from FaceRecognizer."""
        self._store_pending_face(face_id, encoding, crop_b64)
        return face_id

    def _store_pending_face(self, face_id: str, encoding, crop_b64: str) -> None:
        """Internal helper to stash a pending face."""
        enc_list: list = (
            encoding.tolist() if hasattr(encoding, "tolist") else
            list(encoding) if encoding else []
        )
        self.pending_faces[face_id] = {
            "encoding": enc_list,
            "crop_b64": crop_b64,
            "timestamp": datetime.now().isoformat(),
        }
        logger.info("👤 Flagged unknown face: %s", face_id)

    def learn_face(self, face_id: str | None, name: str) -> dict:
        """
        Learn a user-provided name for an unknown face.

        Returns:
            {"success": bool, "encoding": list | None}
        """
        name = name.strip()
        if not name:
            return {"success": False, "encoding": None}

        if not face_id or face_id not in self.pending_faces:
            logger.warning("Face ID %s not found in pending", face_id)
            return {"success": False, "encoding": None}

        pending = self.pending_faces.pop(face_id)
        encoding = pending.get("encoding")

        # Update metadata store
        if name not in self.learned_faces_meta:
            self.learned_faces_meta[name] = {
                "first_seen": datetime.now().isoformat(),
                "times_seen": 0,
            }
        self.learned_faces_meta[name]["times_seen"] = (
            self.learned_faces_meta[name].get("times_seen", 0) + 1
        )
        self.learned_faces_meta[name]["last_seen"] = datetime.now().isoformat()

        self._save_json(LEARNED_FACES_FILE, self.learned_faces_meta)
        logger.info("👤 Learned face: %s (from %s)", name, face_id)

        return {"success": True, "encoding": encoding}

    # ════════════════════════════════════════════
    # Delete / Undo
    # ════════════════════════════════════════════

    def delete_learned_object(self, label: str) -> bool:
        """Remove a previously learned object label."""
        if label in self.learned_objects:
            del self.learned_objects[label]
            self._save_json(LEARNED_OBJECTS_FILE, self.learned_objects)
            logger.info("🗑️  Deleted learned object: %s", label)
            return True
        return False

    def delete_learned_face(self, name: str) -> bool:
        """Remove a previously learned face from metadata."""
        if name in self.learned_faces_meta:
            del self.learned_faces_meta[name]
            self._save_json(LEARNED_FACES_FILE, self.learned_faces_meta)
            logger.info("🗑️  Deleted learned face: %s", name)
            return True
        return False

    # ════════════════════════════════════════════
    # Query Helpers
    # ════════════════════════════════════════════

    def get_pending_objects(self) -> dict:
        """All pending unknown objects."""
        return dict(self.pending_objects)

    def get_pending_faces(self) -> dict:
        """All pending unknown faces (sans encodings for safety)."""
        return {
            fid: {"crop_b64": d.get("crop_b64", ""), "timestamp": d.get("timestamp")}
            for fid, d in self.pending_faces.items()
        }

    def get_learned_summary(self) -> dict:
        """Summary of all learned + pending items."""
        return {
            "objects_count": len(self.learned_objects),
            "faces_count": len(self.learned_faces_meta),
            "objects": list(self.learned_objects.keys()),
            "faces": list(self.learned_faces_meta.keys()),
            "pending_objects": len(self.pending_objects),
            "pending_faces": len(self.pending_faces),
        }

    def is_object_known(self, label: str) -> bool:
        """Check if an object label has been learned (case-insensitive)."""
        lower_keys = {k.lower() for k in self.learned_objects}
        return label.lower() in lower_keys

    # ════════════════════════════════════════════
    # Stale Cleanup
    # ════════════════════════════════════════════

    def clear_stale_pending(self, max_age_seconds: int | None = None) -> int:
        """Remove pending items older than *max_age_seconds*."""
        ttl = max_age_seconds if max_age_seconds is not None else PENDING_TTL_SECONDS
        cutoff = datetime.now() - timedelta(seconds=ttl)
        removed = 0

        for store in (self.pending_objects, self.pending_faces):
            stale = [
                key for key, data in store.items()
                if datetime.fromisoformat(data["timestamp"]) < cutoff
            ]
            for key in stale:
                del store[key]
            removed += len(stale)

        if removed:
            logger.info("🧹 Cleared %d stale pending item(s)", removed)

        return removed

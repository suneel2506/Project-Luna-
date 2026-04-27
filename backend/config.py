"""
Project LUNA — Configuration Settings
══════════════════════════════════════
Centralized configuration for every module in the backend.
Uses pathlib for cross-platform path handling.
Requires Python 3.11.9+
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

# ──────────────────────────────────────────────
# Python Version Gate
# ──────────────────────────────────────────────
REQUIRED_PYTHON: tuple[int, int, int] = (3, 11, 9)

if sys.version_info < REQUIRED_PYTHON:
    import warnings
    warnings.warn(
        f"Project LUNA recommends Python {'.'.join(map(str, REQUIRED_PYTHON))}+ "
        f"but found {sys.version}. Some features may not work optimally.",
        RuntimeWarning,
        stacklevel=2,
    )

# ──────────────────────────────────────────────
# Directory Layout
# ──────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
STORAGE_DIR: Path = BASE_DIR / "storage"

# ── Per-Module Storage ──
# Faces
FACES_DIR: Path = STORAGE_DIR / "faces"
FACE_IMAGES_DIR: Path = FACES_DIR / "images"        # {name}/*.jpg crops
LBPH_MODEL_FILE: Path = FACES_DIR / "lbph_model.yml"
FACE_LABELS_FILE: Path = FACES_DIR / "labels.json"  # {id: name} mapping
LEARNED_FACES_FILE: Path = FACES_DIR / "learned_faces.json"

# Objects
OBJECTS_DIR: Path = STORAGE_DIR / "objects"
LEARNED_OBJECTS_FILE: Path = OBJECTS_DIR / "learned.json"

# Emotions
EMOTIONS_DIR: Path = STORAGE_DIR / "emotions"
EMOTION_HISTORY_FILE: Path = EMOTIONS_DIR / "history.json"

# Signs
SIGNS_DIR: Path = STORAGE_DIR / "signs"
SIGN_HISTORY_FILE: Path = SIGNS_DIR / "history.json"

# Ensure all storage directories exist at import time
for _dir in (FACE_IMAGES_DIR, OBJECTS_DIR, EMOTIONS_DIR, SIGNS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATE_FORMAT: str = "%H:%M:%S"

# ──────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────
FLASK_HOST: str = os.getenv("LUNA_HOST", "0.0.0.0")
FLASK_PORT: int = int(os.getenv("LUNA_PORT", "5000"))
DEBUG: bool = os.getenv("LUNA_DEBUG", "true").lower() in ("1", "true", "yes")

CORS_ORIGINS = [
    "https://project-luna-opal.vercel.app"
]

# ──────────────────────────────────────────────
# Detection Confidence Thresholds
# ──────────────────────────────────────────────
YOLO_CONFIDENCE: float = 0.35        # Lowered from 0.45 for better detection
FACE_TOLERANCE: float = 0.6          # Lower = stricter matching
LBPH_CONFIDENCE_THRESHOLD: float = 80.0  # LBPH: lower = better match
EMOTION_CONFIDENCE: float = 0.4      # Lowered from 0.5

# ──────────────────────────────────────────────
# Model Paths
# ──────────────────────────────────────────────
YOLO_MODEL: str = "yolov8n.pt"       # Auto-downloads on first run

# ──────────────────────────────────────────────
# Rate Limiting / Throttle
# ──────────────────────────────────────────────
MIN_FRAME_INTERVAL_MS: int = 500     # Ignore frames faster than this
PENDING_TTL_SECONDS: int = 300       # Stale pending items expire after 5 min

# ──────────────────────────────────────────────
# Conversation Context
# ──────────────────────────────────────────────
MAX_CONTEXT_HISTORY: int = 20        # Messages kept in context window
CONTEXT_TTL_SECONDS: int = 600       # Context expires after 10 min idle
DEDUP_WINDOW_SECONDS: int = 10       # Suppress duplicate messages within window

# ──────────────────────────────────────────────
# Sign Language
# ──────────────────────────────────────────────
SUPPORTED_SIGNS: list[str] = [
    "hello", "yes", "no", "thank_you", "help", "i_love_you",
]

# ──────────────────────────────────────────────
# Supported Languages
# ──────────────────────────────────────────────
LANGUAGES: dict[str, str] = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
}

"""
Project LUNA — Configuration Settings
Centralized configuration for all modules.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Storage paths
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
FACES_DIR = os.path.join(STORAGE_DIR, "faces")
LEARNED_OBJECTS_FILE = os.path.join(STORAGE_DIR, "learned_objects.json")
LEARNED_FACES_FILE = os.path.join(STORAGE_DIR, "learned_faces.json")

# Ensure storage directories exist
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# Detection confidence thresholds
YOLO_CONFIDENCE = 0.45
FACE_TOLERANCE = 0.6  # Lower = stricter matching
EMOTION_CONFIDENCE = 0.5

# Camera settings
FRAME_INTERVAL_MS = 2000  # Process frame every 2 seconds

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
DEBUG = True

# CORS origins
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

# Model paths
YOLO_MODEL = "yolov8n.pt"  # Will auto-download on first use

# Sign language gesture set
SUPPORTED_SIGNS = ["hello", "yes", "no", "thank_you", "help", "i_love_you"]

# Supported languages
LANGUAGES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi"
}

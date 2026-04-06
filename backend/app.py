"""
Project LUNA — Flask Application
Main API server for the Interactive AI Vision Assistant.

Endpoints:
    POST /api/process   → Process a camera frame through all detectors
    POST /api/learn     → Submit user label for unknown detection
    GET  /api/status    → Health check
    GET  /api/history   → Get learned items summary
"""

import base64
import logging
import sys
import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LUNA")

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FLASK_HOST, FLASK_PORT, DEBUG, CORS_ORIGINS
from models.object_detection import ObjectDetector
from models.face_recognition_module import FaceRecognizer
from models.emotion_detection import EmotionDetector
from models.sign_detection import SignDetector
from learning.unknown_handler import UnknownHandler
from utils.translator import Translator
from utils.response_generator import ResponseGenerator

# ---- Initialize Flask App ----
app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)

# ---- Initialize Modules ----
logger.info("🌙 Initializing LUNA modules...")

object_detector = ObjectDetector()
face_recognizer = FaceRecognizer()
emotion_detector = EmotionDetector()
sign_detector = SignDetector()
unknown_handler = UnknownHandler()
translator = Translator()
response_generator = ResponseGenerator()

logger.info("✅ All LUNA modules initialized!")


def decode_base64_image(b64_string):
    """Decode a base64 image string to a numpy array (BGR)."""
    try:
        # Remove data URL prefix if present
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/api/status", methods=["GET"])
def status():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "name": "LUNA — AI Vision Assistant",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "object_detection": "yolo" if object_detector.model else "placeholder",
            "face_recognition": "encoding" if face_recognizer.face_rec_available else "cascade",
            "emotion_detection": "deepface" if emotion_detector.deepface_available else "placeholder",
            "sign_detection": "mediapipe" if sign_detector.mediapipe_available else "placeholder",
        },
        "learned": unknown_handler.get_learned_summary()
    })


@app.route("/api/process", methods=["POST"])
def process_frame():
    """
    Process a camera frame through all AI detectors.

    Expects JSON body:
        {
            "image": "base64_encoded_image",
            "language": "en" | "ta" | "hi",
            "detectors": {
                "objects": true,
                "faces": true,
                "emotions": true,
                "signs": true
            }
        }

    Returns:
        {
            "objects": [...],
            "faces": [...],
            "emotions": [...],
            "signs": [...],
            "unknown_faces": [...],
            "message": "...",
            "translations": {"en": "...", "ta": "...", "hi": "..."},
            "timestamp": "..."
        }
    """
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode the image
        frame = decode_base64_image(data["image"])
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        language = data.get("language", "en")
        detectors = data.get("detectors", {
            "objects": True,
            "faces": True,
            "emotions": True,
            "signs": True
        })

        # ---- Run detectors ----
        results = {
            "objects": [],
            "faces": [],
            "emotions": [],
            "signs": [],
            "unknown_faces": [],
            "timestamp": datetime.now().isoformat()
        }

        # Object Detection
        if detectors.get("objects", True):
            results["objects"] = object_detector.detect(frame)

        # Face Recognition
        face_results = {"faces": [], "unknown_faces": []}
        if detectors.get("faces", True):
            face_results = face_recognizer.detect(frame)
            results["faces"] = face_results.get("faces", [])
            results["unknown_faces"] = face_results.get("unknown_faces", [])

            # Flag unknown faces for learning
            for uf in results["unknown_faces"]:
                unknown_handler.flag_unknown_face(
                    encoding=uf.get("encoding", []),
                    crop_b64=uf.get("crop_base64", "")
                )

        # Emotion Detection
        if detectors.get("emotions", True):
            face_locations = [
                f["location"] for f in results["faces"]
            ] if results["faces"] else None
            results["emotions"] = emotion_detector.detect(frame, face_locations)

        # Sign Language Detection
        if detectors.get("signs", True):
            results["signs"] = sign_detector.detect(frame)

        # ---- Generate response message ----
        message = response_generator.generate(
            objects=results["objects"],
            faces=face_results,
            emotions=results["emotions"],
            signs=results["signs"]
        )
        results["message"] = message

        # ---- Translate ----
        results["translations"] = {
            "en": message,
            "ta": translator.translate_message(message, "ta"),
            "hi": translator.translate_message(message, "hi"),
        }

        # ---- Add translated labels if requested language isn't English ----
        if language != "en":
            translated = translator.translate_detection_results(results, language)
            results.update(translated)

        # Clean response for JSON serialization
        _clean_for_json(results)

        return jsonify(results)

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/learn", methods=["POST"])
def learn():
    """
    Accept user label for unknown detection.

    Expects JSON body:
        {
            "type": "face" | "object",
            "id": "face_xxxx" | "obj_xxxx",
            "label": "User provided name/label"
        }

    Returns:
        {
            "success": true,
            "message": "Confirmation message",
            "learned_summary": {...}
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        item_type = data.get("type")
        item_id = data.get("id")
        label = data.get("label", "").strip()

        if not item_type or not label:
            return jsonify({"error": "Missing 'type' or 'label'"}), 400

        success = False

        if item_type == "face":
            # Learn the face
            result = unknown_handler.learn_face(item_id, label)
            if result["success"] and result["encoding"]:
                # Also update face recognizer's known faces
                face_recognizer.learn_face(item_id, label)
            success = result["success"]

        elif item_type == "object":
            success = unknown_handler.learn_object(item_id, label)
            if success:
                object_detector.reload_learned_objects()

        else:
            return jsonify({"error": "Invalid type. Use 'face' or 'object'"}), 400

        if success:
            message = response_generator.generate_learning_response(item_type, label)
            return jsonify({
                "success": True,
                "message": message,
                "learned_summary": unknown_handler.get_learned_summary()
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Could not learn {item_type}. ID may have expired.",
            }), 404

    except Exception as e:
        logger.error(f"Learning error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def history():
    """Get summary of all learned items."""
    return jsonify(unknown_handler.get_learned_summary())


@app.route("/api/languages", methods=["GET"])
def languages():
    """Get supported languages."""
    return jsonify(translator.get_supported_languages())


def _clean_for_json(obj):
    """Recursively clean numpy types for JSON serialization."""
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = _clean_for_json(obj[key])
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _clean_for_json(obj[i])
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    logger.info(f"""
    ╔══════════════════════════════════════╗
    ║  🌙 LUNA — AI Vision Assistant      ║
    ║  Running on {FLASK_HOST}:{FLASK_PORT}            ║
    ╚══════════════════════════════════════╝
    """)
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=DEBUG
    )

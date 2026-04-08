"""
Project LUNA — Flask Application
═════════════════════════════════
Main API server for the Interactive AI Vision Assistant.
Requires Python 3.11.9+

Endpoints:
    POST /api/process   → Process a camera frame through all detectors
    POST /api/learn     → Submit user label for unknown detection
    GET  /api/status    → Health check / module status
    GET  /api/history   → Learned items summary
    GET  /api/languages → Supported languages list
"""

from __future__ import annotations

import base64
import logging
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from config import (
    FLASK_HOST,
    FLASK_PORT,
    DEBUG,
    CORS_ORIGINS,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    MIN_FRAME_INTERVAL_MS,
)
from models.object_detection import ObjectDetector
from models.face_recognition_module import FaceRecognizer
from models.emotion_detection import EmotionDetector
from models.sign_detection import SignDetector
from learning.unknown_handler import UnknownHandler
from utils.translator import Translator
from utils.response_generator import ResponseGenerator


# ════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)
logger = logging.getLogger("LUNA")


# ════════════════════════════════════════════════
# Application Factory
# ════════════════════════════════════════════════

def create_app() -> Flask:
    """Create, configure, and return the Flask application."""

    app = Flask(__name__)

    # ── CORS ──
    CORS(app, origins=CORS_ORIGINS)

    # ── Initialise AI modules ──
    logger.info("🌙 Initializing LUNA modules...")
    modules = _init_modules()
    app.config["LUNA_MODULES"] = modules
    logger.info("✅ All LUNA modules initialized!")

    # ── Frame-rate throttle state ──
    app.config["_last_process_time"] = 0.0

    # ── Register routes ──
    _register_routes(app, modules)

    # ── Register error handlers ──
    _register_error_handlers(app)

    return app


# ════════════════════════════════════════════════
# Module Initialisation
# ════════════════════════════════════════════════

class _Modules:
    """Simple container so we can pass all modules around easily."""

    __slots__ = (
        "object_detector",
        "face_recognizer",
        "emotion_detector",
        "sign_detector",
        "unknown_handler",
        "translator",
        "response_generator",
    )

    def __init__(self) -> None:
        self.object_detector = ObjectDetector()
        self.face_recognizer = FaceRecognizer()
        self.emotion_detector = EmotionDetector()
        self.sign_detector = SignDetector()
        self.unknown_handler = UnknownHandler()
        self.translator = Translator()
        self.response_generator = ResponseGenerator()


def _init_modules() -> _Modules:
    return _Modules()


# ════════════════════════════════════════════════
# Route Registration
# ════════════════════════════════════════════════

def _register_routes(app: Flask, m: _Modules) -> None:
    """Attach all API endpoints to *app*."""

    # ────────────────────────────────────────────
    # GET /api/status
    # ────────────────────────────────────────────
    @app.route("/api/status", methods=["GET"])
    def status() -> Response:
        """Health check with module availability info."""
        return jsonify({
            "status": "online",
            "name": "LUNA — AI Vision Assistant",
            "version": "2.0.0",
            "python_version": sys.version,
            "timestamp": datetime.now().isoformat(),
            "modules": {
                "object_detection": "yolo" if m.object_detector.model else "placeholder",
                "face_recognition": (
                    "encoding" if m.face_recognizer.face_rec_available else "cascade"
                ),
                "emotion_detection": (
                    "deepface" if m.emotion_detector.deepface_available else "placeholder"
                ),
                "sign_detection": (
                    "mediapipe" if m.sign_detector.mediapipe_available else "placeholder"
                ),
            },
            "learned": m.unknown_handler.get_learned_summary(),
        })

    # ────────────────────────────────────────────
    # POST /api/process
    # ────────────────────────────────────────────
    @app.route("/api/process", methods=["POST"])
    def process_frame() -> tuple[Response, int] | Response:
        """
        Process a camera frame through all AI detectors.

        Expects JSON:
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
        """
        # ── Frame-rate throttle ──
        now = time.time()
        last = app.config.get("_last_process_time", 0.0)
        if (now - last) * 1000 < MIN_FRAME_INTERVAL_MS:
            return jsonify({"skipped": True, "reason": "throttled"}), 429

        app.config["_last_process_time"] = now

        # ── Parse request ──
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        frame = _decode_base64_image(data["image"])
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        language: str = data.get("language", "en")
        detectors: dict = data.get("detectors", {
            "objects": True,
            "faces": True,
            "emotions": True,
            "signs": True,
        })

        # ── Run detectors ──
        results: dict = {
            "objects": [],
            "faces": [],
            "emotions": [],
            "signs": [],
            "unknown_faces": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Object Detection
        if detectors.get("objects", True):
            results["objects"] = m.object_detector.detect(frame)

        # Face Recognition
        face_results: dict = {"faces": [], "unknown_faces": []}
        if detectors.get("faces", True):
            face_results = m.face_recognizer.detect(frame)
            results["faces"] = face_results.get("faces", [])
            results["unknown_faces"] = face_results.get("unknown_faces", [])

            # Flag unknown faces for the learning system (keep IDs in sync)
            for uf in results["unknown_faces"]:
                face_id = uf.get("face_id")
                encoding = m.face_recognizer.get_pending_encoding(face_id)
                m.unknown_handler.flag_unknown_face_with_id(
                    face_id=face_id,
                    encoding=encoding if encoding is not None else [],
                    crop_b64=uf.get("crop_base64", ""),
                )

        # Emotion Detection
        if detectors.get("emotions", True):
            face_locs = [
                f["location"] for f in results["faces"]
            ] if results["faces"] else None
            results["emotions"] = m.emotion_detector.detect(frame, face_locs)

        # Sign Language Detection
        if detectors.get("signs", True):
            results["signs"] = m.sign_detector.detect(frame)

        # ── Generate response message ──
        message: str = m.response_generator.generate(
            objects=results["objects"],
            faces=face_results,
            emotions=results["emotions"],
            signs=results["signs"],
        )
        results["message"] = message

        # ── Translations ──
        results["translations"] = {
            "en": message,
            "ta": m.translator.translate_message(message, "ta"),
            "hi": m.translator.translate_message(message, "hi"),
        }

        # Add translated labels if requested language isn't English
        if language != "en":
            translated = m.translator.translate_detection_results(results, language)
            results.update(translated)

        # ── Serialise & respond ──
        _clean_for_json(results)
        return jsonify(results)

    # ────────────────────────────────────────────
    # POST /api/learn
    # ────────────────────────────────────────────
    @app.route("/api/learn", methods=["POST"])
    def learn() -> tuple[Response, int] | Response:
        """
        Accept a user-provided label for an unknown detection.

        Expects JSON:
            {
                "type": "face" | "object",
                "id": "face_xxxx" | "obj_xxxx",
                "label": "User provided name/label"
            }
        """
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No data provided"}), 400

        item_type: str = data.get("type", "")
        item_id: str | None = data.get("id")
        label: str = data.get("label", "").strip()

        if not item_type or not label:
            return jsonify({"error": "Missing 'type' or 'label'"}), 400

        success: bool = False

        match item_type:
            case "face":
                result = m.unknown_handler.learn_face(item_id, label)
                if result["success"] and result["encoding"]:
                    m.face_recognizer.add_known_face(label, result["encoding"])
                success = result["success"]

            case "object":
                success = m.unknown_handler.learn_object(item_id, label)
                if success:
                    m.object_detector.reload_learned_objects()

            case _:
                return jsonify({"error": "Invalid type. Use 'face' or 'object'"}), 400

        if success:
            message = m.response_generator.generate_learning_response(item_type, label)
            return jsonify({
                "success": True,
                "message": message,
                "learned_summary": m.unknown_handler.get_learned_summary(),
            })

        return jsonify({
            "success": False,
            "message": f"Could not learn {item_type}. ID may have expired.",
        }), 404

    # ────────────────────────────────────────────
    # GET /api/history
    # ────────────────────────────────────────────
    @app.route("/api/history", methods=["GET"])
    def history() -> Response:
        """Summary of all learned items."""
        return jsonify(m.unknown_handler.get_learned_summary())

    # ────────────────────────────────────────────
    # GET /api/languages
    # ────────────────────────────────────────────
    @app.route("/api/languages", methods=["GET"])
    def languages() -> Response:
        """Supported languages."""
        return jsonify(m.translator.get_supported_languages())


# ════════════════════════════════════════════════
# Error Handlers
# ════════════════════════════════════════════════

def _register_error_handlers(app: Flask) -> None:
    """Global JSON error handlers."""

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad request", "detail": str(error)}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error("Internal server error: %s", error, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# ════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════

def _decode_base64_image(b64_string: str) -> np.ndarray | None:
    """Decode a base64-encoded image (with optional data-URL prefix) to BGR ndarray."""
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:
        logger.error("Failed to decode image: %s", exc)
        return None


def _clean_for_json(obj):
    """Recursively convert numpy types to native Python for JSON serialisation."""
    match obj:
        case dict():
            for key in obj:
                obj[key] = _clean_for_json(obj[key])
        case list():
            for i in range(len(obj)):
                obj[i] = _clean_for_json(obj[i])
        case np.integer():
            return int(obj)
        case np.floating():
            return float(obj)
        case np.ndarray():
            return obj.tolist()
    return obj


# ════════════════════════════════════════════════
# Entrypoint
# ════════════════════════════════════════════════

app = create_app()

if __name__ == "__main__":
    logger.info(
        "\n"
        "    ╔══════════════════════════════════════╗\n"
        "    ║  🌙 LUNA — AI Vision Assistant v2.0  ║\n"
        "    ║  Python %-28s ║\n"
        "    ║  Running on %s:%-18d ║\n"
        "    ╚══════════════════════════════════════╝",
        sys.version.split()[0],
        FLASK_HOST,
        FLASK_PORT,
    )
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=DEBUG,
    )

"""
Project LUNA — Test Fixtures
"""

import sys
import os
import base64

import pytest
import numpy as np

# Ensure the backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def flask_app():
    """Create the Flask app once for the entire test session."""
    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    # Disable throttle for tests
    app.config["_last_process_time"] = 0.0
    return app


@pytest.fixture()
def client(flask_app):
    """Fresh test client per test — resets throttle timer."""
    flask_app.config["_last_process_time"] = 0.0
    return flask_app.test_client()


@pytest.fixture()
def test_image_b64() -> str:
    """A simple 200×200 synthetic image encoded as base64 JPEG."""
    import cv2
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = 255  # white square
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

"""
Project LUNA — Backend API Tests
Basic tests for the Flask API endpoints.
"""

import sys
import os
import json
import base64
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_image():
    """Create a simple test image as base64."""
    # Create a 200x200 black image with a white rectangle
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = 255  # White square
    import cv2
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


def test_status():
    """Test the status endpoint."""
    from app import app
    client = app.test_client()
    response = client.get('/api/status')
    data = json.loads(response.data)

    assert response.status_code == 200
    assert data['status'] == 'online'
    assert 'modules' in data
    print("✅ Status endpoint: PASS")


def test_process():
    """Test the process endpoint."""
    from app import app
    client = app.test_client()

    test_image = create_test_image()
    response = client.post('/api/process', json={
        "image": test_image,
        "language": "en",
        "detectors": {
            "objects": True,
            "faces": True,
            "emotions": True,
            "signs": True
        }
    })
    data = json.loads(response.data)

    assert response.status_code == 200
    assert 'message' in data
    assert 'translations' in data
    assert 'objects' in data
    assert 'faces' in data
    print("✅ Process endpoint: PASS")
    print(f"   Message: {data['message'][:80]}...")


def test_process_no_image():
    """Test process endpoint with no image."""
    from app import app
    client = app.test_client()
    response = client.post('/api/process', json={})
    assert response.status_code == 400
    print("✅ Process (no image) validation: PASS")


def test_learn():
    """Test the learn endpoint."""
    from app import app
    client = app.test_client()

    response = client.post('/api/learn', json={
        "type": "object",
        "id": "obj_test123",
        "label": "water bottle"
    })
    data = json.loads(response.data)

    assert response.status_code == 200
    assert data['success'] is True
    assert 'message' in data
    print("✅ Learn endpoint: PASS")
    print(f"   Message: {data['message']}")


def test_history():
    """Test the history endpoint."""
    from app import app
    client = app.test_client()
    response = client.get('/api/history')
    data = json.loads(response.data)

    assert response.status_code == 200
    assert 'objects_count' in data
    assert 'faces_count' in data
    print("✅ History endpoint: PASS")


def test_languages():
    """Test the languages endpoint."""
    from app import app
    client = app.test_client()
    response = client.get('/api/languages')
    data = json.loads(response.data)

    assert response.status_code == 200
    assert 'en' in data
    assert 'ta' in data
    assert 'hi' in data
    print("✅ Languages endpoint: PASS")


if __name__ == "__main__":
    print("🌙 Running LUNA API Tests...\n")
    test_status()
    test_process()
    test_process_no_image()
    test_learn()
    test_history()
    test_languages()
    print("\n✅ All tests passed!")

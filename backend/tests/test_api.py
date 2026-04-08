"""
Project LUNA — Backend API Tests
═════════════════════════════════
Comprehensive pytest suite for all API endpoints.
Uses shared fixtures from conftest.py.
"""

import json


# ════════════════════════════════════════════════
# GET /api/status
# ════════════════════════════════════════════════

class TestStatus:
    """Health-check endpoint."""

    def test_returns_200(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_status_online(self, client):
        data = resp_json(client.get("/api/status"))
        assert data["status"] == "online"

    def test_contains_modules(self, client):
        data = resp_json(client.get("/api/status"))
        assert "modules" in data
        for key in ("object_detection", "face_recognition", "emotion_detection", "sign_detection"):
            assert key in data["modules"]

    def test_contains_version(self, client):
        data = resp_json(client.get("/api/status"))
        assert "version" in data
        assert data["version"].startswith("2.")

    def test_contains_learned_summary(self, client):
        data = resp_json(client.get("/api/status"))
        assert "learned" in data
        assert "objects_count" in data["learned"]
        assert "faces_count" in data["learned"]


# ════════════════════════════════════════════════
# POST /api/process
# ════════════════════════════════════════════════

class TestProcess:
    """Frame processing endpoint."""

    def test_missing_image_returns_400(self, client):
        resp = client.post("/api/process", json={})
        assert resp.status_code == 400

    def test_empty_body_returns_400(self, client):
        resp = client.post("/api/process", content_type="application/json", data="{}")
        assert resp.status_code == 400

    def test_invalid_base64_returns_400(self, client):
        resp = client.post("/api/process", json={"image": "not-valid-base64!!!"})
        assert resp.status_code == 400

    def test_valid_image_returns_200(self, client, test_image_b64):
        resp = client.post("/api/process", json={
            "image": test_image_b64,
            "language": "en",
            "detectors": {
                "objects": True,
                "faces": True,
                "emotions": True,
                "signs": True,
            },
        })
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client, test_image_b64):
        data = resp_json(client.post("/api/process", json={
            "image": test_image_b64,
        }))
        for field in ("objects", "faces", "emotions", "signs", "message", "translations", "timestamp"):
            assert field in data, f"Missing field: {field}"

    def test_translations_has_all_languages(self, client, test_image_b64):
        data = resp_json(client.post("/api/process", json={
            "image": test_image_b64,
        }))
        for lang in ("en", "ta", "hi"):
            assert lang in data["translations"]

    def test_detectors_can_be_disabled(self, client, test_image_b64):
        data = resp_json(client.post("/api/process", json={
            "image": test_image_b64,
            "detectors": {
                "objects": False,
                "faces": False,
                "emotions": False,
                "signs": False,
            },
        }))
        assert data["objects"] == []
        assert data["faces"] == []
        assert data["emotions"] == []
        assert data["signs"] == []

    def test_data_url_prefix_handled(self, client, test_image_b64):
        """Ensure data:image/jpeg;base64,... prefix is stripped correctly."""
        resp = client.post("/api/process", json={
            "image": f"data:image/jpeg;base64,{test_image_b64}",
        })
        assert resp.status_code == 200


# ════════════════════════════════════════════════
# POST /api/learn
# ════════════════════════════════════════════════

class TestLearn:
    """Learning endpoint."""

    def test_missing_body_returns_400(self, client):
        resp = client.post("/api/learn", content_type="application/json", data="{}")
        assert resp.status_code == 400

    def test_missing_label_returns_400(self, client):
        resp = client.post("/api/learn", json={"type": "object"})
        assert resp.status_code == 400

    def test_invalid_type_returns_400(self, client):
        resp = client.post("/api/learn", json={
            "type": "invalid",
            "label": "something",
        })
        assert resp.status_code == 400

    def test_learn_object_succeeds(self, client):
        resp = client.post("/api/learn", json={
            "type": "object",
            "id": "obj_test123",
            "label": "water bottle",
        })
        data = resp_json(resp)
        assert resp.status_code == 200
        assert data["success"] is True
        assert "message" in data

    def test_learn_face_with_bad_id_fails(self, client):
        resp = client.post("/api/learn", json={
            "type": "face",
            "id": "face_nonexistent",
            "label": "Test Person",
        })
        assert resp.status_code == 404

    def test_learned_summary_returned(self, client):
        resp = client.post("/api/learn", json={
            "type": "object",
            "id": None,
            "label": "test_obj",
        })
        data = resp_json(resp)
        assert "learned_summary" in data


# ════════════════════════════════════════════════
# GET /api/history
# ════════════════════════════════════════════════

class TestHistory:
    """History endpoint."""

    def test_returns_200(self, client):
        resp = client.get("/api/history")
        assert resp.status_code == 200

    def test_has_counts(self, client):
        data = resp_json(client.get("/api/history"))
        assert "objects_count" in data
        assert "faces_count" in data
        assert "objects" in data
        assert "faces" in data


# ════════════════════════════════════════════════
# GET /api/languages
# ════════════════════════════════════════════════

class TestLanguages:
    """Languages endpoint."""

    def test_returns_200(self, client):
        resp = client.get("/api/languages")
        assert resp.status_code == 200

    def test_has_all_languages(self, client):
        data = resp_json(client.get("/api/languages"))
        for lang in ("en", "ta", "hi"):
            assert lang in data


# ════════════════════════════════════════════════
# 404 Handler
# ════════════════════════════════════════════════

class TestErrorHandlers:
    """Custom error handlers."""

    def test_unknown_route_returns_404_json(self, client):
        resp = client.get("/api/nonexistent")
        assert resp.status_code == 404
        data = resp_json(resp)
        assert "error" in data


# ════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════

def resp_json(resp) -> dict:
    """Extract JSON from a test response."""
    return json.loads(resp.data)

import io
import pytest
from fastapi.testclient import TestClient
from backend.server import app, rate_limits

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Reset rate limiting dictionary between tests."""
    rate_limits.clear()

def test_health_check():
    """Test that /health returns 200 OK and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_inpaint_success_with_preview():
    """Test successful image inpainting request with image, mask, and preview."""
    dummy_image = b"\xFF\xD8\xFF\xE0\x00\x10JFIF" # JPEG header stub
    dummy_mask = b"\xFF\xD8\xFF\xE0\x00\x10JFIF"
    dummy_preview = b"\xFF\xD8\xFF\xE0\x00\x10JFIF_PREVIEW"

    files = {
        "image": ("test.jpg", io.BytesIO(dummy_image), "image/jpeg"),
        "mask": ("mask.jpg", io.BytesIO(dummy_mask), "image/jpeg"),
        "preview": ("preview.jpg", io.BytesIO(dummy_preview), "image/jpeg"),
    }
    data = {
        "device_uuid": "test-device-123",
        "play_integrity_token": "mock-valid-token"
    }

    response = client.post("/v1/inpaint", data=data, files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == dummy_preview

def test_inpaint_success_without_preview():
    """Test inpainting request returning original image when preview is omitted."""
    dummy_image = b"\xFF\xD8\xFF\xE0\x00\x10JFIF_ORIGINAL"
    dummy_mask = b"\xFF\xD8\xFF\xE0\x00\x10JFIF_MASK"

    files = {
        "image": ("test.jpg", io.BytesIO(dummy_image), "image/jpeg"),
        "mask": ("mask.jpg", io.BytesIO(dummy_mask), "image/jpeg"),
    }
    data = {
        "device_uuid": "test-device-456",
        "play_integrity_token": "mock-valid-token"
    }

    response = client.post("/v1/inpaint", data=data, files=files)
    assert response.status_code == 200
    assert response.content == dummy_image

def test_inpaint_rate_limit_exceeded():
    """Test that requests exceeding MAX_DAILY_REQUESTS (10) return HTTP 429."""
    dummy_image = b"\xFF\xD8\xFF\xE0\x00\x10JFIF"
    dummy_mask = b"\xFF\xD8\xFF\xE0\x00\x10JFIF"

    data = {
        "device_uuid": "rate-limited-uuid",
        "play_integrity_token": "mock-valid-token"
    }

    # Make 10 successful requests
    for i in range(10):
        files = {
            "image": ("test.jpg", io.BytesIO(dummy_image), "image/jpeg"),
            "mask": ("mask.jpg", io.BytesIO(dummy_mask), "image/jpeg"),
        }
        res = client.post("/v1/inpaint", data=data, files=files)
        assert res.status_code == 200

    # 11th request must fail with 429
    files = {
        "image": ("test.jpg", io.BytesIO(dummy_image), "image/jpeg"),
        "mask": ("mask.jpg", io.BytesIO(dummy_mask), "image/jpeg"),
    }
    res_exceeded = client.post("/v1/inpaint", data=data, files=files)
    assert res_exceeded.status_code == 429
    assert "Daily request limit exceeded" in res_exceeded.json()["detail"]

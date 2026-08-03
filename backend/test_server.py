import io
from collections.abc import Generator

import icontract
import pytest
from fastapi.testclient import TestClient

from backend.server import (
    app,
    process_inpainting_payload,
    rate_limits,
    verify_token,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_rate_limits() -> Generator[None, None, None]:
    """Reset rate limiting dictionary between tests."""
    rate_limits.clear()
    yield
    rate_limits.clear()


def test_health_check() -> None:
    """Test that /health returns 200 OK and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_inpaint_success_with_preview() -> None:
    """Test successful image inpainting request with image, mask, and preview."""
    dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # JPEG header stub
    dummy_mask = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    dummy_preview = b"\xff\xd8\xff\xe0\x00\x10JFIF_PREVIEW"

    files = {
        "image": ("test.jpg", io.BytesIO(dummy_image), "image/jpeg"),
        "mask": ("mask.jpg", io.BytesIO(dummy_mask), "image/jpeg"),
        "preview": ("preview.jpg", io.BytesIO(dummy_preview), "image/jpeg"),
    }
    data = {
        "device_uuid": "test-device-123",
        "play_integrity_token": "mock-valid-token",
    }

    response = client.post("/v1/inpaint", data=data, files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == dummy_preview


def test_inpaint_success_without_preview() -> None:
    """Test inpainting request returning original image when preview is omitted."""
    dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF_ORIGINAL"
    dummy_mask = b"\xff\xd8\xff\xe0\x00\x10JFIF_MASK"

    files = {
        "image": ("test.jpg", io.BytesIO(dummy_image), "image/jpeg"),
        "mask": ("mask.jpg", io.BytesIO(dummy_mask), "image/jpeg"),
    }
    data = {
        "device_uuid": "test-device-456",
        "play_integrity_token": "mock-valid-token",
    }

    response = client.post("/v1/inpaint", data=data, files=files)
    assert response.status_code == 200
    assert response.content == dummy_image


def test_inpaint_contract_violation_empty_image() -> None:
    """Test that submitting empty image payload triggers contract violation."""
    dummy_mask = b"\xff\xd8\xff\xe0\x00\x10JFIF_MASK"

    files = {
        "image": ("test.jpg", io.BytesIO(b""), "image/jpeg"),
        "mask": ("mask.jpg", io.BytesIO(dummy_mask), "image/jpeg"),
    }
    data = {
        "device_uuid": "test-device-contract",
        "play_integrity_token": "mock-valid-token",
    }

    response = client.post("/v1/inpaint", data=data, files=files)
    assert response.status_code == 400
    assert "Image payload must be non-empty" in response.json()["detail"]


@pytest.mark.parametrize(
    ("device_uuid", "token"),
    [
        ("", "mock-valid-token"),
        ("   ", "mock-valid-token"),
        ("valid-uuid", ""),
        ("valid-uuid", "   "),
    ],
)
def test_verify_token_contract_violations(device_uuid: str, token: str) -> None:
    """Test that empty device UUID or token violates verify_token preconditions."""
    with pytest.raises(icontract.ViolationError):
        verify_token(device_uuid, token)


def test_process_payload_contract_violation() -> None:
    """Direct contract testing of process_inpainting_payload function."""
    with pytest.raises(icontract.ViolationError):
        process_inpainting_payload(b"", b"\x00")

    with pytest.raises(icontract.ViolationError):
        process_inpainting_payload(b"\x00", b"")


def test_inpaint_rate_limit_exceeded() -> None:
    """Test that requests exceeding MAX_DAILY_REQUESTS (10) return HTTP 429."""
    dummy_image = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    dummy_mask = b"\xff\xd8\xff\xe0\x00\x10JFIF"

    data = {
        "device_uuid": "rate-limited-uuid",
        "play_integrity_token": "mock-valid-token",
    }

    for _ in range(10):
        files = {
            "image": ("test.jpg", io.BytesIO(dummy_image), "image/jpeg"),
            "mask": ("mask.jpg", io.BytesIO(dummy_mask), "image/jpeg"),
        }
        res = client.post("/v1/inpaint", data=data, files=files)
        assert res.status_code == 200

    files = {
        "image": ("test.jpg", io.BytesIO(dummy_image), "image/jpeg"),
        "mask": ("mask.jpg", io.BytesIO(dummy_mask), "image/jpeg"),
    }
    res_exceeded = client.post("/v1/inpaint", data=data, files=files)
    assert res_exceeded.status_code == 429
    assert "Daily request limit exceeded" in res_exceeded.json()["detail"]

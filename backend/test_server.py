import io
from collections.abc import Generator
from pathlib import Path

import icontract
import pytest
from fastapi.testclient import TestClient

from backend.config import BackendSettings, settings
from backend.server import (
    app,
    process_inpainting_payload,
    rate_limits,
    verify_token,
)

client = TestClient(app)

# A valid 1x1 black RGB JPEG image
VALID_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffdb0043010909090c0b0c180d0d1832211c213232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232ffc00011080001000103012200021101031101ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f0156272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a636465666768696a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00f9fe8a28a00fffd9"
)


@pytest.fixture(autouse=True)
def reset_rate_limits() -> Generator[None, None, None]:
    """Reset rate limiting dictionary between tests and configure mock tokens."""
    rate_limits.clear()
    orig_tokens = list(settings.allowed_integrity_tokens)
    settings.allowed_integrity_tokens = ["mock-valid-token"]
    yield
    rate_limits.clear()
    settings.allowed_integrity_tokens = orig_tokens


def test_health_check() -> None:
    """Test that /health returns 200 OK and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_web_workbench_endpoint() -> None:
    """Test that GET / returns the EiPy interactive web UI html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>AutoKorrektur SDXL Workbench</title>" in response.text


def test_backend_settings() -> None:
    """Test pydantic-settings defaults and configuration."""
    cfg = BackendSettings()
    assert cfg.max_daily_requests == 10
    assert cfg.allowed_integrity_tokens == []
    assert settings.max_daily_requests == 10


def test_inpaint_success_with_preview() -> None:
    """Test successful image inpainting request with image, mask, and preview."""
    dummy_image = VALID_JPEG
    dummy_mask = VALID_JPEG
    dummy_preview = VALID_JPEG

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
    # C18: Re-encoding means binary equality is not guaranteed.
    # Check if it's a valid JPEG.
    assert response.content.startswith(b"\xff\xd8\xff")


def test_inpaint_success_without_preview() -> None:
    """Test inpainting request returning original image when preview is omitted."""
    dummy_image = VALID_JPEG
    dummy_mask = VALID_JPEG

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
    assert response.content.startswith(b"\xff\xd8\xff")


def test_inpaint_contract_violation_empty_image() -> None:
    """Test that submitting empty image payload triggers contract violation."""
    dummy_mask = VALID_JPEG

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
    dummy_image = VALID_JPEG
    dummy_mask = VALID_JPEG

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


@pytest.mark.parametrize("triple_index", range(1, 51))
def test_fifty_image_triples_inpaint_suite(triple_index: int) -> None:
    """Test all 50 image-triples (car, mask, carless) against the backend inpainting service."""
    fixtures_dir = Path(__file__).parent / "tests" / "fixtures" / "triples"
    prefix = f"triple_{triple_index:02d}"

    car_path = fixtures_dir / f"{prefix}_with_car.png"
    mask_path = fixtures_dir / f"{prefix}_mask.png"
    carless_path = fixtures_dir / f"{prefix}_without_car.png"

    assert car_path.exists(), f"Car image missing for triple {triple_index}"
    assert mask_path.exists(), f"Mask image missing for triple {triple_index}"
    assert carless_path.exists(), f"Carless image missing for triple {triple_index}"

    with (
        open(car_path, "rb") as car_f,
        open(mask_path, "rb") as mask_f,
        open(carless_path, "rb") as carless_f,
    ):
        files = {
            "image": (car_path.name, io.BytesIO(car_f.read()), "image/png"),
            "mask": (mask_path.name, io.BytesIO(mask_f.read()), "image/png"),
            "preview": (carless_path.name, io.BytesIO(carless_f.read()), "image/png"),
        }
        data = {
            "device_uuid": f"device-triple-{triple_index}",
            "play_integrity_token": "mock-valid-token",
        }

        response = client.post("/v1/inpaint", data=data, files=files)
        assert response.status_code == 200, f"Triple {triple_index} failed inpaint request"
        assert response.headers["content-type"] in ("image/jpeg", "image/png")


def test_invalid_integrity_token_rejected() -> None:
    """Test that requests with unapproved Play Integrity tokens return 403."""
    dummy_image = VALID_JPEG
    dummy_mask = VALID_JPEG
    files = {
        "image": ("test.jpg", io.BytesIO(dummy_image), "image/jpeg"),
        "mask": ("mask.jpg", io.BytesIO(dummy_mask), "image/jpeg"),
    }
    data = {
        "device_uuid": "test-device-unauthorized",
        "play_integrity_token": "unknown-or-forged-token",
    }
    response = client.post("/v1/inpaint", data=data, files=files)
    assert response.status_code == 403
    assert "Invalid Google Play Integrity attestation token" in response.json()["detail"]


def test_oversized_upload_rejected() -> None:
    """Test that uploads exceeding max_upload_bytes return 413."""
    orig_max = settings.max_upload_bytes
    try:
        settings.max_upload_bytes = 100  # 100 bytes limit
        dummy_image = VALID_JPEG  # > 100 bytes
        files = {
            "image": ("test.jpg", io.BytesIO(dummy_image), "image/jpeg"),
            "mask": ("mask.jpg", io.BytesIO(dummy_image), "image/jpeg"),
        }
        data = {
            "device_uuid": "test-device-oversized",
            "play_integrity_token": "mock-valid-token",
        }
        response = client.post("/v1/inpaint", data=data, files=files)
        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"].lower()
    finally:
        settings.max_upload_bytes = orig_max


def test_invalid_magic_bytes_rejected() -> None:
    """Test that uploaded files with non-image magic bytes return 400."""
    fake_payload = b"NOT_A_VALID_IMAGE_HEADER_DATA_STREAM"
    files = {
        "image": ("test.bin", io.BytesIO(fake_payload), "application/octet-stream"),
        "mask": ("mask.bin", io.BytesIO(fake_payload), "application/octet-stream"),
    }
    data = {
        "device_uuid": "test-device-magic",
        "play_integrity_token": "mock-valid-token",
    }
    response = client.post("/v1/inpaint", data=data, files=files)
    assert response.status_code == 400



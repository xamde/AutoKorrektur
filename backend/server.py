import asyncio
import io
import logging
from collections import defaultdict
from datetime import date
from typing import Any

import icontract
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autokorrektur-backend")

# Log configuration on startup
logger.info("Effective configuration:")
for key, value in settings.model_dump().items():
    if "password" in key.lower() or "token" in key.lower():
        logger.info(f"  {key}: [REDACTED]")
    else:
        logger.info(f"  {key}: {value}")

app = FastAPI(
    title="AutoKorrektur SDXL Inpainting API",
    description="Photorealistic SDXL cloud inpainting backend service with memory-only GDPR processing.",
    version="1.0.0",
)

# Redis / Rate limiting setup
redis_client: Any | None = None

if settings.redis_url:
    try:
        import redis.asyncio as redis

        redis_client = redis.from_url(settings.redis_url)
        logger.info(f"Connected to Redis rate-limiting store at {settings.redis_url}")
    except Exception as e:
        logger.warning(
            f"Failed to connect to Redis at {settings.redis_url}, falling back to in-memory store: {e}"
        )

rate_limits: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))

# SDXL Pipeline loader (Optional GPU / PyTorch execution)
sdxl_pipeline: Any | None = None
# B11: Guard global SDXL pipeline with a semaphore to prevent CUDA OOM/crashes.
sdxl_semaphore: asyncio.Semaphore = asyncio.Semaphore(1)

try:
    import torch
    from diffusers import AutoPipelineForInpainting

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch detected with device: {device}")
    if settings.enable_sdxl_load:
        sdxl_pipeline = AutoPipelineForInpainting.from_pretrained(
            settings.sdxl_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        logger.info(f"Loaded inpainting model: {settings.sdxl_model_id} on {device}")
except Exception as e:
    logger.info(f"SDXL pipeline initialized in mock/test mode: {e}")


# --- Pydantic Data Models (EiPy Data-Handling Standards) ---


class HealthCheckResponse(BaseModel):
    """Pydantic schema for health check endpoint response."""

    status: str = Field(..., description="Service health status", examples=["ok"])
    redis_connected: bool = Field(..., description="Whether Redis connection is active")
    sdxl_loaded: bool = Field(..., description="Whether SDXL PyTorch pipeline is active")


# --- Design by Contract (DbC) Domain Functions ---


@icontract.require(lambda device_uuid: len(device_uuid.strip()) > 0, "device_uuid must be non-empty")
@icontract.require(
    lambda play_integrity_token: len(play_integrity_token.strip()) > 0,
    "play_integrity_token must be non-empty",
)
def verify_token(device_uuid: str, play_integrity_token: str) -> bool:
    """Verifies Play Integrity token to prevent third-party API abuse."""
    # 1. Allow bypass for development/testing
    if play_integrity_token in settings.allowed_integrity_tokens:
        return True

    # 2. Genuine attestation check using Google Play Integrity API
    if settings.google_application_credentials:
        try:
            from google.cloud import play_integrity_v1

            client = play_integrity_v1.PlayIntegrityServiceClient.from_service_account_json(
                settings.google_application_credentials
            )
            request = play_integrity_v1.DecodeIntegrityTokenRequest(
                integrity_token=play_integrity_token
            )
            response = client.decode_integrity_token(request=request)

            # Check package name and device integrity
            token_payload = response.token_payload_external
            if token_payload.app_integrity.package_name != settings.android_package_name:
                 logger.warning(f"Integrity token package name mismatch: {token_payload.app_integrity.package_name}")
                 if settings.strict_integrity_check:
                     raise HTTPException(status_code=403, detail="App integrity check failed: package mismatch")

            # Check device integrity levels (MEETS_DEVICE_INTEGRITY or better)
            integrity_levels = token_payload.device_integrity.device_recognition_verdict
            if "MEETS_DEVICE_INTEGRITY" not in integrity_levels:
                logger.warning(f"Integrity token device recognition failed for {device_uuid}: {integrity_levels}")
                if settings.strict_integrity_check:
                    raise HTTPException(status_code=403, detail="Device integrity check failed")

            logger.info(f"Play Integrity token verified for device {device_uuid}")
            return True
        except Exception as e:
            logger.error(f"Error during Play Integrity verification: {e}")
            if settings.strict_integrity_check:
                raise HTTPException(status_code=403, detail=f"Play Integrity verification error: {str(e)}")

    logger.warning(
        f"Invalid Play Integrity token rejected for device {device_uuid}: {play_integrity_token[:10]}..."
    )
    if settings.strict_integrity_check:
        raise HTTPException(status_code=403, detail="Invalid Google Play Integrity attestation token")

    # If not strict and no credentials, we still log but allow (dev mode)
    return True


@icontract.require(lambda image_bytes: len(image_bytes) > 0, "Image payload must be non-empty")
@icontract.require(lambda mask_bytes: len(mask_bytes) > 0, "Mask payload must be non-empty")
@icontract.ensure(lambda result: len(result) > 0, "Result image payload must be non-empty")
def process_inpainting_payload(
    image_bytes: bytes, mask_bytes: bytes, preview_bytes: bytes | None = None
) -> bytes:
    """Core image inpainting processor contract.

    Synchronous CPU-bound PIL / PyTorch transformation function.
    """
    from PIL import Image

    # C18: Magic byte sniffing and dimension validation
    def validate_and_open(data: bytes, name: str) -> Image.Image:
        # Sniff magic bytes for JPEG/PNG
        if not (data.startswith(b"\xff\xd8\xff") or data.startswith(b"\x89PNG\r\n\x1a\n")):
             raise HTTPException(status_code=400, detail=f"Invalid image format for {name}. Only JPEG and PNG are allowed.")

        try:
            img = Image.open(io.BytesIO(data))
            img.verify() # verify it's not truncated
            img = Image.open(io.BytesIO(data)) # reopen as verify() closes the file

            if img.width > 2048 or img.height > 2048:
                raise HTTPException(status_code=400, detail=f"{name} dimensions exceed 2048x2048 limit")
            return img
        except Exception as e:
            if isinstance(e, HTTPException): raise
            raise HTTPException(status_code=400, detail=f"Failed to process {name}: {str(e)}")

    init_img = validate_and_open(image_bytes, "image").convert("RGB")
    mask_img = validate_and_open(mask_bytes, "mask").convert("L")

    if sdxl_pipeline is not None:
        prompt = "seamless background, clean street, photorealistic"
        result = sdxl_pipeline(prompt=prompt, image=init_img, mask_image=mask_img).images[0]
    else:
        # Mock/dev environment mode: use preview if available, else original
        if preview_bytes:
            result = validate_and_open(preview_bytes, "preview").convert("RGB")
        else:
            result = init_img

    # C18: Re-encode image before returning to ensure clean metadata and format
    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=95, optimize=True)
    return buf.getvalue()


async def check_rate_limit(device_uuid: str, client_ip: str) -> None:
    """Enforce max daily requests limit per device/IP per day."""
    today_str = date.today().isoformat()
    # B12: Key rate limits on both device UUID and IP to prevent spoofing/bypass
    key = f"rate:{device_uuid}:{client_ip}:{today_str}"

    if redis_client:
        try:
            # B12: Use atomic Redis operations. INCR is atomic.
            count = await redis_client.incr(key)
            if count == 1:
                # Set expiration for the first request of the day
                await redis_client.expire(key, 86400)

            if count > settings.max_daily_requests:
                logger.warning(f"Rate limit exceeded for {device_uuid} at {client_ip}")
                raise HTTPException(
                    status_code=429,
                    detail="Daily request limit exceeded. Please try again tomorrow.",
                )
            return
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Redis rate limit check error: {e}, falling back to in-memory")

    # B12: In-memory fallback (only useful for single-worker dev setups)
    today = date.today()
    mem_key = f"{device_uuid}:{client_ip}"
    if rate_limits[mem_key][today] >= settings.max_daily_requests:
        logger.warning(f"In-memory rate limit exceeded for {mem_key}")
        raise HTTPException(
            status_code=429,
            detail="Daily request limit exceeded. Please try again tomorrow.",
        )
    rate_limits[mem_key][today] += 1


# --- API Routes & Interactive Web UI ---


@app.get("/", response_class=HTMLResponse)
def get_web_workbench() -> str:
    """EiPy User-Interfaces: Interactive in-browser drag-and-drop web workbench."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutoKorrektur SDXL Workbench</title>
        <style>
            :root { --bg: #0f172a; --card: #1e293b; --accent: #38bdf8; --text: #f8fafc; }
            body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; }
            .container { max-width: 900px; margin: 0 auto; background: var(--card); border-radius: 12px; padding: 2rem; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            h1 { margin-top: 0; color: var(--accent); }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem; }
            .drop-zone { border: 2px dashed #475569; border-radius: 8px; padding: 1.5rem; text-align: center; cursor: pointer; transition: 0.2s; }
            .drop-zone:hover { border-color: var(--accent); background: rgba(56, 189, 248, 0.05); }
            input[type="file"] { display: none; }
            button { width: 100%; padding: 0.8rem; background: var(--accent); color: #000; border: none; border-radius: 6px; font-weight: bold; font-size: 1rem; cursor: pointer; margin-top: 1.5rem; }
            button:hover { opacity: 0.9; }
            .preview-container { margin-top: 1.5rem; text-align: center; }
            img { max-width: 100%; border-radius: 8px; border: 1px solid #475569; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AutoKorrektur Inpainting Workbench</h1>
            <p>Test the SDXL Cloud Inpainting API directly from your browser.</p>
            <form id="inpaintForm">
                <div class="grid">
                    <div class="drop-zone" onclick="document.getElementById('imageFile').click()">
                        <strong>📷 Source Image</strong>
                        <p id="imageName">Click or drop JPEG</p>
                        <input type="file" id="imageFile" accept="image/*" required onchange="updateName('imageFile', 'imageName')">
                    </div>
                    <div class="drop-zone" onclick="document.getElementById('maskFile').click()">
                        <strong>🎭 Vehicle Mask</strong>
                        <p id="maskName">Click or drop Mask JPEG</p>
                        <input type="file" id="maskFile" accept="image/*" required onchange="updateName('maskFile', 'maskName')">
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <label for="integrityToken" style="display: block; margin-bottom: 0.5rem;">🔑 Play Integrity Token</label>
                    <input type="text" id="integrityToken" placeholder="Paste your token here" style="width: 100%; padding: 0.8rem; background: #334155; border: 1px solid #475569; border-radius: 6px; color: white;">
                </div>
                <button type="submit" id="submitBtn">Run SDXL Inpainting</button>
            </form>
            <div class="preview-container" id="resultContainer" style="display:none;">
                <h3>Inpainted Output Result</h3>
                <img id="resultImg" src="" alt="Result">
            </div>
        </div>
        <script>
            function updateName(inputId, textId) {
                const input = document.getElementById(inputId);
                if (input.files.length > 0) {
                    document.getElementById(textId).innerText = input.files[0].name;
                }
            }
            document.getElementById('inpaintForm').onsubmit = async (e) => {
                e.preventDefault();
                const btn = document.getElementById('submitBtn');
                btn.disabled = true;
                btn.innerText = 'Processing Inpainting...';

                const formData = new FormData();
                formData.append('device_uuid', 'web-workbench-demo');
                formData.append('play_integrity_token', document.getElementById('integrityToken').value);
                formData.append('image', document.getElementById('imageFile').files[0]);
                formData.append('mask', document.getElementById('maskFile').files[0]);

                try {
                    const res = await fetch('/v1/inpaint', { method: 'POST', body: formData });
                    if (!res.ok) throw new Error(await res.text());
                    const blob = await res.blob();
                    document.getElementById('resultImg').src = URL.createObjectURL(blob);
                    document.getElementById('resultContainer').style.display = 'block';
                } catch (err) {
                    alert('Error: ' + err.message);
                } finally {
                    btn.disabled = false;
                    btn.innerText = 'Run SDXL Inpainting';
                }
            };
        </script>
    </body>
    </html>
    """


@app.post("/v1/inpaint")
async def inpaint_image(
    request: Request,
    device_uuid: str = Form(...),
    play_integrity_token: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    preview: UploadFile | None = File(None),
) -> StreamingResponse:
    """Receives an image, a mask, and an optional preview, and performs SDXL inpainting.

    To ensure GDPR compliance, all image data is processed in memory and never written to disk.
    Uses asyncio.to_thread to execute CPU-bound PIL/PyTorch code without blocking the event loop.
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Received request for /v1/inpaint from {device_uuid} at {client_ip}")

    verify_token(device_uuid, play_integrity_token)
    await check_rate_limit(device_uuid, client_ip)

    # A7: Upload size limit check
    total_size = (image.size or 0) + (mask.size or 0) + (preview.size if preview else 0)
    if total_size > settings.max_upload_bytes:
         raise HTTPException(status_code=413, detail=f"Total upload size exceeds {settings.max_upload_bytes} bytes")

    try:
        image_data = await image.read()
        mask_data = await mask.read()

        preview_data = None
        if preview:
            preview_data = await preview.read()

        logger.info(f"Loaded image ({len(image_data)} bytes) and mask ({len(mask_data)} bytes) into memory")

        if sdxl_pipeline is None:
            await asyncio.sleep(0.1)

        # B11: Use the semaphore to limit concurrent SDXL inferences
        async with sdxl_semaphore:
            # EiPy Asyncio: Offload CPU-bound image manipulation to threadpool to avoid event loop blocking
            output_data = await asyncio.to_thread(process_inpainting_payload, image_data, mask_data, preview_data)

        logger.info("Returning processed image from memory")
        return StreamingResponse(io.BytesIO(output_data), media_type="image/jpeg")

    except icontract.ViolationError as e:
        logger.error(f"Contract violation: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during inpainting") from e


@app.get("/health")
def health_check() -> HealthCheckResponse:
    """Health check endpoint returning structured Pydantic schema."""
    return HealthCheckResponse(
        status="ok",
        redis_connected=redis_client is not None,
        sdxl_loaded=sdxl_pipeline is not None,
    )

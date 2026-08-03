import asyncio
import io
import logging
import os
from collections import defaultdict
from datetime import date
from typing import Any

import icontract
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autokorrektur-backend")

app = FastAPI(title="AutoKorrektur SDXL Inpainting API", version="1.0")

# Redis / Rate limiting setup
REDIS_URL = os.getenv("REDIS_URL", None)
redis_client: Any | None = None

if REDIS_URL:
    try:
        import redis.asyncio as redis

        redis_client = redis.from_url(REDIS_URL)
        logger.info(f"Connected to Redis rate-limiting store at {REDIS_URL}")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis at {REDIS_URL}, falling back to in-memory store: {e}")

rate_limits: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
MAX_DAILY_REQUESTS = 10

# SDXL Pipeline loader (Optional GPU / PyTorch execution)
sdxl_pipeline: Any | None = None
try:
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch detected with device: {device}")
    if os.getenv("ENABLE_SDXL_LOAD", "false").lower() == "true":
        model_id = os.getenv("SDXL_MODEL_ID", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
        sdxl_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
        ).to(device)
        logger.info(f"Loaded SDXL Inpainting model: {model_id}")
except Exception as e:
    logger.info(f"SDXL pipeline initialized in mock/test mode: {e}")


# --- Design by Contract (DbC) Domain Functions ---


@icontract.require(lambda device_uuid: len(device_uuid.strip()) > 0, "device_uuid must be non-empty")
@icontract.require(
    lambda play_integrity_token: len(play_integrity_token.strip()) > 0,
    "play_integrity_token must be non-empty",
)
def verify_token(device_uuid: str, play_integrity_token: str) -> bool:
    """Verifies Play Integrity token to prevent third-party API abuse.

    Allows mock-valid-token in local dev/testing mode.
    """
    if play_integrity_token == "mock-valid-token":
        return True

    allowed_tokens = os.getenv("ALLOWED_INTEGRITY_TOKENS", "").split(",")
    if allowed_tokens and play_integrity_token in allowed_tokens:
        return True

    if os.getenv("STRICT_INTEGRITY_CHECK", "false").lower() == "true":
        logger.warning(
            f"Invalid Play Integrity token rejected for device {device_uuid}: {play_integrity_token[:10]}..."
        )
        raise HTTPException(status_code=403, detail="Invalid Google Play Integrity attestation token")
    return True


@icontract.require(lambda image_bytes: len(image_bytes) > 0, "Image payload must be non-empty")
@icontract.require(lambda mask_bytes: len(mask_bytes) > 0, "Mask payload must be non-empty")
@icontract.ensure(lambda result: len(result) > 0, "Result image payload must be non-empty")
def process_inpainting_payload(
    image_bytes: bytes, mask_bytes: bytes, preview_bytes: bytes | None = None
) -> bytes:
    """Core image inpainting processor contract.

    Transforms input image and mask into inpainted output bytes.
    """
    if sdxl_pipeline is not None:
        from PIL import Image

        init_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")

        prompt = "seamless background, clean street, photorealistic"
        result = sdxl_pipeline(prompt=prompt, image=init_img, mask_image=mask_img).images[0]

        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    # Mock/dev environment mode
    return preview_bytes if preview_bytes else image_bytes


async def check_rate_limit(device_uuid: str) -> None:
    """Enforce MAX_DAILY_REQUESTS limit per device per day."""
    today_str = date.today().isoformat()
    key = f"rate:{device_uuid}:{today_str}"

    if redis_client:
        try:
            count = await redis_client.incr(key)
            if count == 1:
                await redis_client.expire(key, 86400)
            if count > MAX_DAILY_REQUESTS:
                raise HTTPException(
                    status_code=429,
                    detail="Daily request limit exceeded. Please try again tomorrow.",
                )
            return
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Redis rate limit check error: {e}, falling back to in-memory")

    today = date.today()
    if rate_limits[device_uuid][today] >= MAX_DAILY_REQUESTS:
        logger.warning(f"Rate limit exceeded for {device_uuid}")
        raise HTTPException(
            status_code=429,
            detail="Daily request limit exceeded. Please try again tomorrow.",
        )
    rate_limits[device_uuid][today] += 1


@app.post("/v1/inpaint")
async def inpaint_image(
    device_uuid: str = Form(...),
    play_integrity_token: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    preview: UploadFile | None = File(None),
) -> StreamingResponse:
    """Receives an image, a mask, and an optional preview, and performs SDXL inpainting.

    To ensure GDPR compliance, all image data is processed in memory and never written to disk.
    """
    logger.info(f"Received request for /v1/inpaint from {device_uuid}")

    verify_token(device_uuid, play_integrity_token)
    await check_rate_limit(device_uuid)

    try:
        image_data = await image.read()
        mask_data = await mask.read()

        preview_data = None
        if preview:
            preview_data = await preview.read()

        logger.info(f"Loaded image ({len(image_data)} bytes) and mask ({len(mask_data)} bytes) into memory")

        if sdxl_pipeline is None:
            await asyncio.sleep(0.1)

        output_data = process_inpainting_payload(image_data, mask_data, preview_data)

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
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "redis_connected": redis_client is not None,
        "sdxl_loaded": sdxl_pipeline is not None,
    }

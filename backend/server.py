import asyncio
import io
import logging
from collections import defaultdict
from datetime import date
from typing import Any

import icontract
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import secrets
from starlette.datastructures import UploadFile as StarletteUploadFile

StarletteUploadFile.spool_max_size = 15 * 1024 * 1024
from pydantic import BaseModel, Field

from backend.config import settings

from contextlib import asynccontextmanager
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autokorrektur-backend")

# Setup template path
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Domain Exceptions
class InpaintingDomainError(Exception):
    """Base domain exception for inpainting backend."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class InvalidImagePayloadError(InpaintingDomainError):
    """Raised when an uploaded image/mask payload is corrupted or has an invalid format."""
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class ImageDimensionExceededError(InpaintingDomainError):
    """Raised when uploaded image dimensions exceed maximum resolution limits (2048x2048)."""
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class IntegrityVerificationError(InpaintingDomainError):
    """Raised when Google Play Integrity attestation token verification fails or is rejected."""
    def __init__(self, message: str):
        super().__init__(message, status_code=403)


# Global state holders for lifespan
redis_client: Any | None = None
sdxl_pipeline: Any | None = None
sdxl_semaphore: asyncio.Semaphore | None = None
integrity_client: Any | None = None
rate_limits: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))


def get_sdxl_semaphore() -> asyncio.Semaphore:
    """Lazily initializes the inpainting semaphore inside the running event loop."""
    global sdxl_semaphore
    if sdxl_semaphore is None:
        sdxl_semaphore = asyncio.Semaphore(1)
    return sdxl_semaphore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager initializing models, caches, and API clients."""
    global redis_client, sdxl_pipeline, integrity_client

    logger.info("Initializing AutoKorrektur backend lifespan...")
    # 1. Redis setup
    if settings.redis_url:
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url(settings.redis_url)
            logger.info(f"Connected to Redis rate-limiting store at {settings.redis_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")

    # 2. PyTorch / SDXL setup
    try:
        import torch
        from diffusers import AutoPipelineForInpainting

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"PyTorch detected with device: {device}")
        if settings.enable_sdxl_load:
            sdxl_pipeline = AutoPipelineForInpainting.from_pretrained(
                settings.sd_model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            logger.info(f"Loaded inpainting model: {settings.sd_model_id} on {device}")
    except Exception as e:
        logger.info(f"SDXL pipeline initialized in mock/test mode: {e}")

    # 3. Google Play Integrity gRPC client setup
    if settings.google_application_credentials:
        try:
            from google.cloud import play_integrity_v1
            integrity_client = play_integrity_v1.PlayIntegrityServiceClient.from_service_account_json(
                settings.google_application_credentials
            )
            logger.info("Initialized Google Play Integrity gRPC client")
        except Exception as e:
            logger.warning(f"Failed to initialize Play Integrity client: {e}")

    yield

    # Clean shutdown
    if redis_client:
        await redis_client.close()
    logger.info("Shutting down AutoKorrektur backend lifespan.")


app = FastAPI(
    title="AutoKorrektur SDXL Inpainting API",
    description="Photorealistic SDXL cloud inpainting backend service with memory-only GDPR processing.",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_upload_bytes:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Payload size exceeds maximum allowed limit of {settings.max_upload_bytes} bytes"}
        )
    return await call_next(request)

nonce_store = {}

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
def verify_token(device_uuid: str, play_integrity_token: str) -> str | bool:
    """Verifies Play Integrity token to prevent third-party API abuse."""
    # 1. Allow bypass for development/testing
    if play_integrity_token in settings.allowed_integrity_tokens:
        return True

    # 2. Genuine attestation check using Google Play Integrity API
    if settings.google_application_credentials:
        try:
            from google.cloud import play_integrity_v1

            client = integrity_client or play_integrity_v1.PlayIntegrityServiceClient.from_service_account_json(
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
                    raise IntegrityVerificationError("App integrity check failed: package mismatch")

            # Check device integrity levels (MEETS_DEVICE_INTEGRITY or better)
            integrity_levels = token_payload.device_integrity.device_recognition_verdict
            if "MEETS_DEVICE_INTEGRITY" not in integrity_levels:
                logger.warning(f"Integrity token device recognition failed for {device_uuid}: {integrity_levels}")
                if settings.strict_integrity_check:
                    raise IntegrityVerificationError("Device integrity check failed")

            logger.info(f"Play Integrity token verified for device {device_uuid}")
            return token_payload.request_details.nonce
        except InpaintingDomainError:
            raise
        except Exception as e:
            logger.error(f"Error during Play Integrity verification: {e}")
            if settings.strict_integrity_check:
                raise IntegrityVerificationError(f"Play Integrity verification error: {str(e)}") from e

    logger.warning(
        f"Invalid Play Integrity token rejected for device {device_uuid}: {play_integrity_token[:10]}..."
    )
    if settings.strict_integrity_check:
        raise IntegrityVerificationError("Invalid Google Play Integrity attestation token")

    return True


@icontract.require(lambda image_bytes: len(image_bytes) > 0, "Image payload must be non-empty")
@icontract.require(lambda mask_bytes: len(mask_bytes) > 0, "Mask payload must be non-empty")
@icontract.ensure(lambda result: len(result) > 0, "Result image payload must be non-empty")
def process_inpainting_payload(
    image_bytes: bytes, mask_bytes: bytes, preview_bytes: bytes | None = None
) -> bytes:
    """Core image inpainting processor contract.

    Synchronous CPU-bound PIL / PyTorch transformation function.
    Raises domain exceptions rather than web-framework HTTP exceptions.
    """
    from PIL import Image

    def validate_and_open(data: bytes, name: str) -> Image.Image:
        # Sniff magic bytes for JPEG/PNG
        if not (data.startswith(b"\xff\xd8\xff") or data.startswith(b"\x89PNG\r\n\x1a\n")):
            raise InvalidImagePayloadError(f"Invalid image format for {name}. Only JPEG and PNG are allowed.")

        try:
            img = Image.open(io.BytesIO(data))
            img.verify()
            img = Image.open(io.BytesIO(data))

            if img.width > 2048 or img.height > 2048:
                raise ImageDimensionExceededError(f"{name} dimensions exceed 2048x2048 limit")
            return img
        except InpaintingDomainError:
            raise
        except Exception as e:
            raise InvalidImagePayloadError(f"Failed to process {name}: {str(e)}") from e

    init_img = validate_and_open(image_bytes, "image").convert("RGB")
    mask_img = validate_and_open(mask_bytes, "mask").convert("L")

    if sdxl_pipeline is not None:
        prompt = settings.inpainting_prompt
        result = sdxl_pipeline(prompt=prompt, image=init_img, mask_image=mask_img).images[0]
    else:
        # Mock/dev environment mode: use preview if available, else original
        if preview_bytes:
            result = validate_and_open(preview_bytes, "preview").convert("RGB")
        else:
            result = init_img

    # Re-encode image before returning to ensure clean metadata and format
    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=95, optimize=True)
    return buf.getvalue()


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def check_rate_limit(device_uuid: str, client_ip: str) -> None:
    """Enforce max daily requests limit per device/IP per day."""
    today_str = date.today().isoformat()
    # B12: Key rate limits on IP to prevent spoofing/bypass
    key = f"rate:{client_ip}:{today_str}"

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
    mem_key = f"{client_ip}"
    # Prune expired dates to prevent unbounded dictionary growth
    expired_days = [d for d in rate_limits[mem_key] if d < today]
    for d in expired_days:
        del rate_limits[mem_key][d]

    if rate_limits[mem_key][today] >= settings.max_daily_requests:
        logger.warning(f"In-memory rate limit exceeded for {mem_key} (device {device_uuid})")
        raise HTTPException(
            status_code=429,
            detail="Daily request limit exceeded. Please try again tomorrow.",
        )
    rate_limits[mem_key][today] += 1


# --- API Routes & Interactive Web UI ---


@app.get("/", response_class=HTMLResponse)
def get_web_workbench() -> str:
    """Interactive in-browser drag-and-drop web workbench loaded from template."""
    workbench_path = TEMPLATES_DIR / "workbench.html"
    if workbench_path.exists():
        return workbench_path.read_text(encoding="utf-8")
    return "<h1>AutoKorrektur Inpainting Workbench</h1>"


@app.get("/v1/nonce")
async def generate_nonce():
    """Generates a cryptographically random nonce for Play Integrity verification."""
    nonce = secrets.token_urlsafe(32)
    nonce_store[nonce] = True
    return {"nonce": nonce}


@app.post("/v1/inpaint", response_class=StreamingResponse)
async def inpaint_image(
    request: Request,
    image: UploadFile = File(..., description="Original image JPEG/PNG"),
    mask: UploadFile = File(..., description="Mask image matching original dimensions"),
    preview: UploadFile | None = File(None, description="Optional preview image"),
    device_uuid: str = Form(..., description="Client device UUID"),
    play_integrity_token: str = Form(..., description="Google Play Integrity attestation token"),
) -> StreamingResponse:
    """Receives an image, a mask, and an optional preview, and performs SDXL inpainting.

    To ensure GDPR compliance, all image data is processed in memory and never written to disk.
    Uses asyncio.to_thread to execute CPU-bound PIL/PyTorch code without blocking the event loop.
    """
    client_ip = get_client_ip(request)
    logger.info(f"Received request for /v1/inpaint from {device_uuid} at {client_ip}")

    try:
        token_nonce = await asyncio.to_thread(verify_token, device_uuid, play_integrity_token)
        if isinstance(token_nonce, str):
            if not nonce_store.pop(token_nonce, None):
                raise InpaintingDomainError("Invalid or reused Play Integrity nonce", status_code=403)
    except InpaintingDomainError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e

    await check_rate_limit(device_uuid, client_ip)

    # A7: Upload size limit check (pre-check if headers provide size)
    total_size = (image.size or 0) + (mask.size or 0) + ((preview.size or 0) if preview else 0)
    if total_size > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"Total upload size exceeds {settings.max_upload_bytes} bytes")

    try:
        total_read = 0

        async def read_limited(upload: UploadFile) -> bytes:
            nonlocal total_read
            chunks = []
            chunk_size = 64 * 1024
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413, detail=f"Total upload size exceeds {settings.max_upload_bytes} bytes"
                    )
                chunks.append(chunk)
            return b"".join(chunks)

        image_data = await read_limited(image)
        mask_data = await read_limited(mask)
        preview_data = await read_limited(preview) if preview else None

        logger.info(f"Loaded image ({len(image_data)} bytes) and mask ({len(mask_data)} bytes) into memory")

        if sdxl_pipeline is None:
            await asyncio.sleep(0.1)

        # B11: Use the semaphore to limit concurrent SDXL inferences
        async with get_sdxl_semaphore():
            # Offload CPU-bound image manipulation to threadpool
            try:
                output_data = await asyncio.wait_for(
                    asyncio.to_thread(process_inpainting_payload, image_data, mask_data, preview_data),
                    timeout=120.0
                )
            except asyncio.TimeoutError as e:
                raise HTTPException(status_code=504, detail="Inpainting inference timed out") from e

        logger.info("Returning processed image from memory")
        return StreamingResponse(io.BytesIO(output_data), media_type="image/jpeg")

    except InpaintingDomainError as e:
        logger.warning(f"Inpainting domain validation error: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
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

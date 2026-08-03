# AutoKorrektur FastAPI Backend Service

This backend service provides opt-in, photorealistic **SDXL Cloud Inpainting** for the AutoKorrektur Android application.

## Best Practices & Standards (EiPy Compliant)

- **Standard PEP 621 Packaging**: Dependencies and tooling configured in [pyproject.toml](file:///home/konrad/files/work/__drafts/AutoKorrektur/backend/pyproject.toml) managed via `uv`.
- **Design by Contract (DbC)**: Preconditions and postconditions enforced at runtime via `icontract` (`@icontract.require`, `@icontract.ensure`).
- **Code Quality & Type Safety**: Checked via `ruff` and `mypy` (`strict = true`).
- **Concurrency & Reliability**: SDXL inference is guarded by an `asyncio.Semaphore(1)` to prevent
  concurrent GPU execution, avoiding CUDA Out-of-Memory (OOM) errors and server crashes.
- **Enhanced Security & Rate Limiting**: Requests are rate-limited by both Device UUID and Client IP
  using atomic Redis operations (or in-memory fallback).
- **Privacy & GDPR Compliance**: Minimal data footprint processing. All images are processed
  strictly in-memory and re-encoded before returning; zero long-term retention.

## API Specification

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**: `{"status": "ok", "redis_connected": false, "sdxl_loaded": false}`

### 2. SDXL Inpainting
- **Endpoint**: `POST /v1/inpaint`
- **Content-Type**: `multipart/form-data`
- **Form Parameters**:
  - `device_uuid`: String - Unique client installation identifier.
  - `play_integrity_token`: String - Play Integrity verification token.
  - `image`: File (`image/jpeg`) - Source photo.
  - `mask`: File (`image/jpeg`) - Binary vehicle segmentation mask.
  - `preview`: File (`image/jpeg`, optional) - Downscaled preview bitmap.
- **Response**: `200 OK` (`image/jpeg`) streamed response or `429 Too Many Requests`.

## Local Setup & Development with `uv`

1. Sync virtual environment and install development dependencies:
   ```bash
   uv sync --directory backend --extra dev
   ```

2. Run code quality checks (Ruff & Mypy):
   ```bash
   uv run --directory backend ruff check .
   uv run --directory backend mypy .
   ```

3. Run test suite with coverage:
   ```bash
   uv run --directory backend pytest --cov=.
   ```

4. Start the FastAPI development server:
   ```bash
   uv run --directory backend uvicorn server:app --host 127.0.0.1 --port 8000
   ```

## Environment Variables

The application can be configured via environment variables (prefixed with `AUTOKORREKTUR_`):

| Variable                                       | Description                            | Default                                  |
|------------------------------------------------|----------------------------------------|------------------------------------------|
| `AUTOKORREKTUR_REDIS_URL`                      | Redis connection URL for rate limiting | `None`                                   |
| `AUTOKORREKTUR_MAX_DAILY_REQUESTS`             | Max requests per device per day        | `10`                                     |
| `AUTOKORREKTUR_MAX_UPLOAD_BYTES`               | Max total size of a single request     | `10485760` (10MB)                        |
| `AUTOKORREKTUR_ENABLE_SDXL_LOAD`               | Whether to load the real SDXL model    | `False`                                  |
| `AUTOKORREKTUR_STRICT_INTEGRITY_CHECK`         | Reject requests with invalid tokens    | `True`                                   |
| `AUTOKORREKTUR_ALLOWED_INTEGRITY_TOKENS`       | List of valid Play Integrity tokens    | `[]`                                     |
| `AUTOKORREKTUR_GOOGLE_APPLICATION_CREDENTIALS` | Path to Google Service Account JSON    | `None`                                   |
| `AUTOKORREKTUR_ANDROID_PACKAGE_NAME`           | Expected Android app package name      | `de.konradvoelkel.android.autokorrektur` |

## Docker Container Deployment

To build and run the production backend container (from the project root):

```bash
docker build -t autokorrektur-backend -f backend/Dockerfile .
docker-compose -f backend/docker-compose.yml up -d
```

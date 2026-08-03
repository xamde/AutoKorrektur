# AutoKorrektur FastAPI Backend Service

This backend service provides opt-in, photorealistic **SDXL Cloud Inpainting** for the AutoKorrektur Android application.

## Best Practices & Standards (EiPy Compliant)

- **Standard PEP 621 Packaging**: Dependencies and tooling configured in [pyproject.toml](file:///home/konrad/files/work/__drafts/AutoKorrektur/backend/pyproject.toml) managed via `uv`.
- **Design by Contract (DbC)**: Preconditions and postconditions enforced at runtime via `icontract` (`@icontract.require`, `@icontract.ensure`).
- **Code Quality & Type Safety**: Checked via `ruff` and `mypy` (`strict = true`).
- **Privacy & GDPR Compliance**: Zero storage / memory-only processing using `io.BytesIO` streams with zero retention.

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

## Docker Container Deployment

To build and run the production backend container:

```bash
docker build -t autokorrektur-backend -f backend/Dockerfile backend/
docker run -d -p 8000:8000 autokorrektur-backend
```

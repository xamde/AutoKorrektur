# AutoKorrektur FastAPI Backend Service

This backend service provides opt-in, photorealistic **SDXL Cloud Inpainting** for the AutoKorrektur Android application.

## Privacy & GDPR Compliance

- **Zero Storage / Memory-Only Processing**: Received images and masks are processed exclusively in RAM using `io.BytesIO` streams.
- **Zero Retention**: Processed image bytes are streamed directly back to the client (`StreamingResponse`). Once the HTTP connection closes, memory is immediately garbage collected. No disk writes or persistent storage are used.
- **Rate Limiting**: Daily limit of 10 requests per device UUID to prevent API abuse.

## API Specification

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**: `{"status": "ok"}`

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

## Local Setup & Development

1. Create Python virtual environment using `uv`:
   ```bash
   uv venv .venv
   uv pip install --python .venv/bin/python -r backend/requirements.txt
   ```

2. Run the FastAPI development server:
   ```bash
   .venv/bin/uvicorn backend.server:app --host 127.0.0.1 --port 8000
   ```

3. Run automated backend unit tests:
   ```bash
   PYTHONPATH=. .venv/bin/pytest backend/test_server.py
   ```

## Docker Container Deployment

To build and run the production backend container:

```bash
docker build -t autokorrektur-backend -f backend/Dockerfile .
docker run -d -p 8000:8000 autokorrektur-backend
```

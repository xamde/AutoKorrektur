FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if required (OpenCV often needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Run Uvicorn server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

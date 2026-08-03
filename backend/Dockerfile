FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Copy dependency specifications
COPY pyproject.toml README.md ./
COPY server.py __init__.py ./

# Install dependencies using uv sync
RUN uv sync --frozen --no-dev

# Expose port for FastAPI
EXPOSE 8000

# Run Uvicorn server via uv run
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

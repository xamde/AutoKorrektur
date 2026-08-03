FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Copy dependency specifications (from project root context)
COPY backend/pyproject.toml backend/uv.lock* backend/README.md ./backend/
COPY backend/*.py ./backend/

# Move into backend for uv sync if pyproject.toml is there
# Or stay in /app and use --project backend
RUN uv sync --frozen --no-dev --project ./backend

# Expose port for FastAPI
EXPOSE 8000

# Run Uvicorn server via uv run
# Use the backend package structure for imports to work
CMD ["uv", "run", "--project", "./backend", "uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]

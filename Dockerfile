# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker caching)
COPY pyproject.toml .
COPY uv.lock .

# Install uv and dependencies
RUN pip install uv
RUN uv sync --frozen

# Copy entire project
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
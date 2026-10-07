# ppgoapibiometriafacialws/Dockerfile
FROM python:3.12-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (OpenCV/ONNX runtime libs for slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless>=4.8.0

# Copy the rest of the application code
COPY . .

# Create non-root user for security
# UID fixo e USER numerico: o kubelet so valida runAsNonRoot com UID numerico,
# e o runAsUser do all.yaml precisa ser o mesmo UID (HOME = /home/appuser).
RUN useradd --uid 1000 --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER 1000:1000

# Expose port (documentation only - actual mapping in docker-compose)
EXPOSE ${API_INTERNAL_PORT}

# Default command: run gunicorn with Uvicorn ASGI workers
# Workers and port configurable via environment variables
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
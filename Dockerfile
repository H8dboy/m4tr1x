FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV and ExifTool
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create required directories
RUN mkdir -p uploads models

# Copy environment template (override at runtime with --env-file)
COPY .env.example .env

EXPOSE 8080

# Environment defaults (can be overridden at runtime)
ENV PORT=8080 \
    DB_PATH=m4tr1x.db \
    MAX_FILE_SIZE_MB=100 \
    ALLOWED_ORIGINS=https://h8dboy.github.io

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')"

CMD ["python", "api_server.py"]

FROM python:3.11-slim

WORKDIR /app

# Dipendenze sistema per OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libimage-exiftool-perl \
        && rm -rf /var/lib/apt/lists/*

        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt

        COPY . .

        # Crea directory necessarie
        RUN mkdir -p uploads models

        EXPOSE 8080

        # Health check
        HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
            CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')"

            CMD ["python", "api_server.py"]

# M4TR1X — The Unfiltered Eye

> **Decentralized open-source social network with integrated AI verification.**  
> Protect, verify, and share authentic video evidence.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Nostr](https://img.shields.io/badge/Protocol-Nostr-purple.svg)](https://nostr.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](Dockerfile)
[![Live PWA](https://img.shields.io/badge/PWA-Live-brightgreen.svg)](https://h8dboy.github.io/m4tr1x/)

---

## What is M4TR1X?

M4TR1X is a **decentralized social network** built on the [Nostr protocol](https://nostr.com/), designed for the preservation and verification of video evidence. Every uploaded video is analyzed by an AI that determines whether it is authentic or artificially generated.

In a world flooded with deepfakes and manipulated media, M4TR1X gives every user a cryptographic, tamper-proof attestation for the content they share — **boots on the ground, not in a studio**.

---

## Architecture

```
Frontend (index.html)        ->  Mobile-first PWA
        |                        Nostr auth (auth.html)
        v
API Server (api_server.py)   ->  FastAPI + CORS
        |
        v
AI Detector (ai_detector.py) ->  EfficientNet-B0 / ONNX
        |
        v
Mesh Server (mesh_server.py) ->  Local P2P node
        |
        v
Core Utils (core.py)         ->  Metadata scrubbing (ExifTool)
        |
        v
Uploader (uploader.py)       ->  Fernet video encryption
```

---

## Modules

| Module | File | Description |
|---|---|---|
| 🤖 AI Detector | `ai_detector.py` | Frame-by-frame analysis with EfficientNet-B0. Extracts 16 equidistant frames, classifies each as REAL/AI-GENERATED, returns confidence score + consistency score. |
| 🌐 API Server | `api_server.py` | FastAPI REST backend. Video upload, AI analysis, Nostr attestation. CORS enabled for PWA. |
| 🎓 Model Training | `train_detector.py` | Training script with data augmentation, cosine annealing LR, early stopping. Supports custom datasets and ONNX export. |
| 🧹 Metadata Scrubbing | `core.py` | GPS/EXIF removal via ExifTool before publishing. |
| 🔐 Video Encryption | `uploader.py` | Fernet symmetric encryption for file protection. |
| 📡 Mesh Node | `mesh_server.py` | Local node for decentralized P2P communication. |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [ExifTool](https://exiftool.org/) installed on your system
- (Optional) Docker

### Installation

```bash
git clone https://github.com/H8dboy/m4tr1x.git
cd m4tr1x
pip install -r requirements.txt
```

### Start the API Server

```bash
python api_server.py
# Server running at  http://localhost:8080
# Interactive API docs at http://localhost:8080/docs
```

### Analyze a Video from CLI

```bash
python ai_detector.py video.mp4
```

### Train a Custom Model

```bash
# 1. Prepare your dataset
mkdir -p data/train/real data/train/ai_generated
mkdir -p data/val/real data/val/ai_generated
# Place frames/images into the respective folders

# 2. Train the model
python train_detector.py

# 3. Export to ONNX for mobile deployment
python train_detector.py --export-onnx
```

### Docker

```bash
docker build -t m4tr1x .
docker run -p 8080:8080 m4tr1x
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/analyze` | Upload + analyze video |
| `GET` | `/api/v1/analysis/{id}` | Retrieve analysis result |
| `GET` | `/api/v1/export-onnx` | Export model as ONNX |

### Example: Analyze a Video

```bash
curl -X POST http://localhost:8080/api/v1/analyze \
  -F "video=@myvideo.mp4" \
  -H "X-Nostr-Pubkey: npub1..."
```

**Response:**

```json
{
  "id": "a1b2c3d4e5f6",
  "verdict": "AUTHENTIC",
  "verdict_emoji": "✅",
  "confidence": {
    "authentic": 0.92,
    "ai_generated": 0.08
  },
  "consistency_score": 0.95,
  "frames_analyzed": 16,
  "video_hash_sha256": "abc123...",
  "nostr_attestation": { "...": "..." }
}
```

---

## Project Structure

```
m4tr1x/
├── ai_detector.py       # AI detection module (EfficientNet-B0)
├── api_server.py        # FastAPI REST backend
├── auth.html            # Nostr authentication (PWA)
├── core.py              # Metadata scrubbing (ExifTool)
├── Dockerfile           # Docker deployment
├── index.html           # Frontend PWA (mobile-first)
├── mesh_server.py       # Local P2P mesh node
├── requirements.txt     # Python dependencies
├── train_detector.py    # Model training + ONNX export
└── uploader.py          # Fernet video encryption
```

---

## Roadmap

- [x] Mobile-first PWA frontend
- [x] Nostr authentication
- [x] AI Detection backend (EfficientNet-B0)
- [x] FastAPI REST API
- [x] ONNX export for mobile
- [x] Docker deployment
- [ ] Training on recent AI-generated video datasets (Sora, Runway, Kling)
- [ ] In-browser detection via ONNX Runtime Web
- [ ] Native Android/iOS app (React Native + ONNX Mobile)
- [ ] Automatic IPFS pinning for verified videos
- [ ] Nostr feed with verified/unverified content filter
- [ ] Dedicated M4TR1X Nostr relay

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

[MIT](LICENSE) — Free to use, fork, and build upon.

---

> *"In the age of synthetic reality, authenticity is the new resistance."*  
> **For the Truth. 👁️**

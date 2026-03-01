"""
M4TR1X AI Detector - Modulo di rilevamento video AI-generated
"""

import os
import sys
import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("m4tr1x.ai_detector")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_PATH = MODEL_DIR / "m4tr1x_detector.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.65
MAX_FRAMES_TO_ANALYZE = 16
FRAME_SIZE = (224, 224)

frame_transform = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class M4tr1xDetector(nn.Module):
    """Classificatore binario Real vs AI-Generated basato su EfficientNet-B0."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def load_model(model_path=None):
    """Carica il modello."""
    model = M4tr1xDetector().to(DEVICE)
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if path.exists():
        logger.info(f"Caricamento modello da {path}")
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    else:
        logger.warning(f"Modello non trovato in {path}. Uso backbone pre-trained.")
    model.eval()
    return model


def extract_frames(video_path, max_frames=MAX_FRAMES_TO_ANALYZE):
    """Estrae frame equidistanti dal video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire il video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("Video vuoto o non leggibile")
    n_frames = min(max_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    logger.info(f"Estratti {len(frames)}/{n_frames} frame")
    return frames


def analyze_frame(model, frame):
    """Analizza un singolo frame."""
    tensor = frame_transform(frame).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    labels = ["REAL", "AI_GENERATED"]
    return {
        "label": labels[pred.item()],
        "confidence": round(conf.item(), 4),
        "probabilities": {
            "real": round(probs[0][0].item(), 4),
            "ai_generated": round(probs[0][1].item(), 4),
        },
    }


def compute_video_hash(video_path):
    """Calcola SHA-256 del file video."""
    sha256 = hashlib.sha256()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def analyze_video(video_path, model=None):
    """Pipeline completa di analisi video."""
    if model is None:
        model = load_model()
    video_hash = compute_video_hash(video_path)
    frames = extract_frames(video_path)
    if not frames:
        return {"error": "Nessun frame estratto", "status": "ERRORE"}
    frame_results, real_scores, ai_scores = [], [], []
    for i, frame in enumerate(frames):
        result = analyze_frame(model, frame)
        result["frame_index"] = i
        frame_results.append(result)
        real_scores.append(result["probabilities"]["real"])
        ai_scores.append(result["probabilities"]["ai_generated"])
    weights = np.array([1.0] * len(frames))
    center = len(frames) // 2
    for i in range(len(frames)):
        dist = abs(i - center) / max(center, 1)
        weights[i] = 1.0 + (1.0 - dist) * 0.5
    weights /= weights.sum()
    avg_real = float(np.average(real_scores, weights=weights))
    avg_ai = float(np.average(ai_scores, weights=weights))
    if avg_ai > CONFIDENCE_THRESHOLD:
        verdict, verdict_emoji = "AI_GENERATED", "\u26a0\ufe0f"
    elif avg_real > CONFIDENCE_THRESHOLD:
        verdict, verdict_emoji = "AUTHENTIC", "\u2705"
    else:
        verdict, verdict_emoji = "UNCERTAIN", "\u2753"
    consistency = max(0.0, 1.0 - float(np.std(ai_scores)) * 2)
    return {
        "status": "OK",
        "video_hash_sha256": video_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict, "verdict_emoji": verdict_emoji,
        "confidence": {"authentic": round(avg_real, 4), "ai_generated": round(avg_ai, 4)},
        "consistency_score": round(consistency, 4),
        "frames_analyzed": len(frames),
        "frame_details": frame_results,
        "model_info": {"name": "M4TR1X-Detector-v1", "backbone": "EfficientNet-B0", "device": str(DEVICE)},
    }


def export_onnx(model, output_path="models/m4tr1x_detector.onnx"):
    """Esporta modello ONNX per browser/mobile."""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    torch.onnx.export(
        model, dummy, output_path, export_params=True, opset_version=13,
        do_constant_folding=True, input_names=["frame"], output_names=["prediction"],
        dynamic_axes={"frame": {0: "batch_size"}, "prediction": {0: "batch_size"}},
    )
    logger.info(f"ONNX esportato in {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ai_detector.py <video> | --export-onnx")
        sys.exit(1)
    if sys.argv[1] == "--export-onnx":
        export_onnx(load_model())
    else:
        print(json.dumps(analyze_video(sys.argv[1]), indent=2, ensure_ascii=False))

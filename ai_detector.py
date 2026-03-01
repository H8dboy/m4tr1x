"""
M4TR1X AI Detector - Modulo di rilevamento video AI-generated
Analizza frame estratti da video per determinare se sono reali o generati da AI.
Usa EfficientNet-B0 come backbone con supporto per ViT come alternativa.
"""

import os
import sys
import json
import hashlib
import tempfile
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

# --- CONFIGURAZIONE ---
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_PATH = MODEL_DIR / "m4tr1x_detector.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.65
MAX_FRAMES_TO_ANALYZE = 16
FRAME_SIZE = (224, 224)

# Transform standard per EfficientNet / ViT
frame_transform = transforms.Compose([
      transforms.Resize(FRAME_SIZE),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class M4tr1xDetector(nn.Module):
      """
          Classificatore binario Real vs AI-Generated basato su EfficientNet-B0.
              Leggero (~20MB), veloce, adatto a deployment mobile via ONNX.
                  """

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


def load_model(model_path: Optional[str] = None) -> M4tr1xDetector:
      """Carica il modello. Se non esiste, ritorna modello pre-trained base."""
      model = M4tr1xDetector().to(DEVICE)
      path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if path.exists():
              logger.info(f"Caricamento modello da {path}")
              state = torch.load(path, map_location=DEVICE, weights_only=True)
              model.load_state_dict(state)
else:
          logger.warning(
                        f"Modello fine-tuned non trovato in {path}. "
                        "Uso backbone EfficientNet-B0 pre-trained (accuracy limitata). "
                        "Esegui train_detector.py per addestrare su dati reali."
          )

    model.eval()
    return model


def extract_frames(video_path: str, max_frames: int = MAX_FRAMES_TO_ANALYZE) -> list:
      """
          Estrae frame equidistanti dal video.
              Strategia: prende frame uniformemente distribuiti per coprire tutto il video.
                  """
      cap = cv2.VideoCapture(video_path)
      if not cap.isOpened():
                raise ValueError(f"Impossibile aprire il video: {video_path}")

      total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      if total_frames <= 0:
                raise ValueError("Video vuoto o non leggibile")

      # Calcola indici dei frame da estrarre (uniformemente distribuiti)
      n_frames = min(max_frames, total_frames)
      indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
              cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
              ret, frame = cap.read()
              if ret:
                            # BGR -> RGB -> PIL
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_frame = Image.fromarray(frame_rgb)
                            frames.append(pil_frame)

          cap.release()
    logger.info(f"Estratti {len(frames)}/{n_frames} frame da {video_path}")
    return frames


def analyze_frame(model: M4tr1xDetector, frame: Image.Image) -> dict:
      """Analizza un singolo frame. Ritorna label + confidence."""
      tensor = frame_transform(frame).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
              output = model(tensor)
              probabilities = torch.softmax(output, dim=1)
              confidence, predicted = torch.max(probabilities, 1)

    labels = ["REAL", "AI_GENERATED"]
    return {
              "label": labels[predicted.item()],
              "confidence": round(confidence.item(), 4),
              "probabilities": {
                            "real": round(probabilities[0][0].item(), 4),
                            "ai_generated": round(probabilities[0][1].item(), 4),
              },
    }


def compute_video_hash(video_path: str) -> str:
      """Calcola SHA-256 del file video per integrità."""
      sha256 = hashlib.sha256()
      with open(video_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                              sha256.update(chunk)
                      return sha256.hexdigest()


def analyze_video(video_path: str, model: Optional[M4tr1xDetector] = None) -> dict:
      """
          Pipeline completa di analisi video.
              Ritorna un report JSON-serializable con verdetto, confidence e dettagli per frame.
                  """
      if model is None:
                model = load_model()

      video_hash = compute_video_hash(video_path)
      frames = extract_frames(video_path)

    if not frames:
              return {"error": "Nessun frame estratto dal video", "status": "ERRORE"}

    # Analisi frame per frame
    frame_results = []
    real_scores = []
    ai_scores = []

    for i, frame in enumerate(frames):
              result = analyze_frame(model, frame)
              result["frame_index"] = i
              frame_results.append(result)
              real_scores.append(result["probabilities"]["real"])
              ai_scores.append(result["probabilities"]["ai_generated"])

    # Verdetto aggregato: media pesata con attenzione ai frame centrali
    weights = np.array([1.0] * len(frames))
    # I frame centrali contano di piu (meno probabilita di artefatti intro/outro)
    center = len(frames) // 2
    for i in range(len(frames)):
              dist = abs(i - center) / max(center, 1)
              weights[i] = 1.0 + (1.0 - dist) * 0.5

    weights /= weights.sum()

    avg_real = float(np.average(real_scores, weights=weights))
    avg_ai = float(np.average(ai_scores, weights=weights))

    if avg_ai > CONFIDENCE_THRESHOLD:
              verdict = "AI_GENERATED"
              verdict_emoji = "\u26a0\ufe0f"
elif avg_real > CONFIDENCE_THRESHOLD:
          verdict = "AUTHENTIC"
          verdict_emoji = "\u2705"
else:
          verdict = "UNCERTAIN"
          verdict_emoji = "\u2753"

    # Calcola consistency (quanto i frame sono d'accordo tra loro)
      ai_std = float(np.std(ai_scores))
    consistency = max(0.0, 1.0 - ai_std * 2)

    report = {
              "status": "OK",
              "video_hash_sha256": video_hash,
              "timestamp": datetime.now(timezone.utc).isoformat(),
              "verdict": verdict,
              "verdict_emoji": verdict_emoji,
              "confidence": {
                            "authentic": round(avg_real, 4),
                            "ai_generated": round(avg_ai, 4),
              },
              "consistency_score": round(consistency, 4),
              "frames_analyzed": len(frames),
              "frame_details": frame_results,
              "model_info": {
                            "name": "M4TR1X-Detector-v1",
                            "backbone": "EfficientNet-B0",
                            "device": str(DEVICE),
              },
    }

    logger.info(
              f"{verdict_emoji} Verdetto: {verdict} "
              f"(real={avg_real:.2%}, ai={avg_ai:.2%}, consistency={consistency:.2%})"
    )

    return report


def export_onnx(model: M4tr1xDetector, output_path: str = "models/m4tr1x_detector.onnx"):
      """
          Esporta il modello in formato ONNX per inference nel browser
              via onnxruntime-web o su mobile via ONNX Runtime Mobile.
                  """
      model.eval()
      dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

    torch.onnx.export(
              model,
              dummy_input,
              output_path,
              export_params=True,
              opset_version=13,
              do_constant_folding=True,
              input_names=["frame"],
              output_names=["prediction"],
              dynamic_axes={
                            "frame": {0: "batch_size"},
                            "prediction": {0: "batch_size"},
              },
    )
    logger.info(f"Modello ONNX esportato in {output_path}")


# --- CLI ---
if __name__ == "__main__":
      if len(sys.argv) < 2:
                print("Uso: python ai_detector.py <percorso_video>")
                print("     python ai_detector.py --export-onnx")
                sys.exit(1)

      if sys.argv[1] == "--export-onnx":
                m = load_model()
                export_onnx(m)
else:
          report = analyze_video(sys.argv[1])
          print(json.dumps(report, indent=2, ensure_ascii=False))

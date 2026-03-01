"""
M4TR1X API Server - Backend FastAPI per il social decentralizzato
Gestisce upload video, analisi AI, autenticazione Nostr e API REST.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ai_detector import analyze_video, load_model, export_onnx

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("m4tr1x.api")

app = FastAPI(
      title="M4TR1X API",
      description="Social decentralizzato con AI verification",
      version="1.0.0",
)

app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
)

logger.info("Caricamento modello AI...")
ai_model = load_model()
logger.info("Modello AI pronto.")


class VideoAnalysisResponse(BaseModel):
      id: str
      status: str
      verdict: Optional[str] = None
      verdict_emoji: Optional[str] = None
      confidence: Optional[dict] = None
      consistency_score: Optional[float] = None
      frames_analyzed: Optional[int] = None
      video_hash_sha256: Optional[str] = None
      timestamp: Optional[str] = None
      nostr_attestation: Optional[dict] = None


class HealthResponse(BaseModel):
      status: str
      version: str
      ai_model_loaded: bool
      device: str


analysis_results: dict = {}


@app.get("/", response_model=HealthResponse)
async def health_check():
      import torch
      return HealthResponse(
          status="online",
          version="1.0.0",
          ai_model_loaded=ai_model is not None,
          device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
      )


@app.post("/api/v1/analyze", response_model=VideoAnalysisResponse)
async def analyze_uploaded_video(
      background_tasks: BackgroundTasks,
      video: UploadFile = File(...),
      nostr_pubkey: Optional[str] = Header(None, alias="X-Nostr-Pubkey"),
):
      ext = Path(video.filename).suffix.lower()
      if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Formato non supportato: {ext}")

      analysis_id = str(uuid.uuid4())[:12]
      temp_path = UPLOAD_DIR / f"{analysis_id}{ext}"

    try:
              with open(temp_path, "wb") as buffer:
                            content = await video.read()
                            if len(content) > MAX_FILE_SIZE:
                                              raise HTTPException(status_code=413, detail="File troppo grande (max 100MB)")
                                          buffer.write(content)
    except HTTPException:
              raise
except Exception as e:
          raise HTTPException(status_code=500, detail=f"Errore salvataggio: {str(e)}")

    try:
              report = analyze_video(str(temp_path), model=ai_model)
except Exception as e:
          logger.error(f"Errore analisi: {e}")
          raise HTTPException(status_code=500, detail=f"Errore analisi: {str(e)}")
finally:
          if temp_path.exists():
                        temp_path.unlink()

      nostr_attestation = None
    if nostr_pubkey:
              nostr_attestation = {
                            "pubkey": nostr_pubkey,
                            "event_kind": 30078,
                            "tags": [
                                              ["d", f"m4tr1x-verify-{analysis_id}"],
                                              ["verdict", report.get("verdict", "UNKNOWN")],
                                              ["hash", report.get("video_hash_sha256", "")],
                            ],
                            "content": f"M4TR1X Verification: {report.get('verdict', 'UNKNOWN')}",
              }

    result = VideoAnalysisResponse(
              id=analysis_id,
              status=report.get("status", "OK"),
              verdict=report.get("verdict"),
              verdict_emoji=report.get("verdict_emoji"),
              confidence=report.get("confidence"),
              consistency_score=report.get("consistency_score"),
              frames_analyzed=report.get("frames_analyzed"),
              video_hash_sha256=report.get("video_hash_sha256"),
              timestamp=report.get("timestamp"),
              nostr_attestation=nostr_attestation,
    )

    analysis_results[analysis_id] = result
    return result


@app.get("/api/v1/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
      if analysis_id not in analysis_results:
                raise HTTPException(status_code=404, detail="Analisi non trovata")
            return analysis_results[analysis_id]


@app.get("/api/v1/export-onnx")
async def trigger_onnx_export():
      try:
                export_onnx(ai_model)
                return {"status": "ok", "path": "models/m4tr1x_detector.onnx"}
except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if Path("index.html").exists():
      app.mount("/app", StaticFiles(directory=".", html=True), name="frontend")

if __name__ == "__main__":
      import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

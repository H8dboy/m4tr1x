"""
M4TR1X API Server - FastAPI backend for the decentralized social network
"""

import os
import uuid
import logging
import sqlite3
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import contextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ai_detector import analyze_video, load_model, export_onnx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DB_PATH = Path(os.getenv("DB_PATH", "m4tr1x.db"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "100")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
API_KEY = os.getenv("M4TR1X_API_KEY", "")          # empty = key auth disabled
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://h8dboy.github.io").split(",")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("m4tr1x.api")

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
    logger.info("Database initialized at %s", DB_PATH)

def save_result(result_id: str, data: dict):
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO analysis_results (id, data, created_at) VALUES (?, ?, ?)",
            (result_id, json.dumps(data), datetime.utcnow().isoformat())
        )
        conn.commit()

def load_result(result_id: str) -> Optional[dict]:
    with sqlite3.connect(str(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT data FROM analysis_results WHERE id = ?", (result_id,)
        ).fetchone()
    return json.loads(row[0]) if row else None

# ---------------------------------------------------------------------------
# API key dependency
# ---------------------------------------------------------------------------
def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="M4TR1X API",
    description="Decentralized social network with integrated AI video verification",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-Nostr-Pubkey", "X-API-Key", "Content-Type"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
init_db()
logger.info("Loading AI model...")
ai_model = load_model()
logger.info("AI model ready.")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_model=HealthResponse)
async def health_check():
    import torch
    return HealthResponse(
        status="online",
        version="1.0.0",
        ai_model_loaded=ai_model is not None,
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    )

@app.post(
    "/api/v1/analyze",
    response_model=VideoAnalysisResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def analyze_uploaded_video(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    nostr_pubkey: Optional[str] = Header(None, alias="X-Nostr-Pubkey"),
):
    ext = Path(video.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    analysis_id = str(uuid.uuid4())[:12]
    temp_path = UPLOAD_DIR / f"{analysis_id}{ext}"

    try:
        with open(temp_path, "wb") as buffer:
            content = await video.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large (max {MAX_FILE_SIZE // 1024 // 1024} MB)"
                )
            buffer.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

    try:
        report = analyze_video(str(temp_path), model=ai_model)
    except Exception as e:
        logger.error("Analysis error: %s", e)
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
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

    save_result(analysis_id, result.dict())
    return result


@app.get("/api/v1/analysis/{analysis_id}", dependencies=[Depends(verify_api_key)])
async def get_analysis(analysis_id: str):
    data = load_result(analysis_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return data


@app.get("/api/v1/export-onnx", dependencies=[Depends(verify_api_key)])
@limiter.limit("2/hour")
async def trigger_onnx_export(request: Request):
    try:
        export_onnx(ai_model)
        return {"status": "ok", "path": "models/m4tr1x_detector.onnx"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if Path("index.html").exists():
    app.mount("/app", StaticFiles(directory=".", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

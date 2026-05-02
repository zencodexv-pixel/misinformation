# FastAPI entrypoint for Render / uvicorn (same behavior as ai_server Flask routes).
# Local (this folder as cwd): python -m uvicorn app:app --host 0.0.0.0 --port 8000
# From repo root: python -m uvicorn guardian_misinformation_module.app:app --host 0.0.0.0 --port 8000
# Render: Root = this folder; Start: uvicorn app:app --host 0.0.0.0 --port $PORT
# Render: set MISINFO_HF_REPO=your-org/misinformation after uploading weights; use HF_TOKEN if the repo is private.
# Or USE_FALLBACK_MODEL=1 if you skip custom weights.

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_pkg = Path(__file__).resolve().parent
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

import ai_server as _mis

app = FastAPI(title="Guardian Misinformation Module")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., description="Text to analyze")
    risk_level: Optional[str] = None
    susceptibility_score: Optional[float] = None


@app.get("/")
def home() -> dict[str, Any]:
    return {
        "status": "AI Server Running",
        "ai_ready": _mis.AI_READY,
        "message": "Behavior-Aware Misinformation Detector API",
    }


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    health_data: dict[str, Any] = {
        "status": "healthy" if _mis.AI_READY else "unhealthy",
        "ai_ready": _mis.AI_READY,
        "model_type": _mis.MODEL_TYPE,
        "timestamp": datetime.now().isoformat(),
    }
    if not _mis.AI_READY and _mis.MODEL_ERROR:
        health_data["error"] = _mis.MODEL_ERROR
        health_data["error_type"] = "model_loading_failed"
    elif _mis.AI_READY and _mis.MODEL_TYPE == "fallback":
        health_data["warning"] = "Using fallback model - limited functionality"
    return health_data


@app.post("/api/chat")
def chat_endpoint(body: ChatRequest) -> dict[str, Any]:
    if not _mis.AI_READY:
        error_msg = "AI model not loaded"
        if _mis.MODEL_ERROR:
            error_msg = f"AI model failed to load: {_mis.MODEL_ERROR}"
        return {
            "success": False,
            "error": error_msg,
            "error_details": _mis.MODEL_ERROR or "Model directory may be missing or corrupted",
            "response": "AI service is currently unavailable. Please check the server console for details.",
        }

    message = (body.message or "").strip()
    if not message:
        return {
            "success": False,
            "error": "No message provided",
            "response": "Please provide a message to analyze.",
        }

    return _mis.generate_with_timeout(
        _mis.chatbot,
        message,
        body.risk_level,
        body.susceptibility_score,
        22,
    )

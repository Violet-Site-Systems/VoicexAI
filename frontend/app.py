"""
FastAPI Dashboard for EPPN

Provides endpoints to list documents, summaries, and ethics reports.
"""

import json
import os
from typing import Dict, Any, List
from io import BytesIO

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

# Text-to-Speech
try:
    from gtts import gTTS
except Exception:  # pragma: no cover
    gTTS = None  # type: ignore


DATA_DIR = "data"
SUMMARY_FILE = os.path.join(DATA_DIR, "summaries.jsonl")
ETHICS_FILE = os.path.join(DATA_DIR, "ethics.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="EPPN Dashboard")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


@app.get("/docs")
def list_documents():
    summaries = read_jsonl(SUMMARY_FILE)
    ethics = read_jsonl(ETHICS_FILE)
    doc_ids = sorted({s.get("doc_id") for s in summaries} | {e.get("doc_id") for e in ethics})
    return {"documents": doc_ids}


@app.get("/docs/{doc_id}/summary")
def get_summary(doc_id: str):
    summaries = [s for s in read_jsonl(SUMMARY_FILE) if s.get("doc_id") == doc_id]
    return summaries[-1] if summaries else JSONResponse(status_code=404, content={"error": "Not found"})


@app.get("/docs/{doc_id}/ethics")
def get_ethics(doc_id: str):
    reports = [e for e in read_jsonl(ETHICS_FILE) if e.get("doc_id") == doc_id]
    return reports[-1] if reports else JSONResponse(status_code=404, content={"error": "Not found"})


# Text-to-Speech endpoint
@app.post("/tts")
def synthesize_speech(payload: Dict[str, Any]):
    """Synthesize speech from provided text and return MP3 audio.

    Request JSON:
    - text: string (required)
    - lang: BCP-47 language code, e.g., "en" (default: "en")
    - slow: bool for slower speech (default: False)
    """
    if gTTS is None:
        return JSONResponse(status_code=500, content={"error": "TTS engine unavailable"})

    text = str(payload.get("text", "")).strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "'text' is required"})

    lang = str(payload.get("lang", "en"))
    slow = bool(payload.get("slow", False))

    # Generate MP3 in-memory
    mp3_bytes = BytesIO()
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.write_to_fp(mp3_bytes)
    mp3_bytes.seek(0)

    headers = {
        "Content-Disposition": "inline; filename=tts.mp3"
    }
    return StreamingResponse(mp3_bytes, media_type="audio/mpeg", headers=headers)


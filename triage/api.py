"""FastAPI service.  Run:  uvicorn triage.api:app --port 8000"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from .pipeline import DEFAULT_MODEL_DIR, Triage

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "triage" not in STATE:
        STATE["triage"] = Triage.from_dir(Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)))
    yield


app = FastAPI(title="Banking Support Triage", version="1.0.0", lifespan=lifespan)
# Public demo: the hosted demo page (e.g. on Netlify) calls this API from another origin.
# No cookies or auth are involved, so a wildcard is safe here.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["content-type"])


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class Batch(BaseModel):
    texts: list[str] = Field(min_length=20, max_length=5000)


def _t() -> Triage:
    if "triage" not in STATE:
        raise HTTPException(503, "model not loaded")
    return STATE["triage"]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/demo")


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo():
    return (Path(__file__).parent / "demo.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    t = _t()
    return {"status": "ok", "model": t.meta["kind"], "llm_drafting": t.client is not None}


@app.get("/model")
def model_card():
    m = _t().meta
    return {k: m[k] for k in ("kind", "params", "threshold", "target_accuracy", "metrics", "trained_at") if k in m}


@app.post("/triage")
def triage(msg: Message):
    return _t().triage(msg.text)


@app.post("/drift")
def drift(batch: Batch):
    return _t().drift_report(batch.texts)

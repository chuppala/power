from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .agent import run_agent, tool_explain_row
from .pipeline import Pipeline

load_dotenv()

BASE = Path(__file__).resolve().parents[1]
FRONTEND = BASE / "frontend"

app = FastAPI(
    title="Power Generation Forecasting & Variance Explanation Agent",
    version="1.0.0",
    description="Weather+outage regression baseline, variance analysis against the operational forecast, and an explanation agent.",
)


@lru_cache(maxsize=1)
def pipeline() -> Pipeline:
    """Built once per process (well under a second on the bundled data)."""
    return Pipeline()


class Question(BaseModel):
    text: str = Field("", max_length=500, description="Natural-language question, e.g. 'Why was Aug 15 unit A so far off?'")
    use_llm: bool = True


class ForecastInput(BaseModel):
    irradiance_wm2: float = Field(..., ge=0, le=1400)
    wind_speed_ms: float = Field(..., ge=0, le=40)
    temperature_c: float = Field(..., ge=-30, le=60)
    outage_flag: int = Field(0, ge=0, le=1)


@app.get("/api/health")
def health():
    p = pipeline()
    return {"status": "ok", "rows": len(p.df), "model": p.best_name, "llm_configured": bool(os.getenv("GEMINI_API_KEY")) and os.getenv("GEMINI_API_KEY") != "PASTE_YOUR_NEW_API_KEY_HERE"}


@app.get("/api/demo")
def demo():
    return {
        "case_study": app.title,
        "endpoints": ["GET /api/summary", "GET /api/series", "GET /api/variance", "POST /api/forecast", "POST /api/run", "GET /docs"],
    }


@app.get("/api/summary")
def summary():
    return pipeline().summary()


@app.get("/api/series")
def series(asset: str = Query("all", description="'A', 'B' or 'all'")):
    return pipeline().series(asset)


@app.get("/api/variance")
def variance(date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"), asset: str = Query(..., min_length=1, max_length=8)):
    p = pipeline()
    row = p.row(date, asset)
    if row is None:
        raise HTTPException(404, f"No record for {date} / unit {asset.upper()}")
    return tool_explain_row(p, row)


@app.post("/api/forecast")
def forecast(inp: ForecastInput):
    return pipeline().predict(**inp.model_dump())


@app.post("/api/run")
def run(q: Question):
    return run_agent(pipeline(), q.text, use_llm=q.use_llm)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

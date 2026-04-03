"""
AI-Based Dental Radiograph Analysis System
FastAPI Backend — Entry Point
"""

import os
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from processor import DentalImageProcessor
from reporter import DentalReportGenerator

# Use local model if trained, otherwise fall back to Gemini API
_MODEL_PATH = Path(__file__).parent / "models" / "efficientnet_dental.pth"
if _MODEL_PATH.exists():
    from local_analyzer import LocalDentalAnalyzer as _Analyzer
else:
    from analyzer import DentalAnalyzer as _Analyzer

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Dental Radiograph Analysis System",
    description="Analyzes panoramic dental X-rays (OPG) using Claude Vision AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── Static files ───────────────────────────────────────────────────────────────
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# ── Services ───────────────────────────────────────────────────────────────────
analyzer  = _Analyzer()
processor = DentalImageProcessor()
reporter  = DentalReportGenerator()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    # Validate file type
    content_type = (file.content_type or "").lower()
    suffix = Path(file.filename or "file.jpg").suffix.lower()

    if content_type not in ALLOWED_TYPES and suffix not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG, PNG, or WebP image.",
        )

    session_id = str(uuid.uuid4())
    image_path = UPLOAD_DIR / f"{session_id}{suffix}"

    try:
        # Save upload
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 1 — Analyse with Claude Vision AI
        analysis = analyzer.analyze(str(image_path))

        # 2 — Create colour-coded segmentation overlay + FDI chart
        processed = processor.process_image(str(image_path), analysis)

        # 3 — Generate PDF diagnostic report
        report_path = RESULTS_DIR / f"report_{session_id}.pdf"
        reporter.generate(analysis, processed, str(report_path))

        return JSONResponse({
            "session_id": session_id,
            "analysis": analysis,
            "images": {
                "enhanced": processed["enhanced"],
                "overlay":  processed["overlay"],
                "fdi_chart": processed["fdi_chart"],
            },
            "tooth_regions": processed["tooth_regions"],
            "report_url": f"/results/report_{session_id}.pdf",
            "status": "success",
        })

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        if image_path.exists():
            image_path.unlink()


# ── Dev entry ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

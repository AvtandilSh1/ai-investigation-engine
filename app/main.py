import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

settings = get_settings()
Path(settings.docs_path).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AI Investigation Engine",
    description="Multi-document investigation with hybrid retrieval and grounded answers.",
    version="1.0.0",
)

app.include_router(router, prefix="/api")

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
def index():
    index_path = FRONTEND / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "frontend missing", "docs": "/docs"}

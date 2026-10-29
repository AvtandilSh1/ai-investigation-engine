import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.deps import get_engine, get_indexer, get_llm, get_meta, get_searcher, get_settings
from app.models.schemas import (
    DocumentInfo,
    HealthResponse,
    InvestigateRequest,
    InvestigateResponse,
)
from app.services.chunking import compare_strategies

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    meta = get_meta()
    return HealthResponse(
        status="ok",
        document_count=meta.doc_count(),
        chunk_count=meta.chunk_count(),
        llm_configured=get_llm().enabled,
    )


def _as_doc(record: dict) -> DocumentInfo:
    return DocumentInfo.model_validate(record)


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    return [_as_doc(d) for d in get_meta().list_docs()]


@router.post("/documents/upload", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    strategy: Optional[str] = Query(None),
    force: bool = Query(False),
):
    suffix = Path(file.filename or "upload.txt").suffix
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        record = get_indexer().ingest_file(
            tmp_path,
            strategy=strategy,
            source="upload",
            force=force,
        )
        # rename stored filename to original
        if file.filename and record.get("filename") != file.filename:
            record["filename"] = file.filename
            get_meta().upsert_doc(record)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    return _as_doc(record)


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not get_indexer().delete(doc_id):
        raise HTTPException(404, "document not found")
    return {"deleted": doc_id}


@router.put("/documents/{doc_id}", response_model=DocumentInfo)
async def update_document(doc_id: str, file: UploadFile = File(...), strategy: Optional[str] = None):
    if not get_meta().get(doc_id):
        raise HTTPException(404, "document not found")
    suffix = Path(file.filename or "upload.txt").suffix
    raw = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        get_indexer().delete(doc_id)
        record = get_indexer().ingest_file(tmp_path, strategy=strategy, source="upload", force=True)
        record["filename"] = file.filename or record["filename"]
        get_meta().upsert_doc(record)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)
    return _as_doc(record)


@router.post("/investigate", response_model=InvestigateResponse)
def investigate(body: InvestigateRequest):
    try:
        return get_engine().investigate(body.question, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(500, f"investigation failed: {e}") from e


@router.get("/search")
def search(q: str = Query(..., min_length=1), top_k: int = 8):
    hits = get_searcher().search(q, top_k=top_k)
    return {"query": q, "hits": hits}


@router.post("/chunking/compare")
async def chunking_compare(file: UploadFile = File(...)):
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")
    s = get_settings()
    return compare_strategies(text, size=s.chunk_size, overlap=s.chunk_overlap)

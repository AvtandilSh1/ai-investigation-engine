from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: str
    filename: str
    format: str
    title: Optional[str] = None
    source: Optional[str] = None
    chunk_count: int = 0
    content_hash: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    deduplicated: bool = False


class Citation(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    excerpt: str
    score: float
    page: Optional[int] = None


class ContradictionNote(BaseModel):
    topic: str
    sources: list[str]
    detail: str


class InvestigateRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    chunk_strategy: Optional[str] = None


class InvestigateResponse(BaseModel):
    answer: str
    confidence: float
    abstained: bool
    citations: list[Citation]
    contradictions: list[ContradictionNote]
    steps: list[str]
    retrieved_count: int


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    document_count: int
    chunk_count: int
    llm_configured: bool


class EvalSummary(BaseModel):
    config_name: str
    n_questions: int
    recall_at_k: float
    mrr: float
    answer_correctness: float
    citation_accuracy: float
    abstain_precision: float
    notes: str = ""

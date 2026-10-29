from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.embeddings import get_embedder
from app.services.hybrid_search import HybridSearcher
from app.services.indexer import DocumentIndexer
from app.services.llm import LLMClient
from app.services.meta_store import MetaStore
from app.services.rag import InvestigationEngine
from app.services.vector_store import VectorStore


@lru_cache
def get_meta() -> MetaStore:
    return MetaStore(get_settings().meta_db_path)


@lru_cache
def get_vectors() -> VectorStore:
    return VectorStore(get_settings().chroma_path)


@lru_cache
def get_searcher() -> HybridSearcher:
    s = get_settings()
    return HybridSearcher(
        get_vectors(),
        get_meta(),
        get_embedder(s.embedding_model),
        alpha=s.hybrid_alpha,
    )


@lru_cache
def get_indexer() -> DocumentIndexer:
    s = get_settings()
    return DocumentIndexer(
        s,
        get_meta(),
        get_vectors(),
        get_embedder(s.embedding_model),
        get_searcher(),
    )


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient(get_settings())


@lru_cache
def get_engine() -> InvestigationEngine:
    return InvestigationEngine(get_searcher(), get_llm(), default_top_k=get_settings().top_k)

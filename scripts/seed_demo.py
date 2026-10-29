"""Quick seed for local UI demos (synthetic docs + a few linked wiki-style pages)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_corpus import SYNTHETIC


EXTRA = [
    {
        "filename": "wiki_ada_lovelace.txt",
        "title": "Ada Lovelace",
        "text": (
            "Ada Lovelace\n\nAugusta Ada King, Countess of Lovelace was an English mathematician "
            "and writer, chiefly known for her work on Charles Babbage's proposed mechanical "
            "general-purpose computer, the Analytical Engine. She was the first to recognise "
            "that the machine had applications beyond pure calculation."
        ),
    },
    {
        "filename": "wiki_analytical_engine.txt",
        "title": "Analytical Engine",
        "text": (
            "Analytical Engine\n\nThe Analytical Engine was a proposed mechanical general-purpose "
            "computer designed by English mathematician and computer pioneer Charles Babbage. "
            "It was first described in 1837. Ada Lovelace wrote what is considered the first "
            "computer algorithm intended for the engine."
        ),
    },
    {
        "filename": "wiki_babbage.txt",
        "title": "Charles Babbage",
        "text": (
            "Charles Babbage\n\nCharles Babbage was an English polymath. He originated the "
            "concept of a digital programmable computer. Babbage is considered by some to be "
            "a father of the computer. He invented the Difference Engine and designed the Analytical Engine."
        ),
    },
]


def run():
    from app.core.config import get_settings
    from app.services.embeddings import get_embedder
    from app.services.hybrid_search import HybridSearcher
    from app.services.indexer import DocumentIndexer
    from app.services.meta_store import MetaStore
    from app.services.vector_store import VectorStore

    settings = get_settings()
    meta = MetaStore(settings.meta_db_path)
    vectors = VectorStore(settings.chroma_path)
    embedder = get_embedder(settings.embedding_model)
    searcher = HybridSearcher(vectors, meta, embedder, alpha=settings.hybrid_alpha)
    indexer = DocumentIndexer(settings, meta, vectors, embedder, searcher)

    for item in SYNTHETIC + EXTRA:
        path = Path(settings.docs_path) / "_seed" / item["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item["text"], encoding="utf-8")
        indexer.ingest_file(path, source="demo", force=False)

    print("seeded docs", meta.doc_count(), "chunks", meta.chunk_count())


if __name__ == "__main__":
    run()

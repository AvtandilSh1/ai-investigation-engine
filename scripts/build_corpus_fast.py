"""
Faster bulk indexer to reach >= 50k records.
Writes sqlite + chroma in batches instead of one API-style ingest per doc.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_corpus import SYNTHETIC, generated_records, load_hotpot_paragraphs, seed_synthetic
from app.core.config import get_settings
from app.services.chunking import chunk_text
from app.services.embeddings import get_embedder
from app.services.meta_store import MetaStore
from app.services.vector_store import VectorStore, new_id
from app.services.document_processor import content_hash


def now():
    return datetime.now(timezone.utc).isoformat()


def bulk_add(meta: MetaStore, vectors: VectorStore, embedder, rows: list[dict], strategy: str, size: int, overlap: int):
    ids, docs, metas = [], [], []
    for row in rows:
        text = row["text"].strip()
        if not text:
            continue
        h = content_hash(text)
        if meta.get_by_hash(h):
            continue
        chunks = chunk_text(text, strategy=strategy, size=size, overlap=overlap)
        if not chunks:
            continue
        doc_id = new_id("doc_")
        ts = now()
        record = {
            "id": doc_id,
            "filename": row["filename"],
            "format": "txt",
            "title": row.get("title") or row["filename"],
            "source": row.get("source"),
            "content_hash": h,
            "chunk_count": len(chunks),
            "created_at": ts,
            "updated_at": ts,
            "metadata": row.get("extra") or {},
        }
        chunk_rows = []
        for ch in chunks:
            cid = f"{doc_id}_{ch.index}"
            chunk_rows.append({"id": cid, "index": ch.index, "text": ch.text, "strategy": ch.strategy})
            ids.append(cid)
            docs.append(ch.text)
            metas.append(
                {
                    "document_id": doc_id,
                    "filename": record["filename"],
                    "title": record["title"],
                    "chunk_index": ch.index,
                    "strategy": ch.strategy,
                    "format": "txt",
                    "source": record.get("source") or "",
                }
            )
        meta.upsert_doc(record)
        meta.replace_chunks(doc_id, chunk_rows)

    if not ids:
        return 0
    emb = embedder.encode(docs, batch_size=256).tolist()
    # chroma upsert in slices
    step = 500
    for i in range(0, len(ids), step):
        vectors.upsert(ids[i : i + step], emb[i : i + step], docs[i : i + step], metas[i : i + step])
    return len({m["document_id"] for m in metas})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=50000)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--skip-hotpot", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    meta = MetaStore(settings.meta_db_path)
    vectors = VectorStore(settings.chroma_path)
    embedder = get_embedder(settings.embedding_model)

    from app.services.hybrid_search import HybridSearcher
    from app.services.indexer import DocumentIndexer

    searcher = HybridSearcher(vectors, meta, embedder, alpha=settings.hybrid_alpha)
    indexer = DocumentIndexer(settings, meta, vectors, embedder, searcher)
    seed_synthetic(indexer, settings)

    need = max(0, args.target - meta.doc_count())
    rows: list[dict] = []
    if need and not args.skip_hotpot:
        try:
            print("loading hotpot…")
            rows = load_hotpot_paragraphs(limit=need + 500)
            print("hotpot paragraphs", len(rows))
        except Exception as e:
            print("hotpot failed:", e)
            rows = []
    if need and len(rows) < need:
        rows.extend(generated_records(len(rows), need - len(rows) + 50))

    added = 0
    for i in range(0, len(rows), args.batch):
        if meta.doc_count() >= args.target:
            break
        chunk = rows[i : i + args.batch]
        n = bulk_add(
            meta,
            vectors,
            embedder,
            chunk,
            settings.chunk_strategy,
            settings.chunk_size,
            settings.chunk_overlap,
        )
        added += n
        print(f"batch {i//args.batch + 1}: +{n} docs, total={meta.doc_count()}, chunks={meta.chunk_count()}")

    searcher.invalidate()
    print("done docs", meta.doc_count(), "chunks", meta.chunk_count(), "added", added)


if __name__ == "__main__":
    main()

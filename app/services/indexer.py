import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.config import Settings
from app.services.chunking import chunk_text
from app.services.document_processor import load_document, load_text_record
from app.services.embeddings import Embedder
from app.services.hybrid_search import HybridSearcher
from app.services.meta_store import MetaStore
from app.services.vector_store import VectorStore, new_id


class DocumentIndexer:
    def __init__(
        self,
        settings: Settings,
        meta: MetaStore,
        vectors: VectorStore,
        embedder: Embedder,
        searcher: HybridSearcher,
    ):
        self.settings = settings
        self.meta = meta
        self.vectors = vectors
        self.embedder = embedder
        self.searcher = searcher
        Path(settings.docs_path).mkdir(parents=True, exist_ok=True)

    def ingest_file(
        self,
        path: Path,
        *,
        strategy: Optional[str] = None,
        source: Optional[str] = None,
        force: bool = False,
    ) -> dict:
        loaded = load_document(path)
        return self._ingest_loaded(loaded, strategy=strategy, source=source, force=force, keep_file=path)

    def ingest_text(
        self,
        text: str,
        filename: str,
        *,
        title: Optional[str] = None,
        source: Optional[str] = None,
        strategy: Optional[str] = None,
        force: bool = False,
        extra: Optional[dict] = None,
    ) -> dict:
        loaded = load_text_record(text, filename=filename, title=title, source=source, extra=extra)
        return self._ingest_loaded(loaded, strategy=strategy, source=source, force=force)

    def _ingest_loaded(
        self,
        loaded: dict,
        *,
        strategy: Optional[str],
        source: Optional[str],
        force: bool,
        keep_file: Optional[Path] = None,
    ) -> dict:
        existing = self.meta.get_by_hash(loaded["content_hash"])
        if existing and not force:
            return {**existing, "deduplicated": True}

        strategy = strategy or self.settings.chunk_strategy
        chunks = chunk_text(
            loaded["text"],
            strategy=strategy,
            size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("chunking produced no chunks")

        doc_id = existing["id"] if existing else new_id("doc_")
        now = datetime.now(timezone.utc).isoformat()

        # persist original file copy
        dest_dir = Path(self.settings.docs_path) / doc_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / loaded["filename"]
        if keep_file and keep_file.exists():
            if keep_file.resolve() != dest.resolve():
                shutil.copy2(keep_file, dest)
        else:
            dest.write_text(loaded["text"], encoding="utf-8")

        chunk_rows = []
        ids, docs, metas = [], [], []
        for ch in chunks:
            cid = f"{doc_id}_{ch.index}"
            chunk_rows.append(
                {
                    "id": cid,
                    "index": ch.index,
                    "text": ch.text,
                    "strategy": ch.strategy,
                }
            )
            ids.append(cid)
            docs.append(ch.text)
            metas.append(
                {
                    "document_id": doc_id,
                    "filename": loaded["filename"],
                    "title": loaded.get("title") or loaded["filename"],
                    "chunk_index": ch.index,
                    "strategy": ch.strategy,
                    "format": loaded["format"],
                    "source": source or loaded.get("source") or "",
                }
            )

        embeddings = self.embedder.encode(docs).tolist()
        self.vectors.delete_by_document(doc_id)
        self.vectors.upsert(ids, embeddings, docs, metas)

        record = {
            "id": doc_id,
            "filename": loaded["filename"],
            "format": loaded["format"],
            "title": loaded.get("title"),
            "source": source or loaded.get("source"),
            "content_hash": loaded["content_hash"],
            "chunk_count": len(chunk_rows),
            "created_at": existing["created_at"] if existing else loaded["created_at"],
            "updated_at": now,
            "metadata": loaded.get("metadata") or {},
            "deduplicated": False,
        }
        self.meta.upsert_doc(record)
        self.meta.replace_chunks(doc_id, chunk_rows)
        self.searcher.invalidate()
        return record

    def delete(self, doc_id: str) -> bool:
        ok = self.meta.delete_doc(doc_id)
        self.vectors.delete_by_document(doc_id)
        doc_dir = Path(self.settings.docs_path) / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)
        self.searcher.invalidate()
        return ok

    def update_file(self, doc_id: str, path: Path, strategy: Optional[str] = None) -> dict:
        old = self.meta.get(doc_id)
        if not old:
            raise KeyError(doc_id)
        # force reindex under same id
        loaded = load_document(path)
        # temporarily remove hash uniqueness by deleting first
        self.delete(doc_id)
        result = self._ingest_loaded(loaded, strategy=strategy, source=old.get("source"), force=True, keep_file=path)
        return result

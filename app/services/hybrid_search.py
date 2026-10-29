from typing import Optional

from app.services.embeddings import Embedder
from app.services.meta_store import MetaStore
from app.services.ranking import bm25_rank, normalize_scores, rerank, rrf_fuse
from app.services.vector_store import VectorStore


class HybridSearcher:
    def __init__(
        self,
        vector_store: VectorStore,
        meta: MetaStore,
        embedder: Embedder,
        alpha: float = 0.55,
    ):
        self.vectors = vector_store
        self.meta = meta
        self.embedder = embedder
        self.alpha = alpha
        self._bm25_cache: Optional[list[dict]] = None

    def invalidate(self):
        self._bm25_cache = None

    def _corpus(self) -> list[dict]:
        if self._bm25_cache is None:
            rows = self.meta.all_chunks()
            self._bm25_cache = [
                {
                    "chunk_id": r["id"],
                    "document_id": r["document_id"],
                    "document_name": r.get("title") or r.get("filename") or r["document_id"],
                    "text": r["text"],
                    "metadata": {
                        "document_id": r["document_id"],
                        "filename": r.get("filename"),
                        "title": r.get("title"),
                        "chunk_index": r.get("chunk_index"),
                    },
                }
                for r in rows
            ]
        return self._bm25_cache

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        if not query.strip():
            return []

        dense_n = max(top_k * 4, 20)
        emb = self.embedder.encode_one(query)
        dense_hits = self.vectors.query(emb, n_results=dense_n)

        dense_norm = []
        for h in dense_hits:
            meta = h.get("metadata") or {}
            dense_norm.append(
                {
                    "chunk_id": h["chunk_id"],
                    "document_id": meta.get("document_id", ""),
                    "document_name": meta.get("title") or meta.get("filename") or meta.get("document_id", ""),
                    "text": h["text"],
                    "score": h["score"],
                    "metadata": meta,
                    "channel": "dense",
                }
            )
        dense_norm = normalize_scores(dense_norm)

        sparse = bm25_rank(query, self._corpus(), top_n=dense_n)
        sparse_norm = []
        for h in sparse:
            item = dict(h)
            item["channel"] = "sparse"
            sparse_norm.append(item)
        sparse_norm = normalize_scores(sparse_norm)

        by_id: dict[str, dict] = {}
        for h in dense_norm + sparse_norm:
            cid = h["chunk_id"]
            if cid not in by_id:
                by_id[cid] = dict(h)
                by_id[cid]["dense"] = 0.0
                by_id[cid]["sparse"] = 0.0
            if h.get("channel") == "dense":
                by_id[cid]["dense"] = h["score"]
            else:
                by_id[cid]["sparse"] = h["score"]
            # keep best text/meta
            if len(h.get("text") or "") > len(by_id[cid].get("text") or ""):
                by_id[cid]["text"] = h["text"]
                by_id[cid]["metadata"] = h.get("metadata") or by_id[cid].get("metadata")

        dense_ids = [h["chunk_id"] for h in dense_norm]
        sparse_ids = [h["chunk_id"] for h in sparse_norm]
        rrf = rrf_fuse([dense_ids, sparse_ids], k=60)

        fused = []
        for cid, item in by_id.items():
            hybrid = self.alpha * item.get("dense", 0.0) + (1 - self.alpha) * item.get("sparse", 0.0)
            hybrid = 0.7 * hybrid + 0.3 * rrf.get(cid, 0.0)
            item["score"] = hybrid
            fused.append(item)

        fused.sort(key=lambda x: x["score"], reverse=True)
        # custom evidence consistency re-ranker
        return rerank(query, fused[: max(top_k * 3, 15)], top_k=top_k)

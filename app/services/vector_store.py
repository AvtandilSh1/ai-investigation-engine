import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    def __init__(self, persist_path: str, collection_name: str = "investigation"):
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        # chroma metadata values must be scalars
        clean_meta = []
        for m in metadatas:
            clean_meta.append({k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in m.items()})
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=clean_meta,
        )

    def delete_by_document(self, document_id: str) -> None:
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception:
            # older collections may be empty
            pass

    def delete_ids(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def query(
        self,
        embedding: list[float],
        n_results: int = 8,
        where: Optional[dict] = None,
    ) -> list[dict]:
        available = max(1, self.collection.count())
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": max(1, min(n_results, available)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        res = self.collection.query(**kwargs)
        hits = []
        if not res or not res.get("ids"):
            return hits
        for i, cid in enumerate(res["ids"][0]):
            dist = res["distances"][0][i] if res.get("distances") else 1.0
            score = 1.0 / (1.0 + float(dist))
            hits.append(
                {
                    "chunk_id": cid,
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i] or {},
                    "score": score,
                    "distance": float(dist),
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex}" if prefix else uuid.uuid4().hex

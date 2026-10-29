import json
import sqlite3
from pathlib import Path
from typing import Optional


class MetaStore:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    format TEXT NOT NULL,
                    title TEXT,
                    source TEXT,
                    content_hash TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT,
                    UNIQUE(content_hash)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER,
                    text TEXT NOT NULL,
                    strategy TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
            conn.commit()

    def get_by_hash(self, content_hash: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            return self._doc_row(row) if row else None

    def get(self, doc_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return self._doc_row(row) if row else None

    def list_docs(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
            return [self._doc_row(r) for r in rows]

    def upsert_doc(self, doc: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents
                (id, filename, format, title, source, content_hash, chunk_count, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    filename=excluded.filename,
                    format=excluded.format,
                    title=excluded.title,
                    source=excluded.source,
                    content_hash=excluded.content_hash,
                    chunk_count=excluded.chunk_count,
                    updated_at=excluded.updated_at,
                    metadata=excluded.metadata
                """,
                (
                    doc["id"],
                    doc["filename"],
                    doc["format"],
                    doc.get("title"),
                    doc.get("source"),
                    doc["content_hash"],
                    doc.get("chunk_count", 0),
                    doc["created_at"],
                    doc["updated_at"],
                    json.dumps(doc.get("metadata") or {}),
                ),
            )
            conn.commit()

    def replace_chunks(self, document_id: str, chunks: list[dict]) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                "INSERT INTO chunks (id, document_id, chunk_index, text, strategy) VALUES (?, ?, ?, ?, ?)",
                [
                    (c["id"], document_id, c["index"], c["text"], c.get("strategy", ""))
                    for c in chunks
                ],
            )
            conn.commit()

    def delete_doc(self, doc_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            return cur.rowcount > 0

    def all_chunks(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.*, d.filename, d.title
                FROM chunks c JOIN documents d ON d.id = c.document_id
                ORDER BY c.document_id, c.chunk_index
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def chunk_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def doc_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def _doc_row(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        return d

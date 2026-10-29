import io

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "document_count" in body


def test_upload_and_search_flow(tmp_path, monkeypatch):
    # isolate storage for this test
    from app.core import config
    from app import deps

    docs = tmp_path / "docs"
    chroma = tmp_path / "chroma"
    meta = tmp_path / "meta.sqlite3"
    docs.mkdir()
    chroma.mkdir()

    monkeypatch.setenv("DOCS_PATH", str(docs))
    monkeypatch.setenv("CHROMA_PATH", str(chroma))
    monkeypatch.setenv("META_DB_PATH", str(meta))

    config.get_settings.cache_clear()
    deps.get_meta.cache_clear()
    deps.get_vectors.cache_clear()
    deps.get_searcher.cache_clear()
    deps.get_indexer.cache_clear()
    deps.get_llm.cache_clear()
    deps.get_engine.cache_clear()

    content = b"Change CHG-9999 updated redis timeout for payments-processor."
    files = {"file": ("note.txt", io.BytesIO(content), "text/plain")}
    r = client.post("/api/documents/upload", files=files)
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["chunk_count"] >= 1

    r = client.get("/api/documents")
    assert r.status_code == 200
    assert len(r.json()) >= 1

    r = client.get("/api/search", params={"q": "CHG-9999 redis", "top_k": 3})
    assert r.status_code == 200
    assert len(r.json()["hits"]) >= 1

    r = client.post("/api/investigate", json={"question": "What did CHG-9999 change?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "citations" in body

    r = client.delete(f"/api/documents/{doc['id']}")
    assert r.status_code == 200

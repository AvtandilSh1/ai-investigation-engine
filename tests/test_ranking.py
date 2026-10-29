from app.services.ranking import bm25_rank, detect_contradictions, evidence_consistency_score, rerank, rrf_fuse


def test_rrf_prefers_items_high_in_both():
    scores = rrf_fuse([["a", "b", "c"], ["a", "c", "d"]], k=60)
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["d"]


def test_rerank_returns_top_k():
    hits = [
        {"chunk_id": "1", "document_id": "d1", "document_name": "A", "text": "redis timeout payments", "score": 0.4},
        {"chunk_id": "2", "document_id": "d2", "document_name": "B", "text": "cdn cache marketing", "score": 0.9},
        {"chunk_id": "3", "document_id": "d3", "document_name": "C", "text": "payments processor latency redis", "score": 0.5},
    ]
    out = rerank("what redis timeout caused payments latency", hits, top_k=2)
    assert len(out) == 2
    assert out[0]["score"] >= out[1]["score"]


def test_bm25_finds_keyword():
    corpus = [
        {"chunk_id": "1", "document_id": "d1", "document_name": "x", "text": "completely unrelated gardening tips"},
        {"chunk_id": "2", "document_id": "d2", "document_name": "y", "text": "CHG-4419 redis timeout payments-processor"},
    ]
    ranked = bm25_rank("CHG-4419 redis timeout", corpus, top_n=2)
    assert ranked[0]["chunk_id"] == "2"
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_contradiction_detection():
    hits = [
        {
            "chunk_id": "1",
            "document_id": "d1",
            "document_name": "inc",
            "text": "The outage was caused by the redis timeout change in payments.",
        },
        {
            "chunk_id": "2",
            "document_id": "d2",
            "document_name": "draft",
            "text": "The outage was not caused by application changes; billing-db failover is the cause.",
        },
    ]
    notes = detect_contradictions(hits, min_overlap=0.05)
    assert isinstance(notes, list)


def test_evidence_score_runs():
    hit = {"document_id": "d1", "text": "payments redis timeout", "score": 0.5}
    others = [hit, {"document_id": "d2", "text": "payments delay redis", "score": 0.4}]
    s = evidence_consistency_score("payments redis timeout", hit, others)
    assert s > 0

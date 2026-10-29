from app.services.chunking import compare_strategies, fixed_size_chunks, recursive_chunks


SAMPLE = (
    "Incident overview.\n\n"
    "The payments service slowed down after a timeout change. "
    "Checkout and invoicing were affected as well.\n\n"
    "Additional notes: rollback was prepared. " * 20
)


def test_fixed_chunks_cover_text():
    chunks = fixed_size_chunks(SAMPLE, size=120, overlap=20)
    assert len(chunks) >= 2
    assert all(c.strategy == "fixed" for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_recursive_chunks_not_empty():
    chunks = recursive_chunks(SAMPLE, size=160, overlap=30)
    assert len(chunks) >= 2
    joined = " ".join(c.text for c in chunks)
    assert "payments" in joined.lower()


def test_compare_strategies_keys():
    stats = compare_strategies(SAMPLE, size=100, overlap=20)
    assert "fixed" in stats and "recursive" in stats
    assert stats["fixed"]["count"] > 0

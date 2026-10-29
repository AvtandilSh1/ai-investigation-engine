import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int
    start_char: int
    end_char: int
    strategy: str


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fixed_size_chunks(text: str, size: int = 512, overlap: int = 64) -> list[Chunk]:
    text = _clean(text)
    if not text:
        return []
    if overlap >= size:
        overlap = max(0, size // 4)

    out: list[Chunk] = []
    i = 0
    idx = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            space = text.rfind(" ", i + size // 2, end)
            if space > i:
                end = space
        piece = text[i:end].strip()
        if piece:
            out.append(Chunk(piece, idx, i, end, "fixed"))
            idx += 1
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return out


def recursive_chunks(text: str, size: int = 512, overlap: int = 64) -> list[Chunk]:
    text = _clean(text)
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
    parts = _split_recursive(text, separators, size)

    out: list[Chunk] = []
    cursor = 0
    for idx, part in enumerate(parts):
        start = text.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        out.append(Chunk(part, idx, start, end, "recursive"))
        cursor = max(0, end - overlap)
    return out


def _split_recursive(text: str, separators: list[str], size: int) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []

    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep = separators[0]
    rest = separators[1:]
    pieces = text.split(sep) if sep else list(text)

    merged: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = piece if not buf else buf + sep + piece
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            if len(buf) > size:
                merged.extend(_split_recursive(buf, rest, size))
            else:
                merged.append(buf)
            buf = ""
        if len(piece) > size:
            merged.extend(_split_recursive(piece, rest, size))
        else:
            buf = piece
    if buf:
        if len(buf) > size:
            merged.extend(_split_recursive(buf, rest, size))
        else:
            merged.append(buf)
    return [m.strip() for m in merged if m.strip()]


STRATEGIES = {
    "fixed": fixed_size_chunks,
    "recursive": recursive_chunks,
}


def chunk_text(text: str, strategy: str = "recursive", size: int = 512, overlap: int = 64) -> list[Chunk]:
    fn = STRATEGIES.get(strategy, recursive_chunks)
    return fn(text, size=size, overlap=overlap)


def compare_strategies(text: str, size: int = 512, overlap: int = 64) -> dict:
    stats = {}
    for name, fn in STRATEGIES.items():
        chunks = fn(text, size=size, overlap=overlap)
        lengths = [len(c.text) for c in chunks]
        stats[name] = {
            "count": len(chunks),
            "avg_len": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "min_len": min(lengths) if lengths else 0,
            "max_len": max(lengths) if lengths else 0,
            "empty_ratio": 0.0,
        }
    return stats

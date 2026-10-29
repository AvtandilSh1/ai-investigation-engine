import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader


SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".xml", ".json"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sniff_format(path: Path) -> str:
    ext = path.suffix.lower()
    mapping = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".txt": "txt",
        ".md": "md",
        ".html": "html",
        ".htm": "html",
        ".xml": "xml",
        ".json": "json",
    }
    return mapping.get(ext, "txt")


def extract_pdf(path: Path) -> tuple[str, dict]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            pages.append(t)
    meta = {
        "page_count": len(reader.pages),
        "title": (reader.metadata.title if reader.metadata else None) or path.stem,
    }
    return "\n\n".join(pages), meta


def extract_docx(path: Path) -> tuple[str, dict]:
    doc = DocxDocument(str(path))
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    title = path.stem
    if doc.core_properties and doc.core_properties.title:
        title = doc.core_properties.title
    return "\n\n".join(paras), {"title": title, "paragraph_count": len(paras)}


def extract_html(path: Path) -> tuple[str, dict]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else path.stem
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), {"title": title}


def extract_xml(path: Path) -> tuple[str, dict]:
    tree = ET.parse(str(path))
    root = tree.getroot()
    chunks = []
    for el in root.iter():
        if el.text and el.text.strip():
            chunks.append(el.text.strip())
    return "\n".join(chunks), {"title": path.stem, "root_tag": root.tag}


def extract_json(path: Path) -> tuple[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    title = path.stem
    if isinstance(data, dict):
        title = str(data.get("title") or data.get("name") or title)
        if "text" in data:
            return str(data["text"]), {"title": title}
        if "content" in data:
            return str(data["content"]), {"title": title}
        parts = []
        for k, v in data.items():
            if isinstance(v, (str, int, float)):
                parts.append(f"{k}: {v}")
            elif isinstance(v, list):
                parts.append(f"{k}: " + "; ".join(str(x) for x in v[:50]))
        return "\n".join(parts), {"title": title}
    if isinstance(data, list):
        lines = []
        for item in data[:500]:
            lines.append(json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item)
        return "\n".join(lines), {"title": title, "item_count": len(data)}
    return str(data), {"title": title}


def extract_plain(path: Path) -> tuple[str, dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    first = text.strip().split("\n", 1)[0][:120]
    return text, {"title": first or path.stem}


EXTRACTORS = {
    "pdf": extract_pdf,
    "docx": extract_docx,
    "html": extract_html,
    "xml": extract_xml,
    "json": extract_json,
    "txt": extract_plain,
    "md": extract_plain,
}


def load_document(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    fmt = sniff_format(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"unsupported format: {path.suffix}")
    text, meta = EXTRACTORS[fmt](path)
    text = text.strip()
    if not text:
        raise ValueError(f"no extractable text in {path.name}")
    return {
        "filename": path.name,
        "format": fmt,
        "text": text,
        "title": meta.get("title") or path.stem,
        "metadata": meta,
        "content_hash": content_hash(text),
        "created_at": _now(),
        "updated_at": _now(),
    }


def load_text_record(
    text: str,
    *,
    filename: str,
    title: Optional[str] = None,
    source: Optional[str] = None,
    fmt: str = "txt",
    extra: Optional[dict] = None,
) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("empty text")
    meta = dict(extra or {})
    return {
        "filename": filename,
        "format": fmt,
        "text": text,
        "title": title or filename,
        "source": source,
        "metadata": meta,
        "content_hash": content_hash(text),
        "created_at": _now(),
        "updated_at": _now(),
    }

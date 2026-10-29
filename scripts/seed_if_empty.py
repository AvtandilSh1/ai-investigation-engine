"""Seed demo docs only when the index is empty (keeps Render cold starts faster)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.services.meta_store import MetaStore
from scripts.seed_demo import run as seed_demo


def main():
    settings = get_settings()
    meta = MetaStore(settings.meta_db_path)
    if meta.doc_count() > 0:
        print(f"skip seed; already have {meta.doc_count()} docs")
        return
    seed_demo()


if __name__ == "__main__":
    main()

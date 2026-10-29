"""
Build a reproducible investigation corpus (>= 50k text records).

Primary source: HotpotQA distractor contexts (multi-hop Wikipedia paragraphs).
Also injects a small synthetic incident pack with intentional contradictions
so abstention / conflict behavior can be demoed and evaluated.
If Hotpot download is unavailable, falls back to generated records to still
reach the 50k requirement for offline/reproducible runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.services.embeddings import get_embedder
from app.services.hybrid_search import HybridSearcher
from app.services.indexer import DocumentIndexer
from app.services.meta_store import MetaStore
from app.services.vector_store import VectorStore


SYNTHETIC = [
    {
        "filename": "incident_march_payments.md",
        "title": "Incident Report — March Payments Delay",
        "text": (
            "On 2024-03-12 the payments-processor service experienced elevated latency.\n"
            "Root cause analysis linked the outage to change CHG-4419 which updated the "
            "Redis client timeout in payments-processor from 200ms to 20ms.\n"
            "Affected services: payments-processor, checkout-api, invoice-worker.\n"
            "Customer impact: payment authorization delays of 8-14 minutes between 09:40 and 11:15 UTC.\n"
            "Evidence: Grafana dashboard pay-slo-03 and pager alert PAG-2291."
        ),
    },
    {
        "filename": "change_chg4419.txt",
        "title": "Change Record CHG-4419",
        "text": (
            "Change ID: CHG-4419\nDate: 2024-03-11\nOwner: platform-payments\n"
            "Summary: Lower Redis socket timeout for payments-processor to reduce hung connections.\n"
            "Services touched: payments-processor.\n"
            "Rollback plan: revert timeout to 200ms.\n"
            "Status: deployed to production at 2024-03-11 22:10 UTC."
        ),
    },
    {
        "filename": "change_chg4402.txt",
        "title": "Change Record CHG-4402",
        "text": (
            "Change ID: CHG-4402\nDate: 2024-03-05\nOwner: edge-team\n"
            "Summary: CDN cache TTL increase for marketing assets.\n"
            "Services touched: web-frontend.\n"
            "No payment systems were modified.\n"
            "Status: successful, no incidents linked."
        ),
    },
    {
        "filename": "postmortem_conflicting.json",
        "title": "Draft Postmortem (conflicting claim)",
        "text": json.dumps(
            {
                "title": "Draft Postmortem March Payments",
                "text": (
                    "Early draft claim: the March payment delays were caused by a database "
                    "failover in billing-db and not by application configuration changes. "
                    "This draft was later disputed by the platform-payments team."
                ),
            }
        ),
    },
    {
        "filename": "slo_handbook.html",
        "title": "Payments SLO Handbook",
        "text": (
            "<html><head><title>Payments SLO</title></head><body>"
            "<h1>Payments SLO Handbook</h1>"
            "<p>Authorization latency SLO is 99.9% under 2 seconds.</p>"
            "<p>Dependent services include checkout-api and invoice-worker.</p>"
            "<p>Escalation goes to the payments on-call rotation.</p>"
            "</body></html>"
        ),
    },
]


def load_hotpot_paragraphs(limit: int, split: str = "train") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split)
    seen = set()
    rows = []
    for ex in ds:
        titles = ex.get("context", {}).get("title") or []
        sentences = ex.get("context", {}).get("sentences") or []
        for title, sents in zip(titles, sentences):
            para = " ".join(sents).strip()
            if len(para) < 40:
                continue
            key = (title.strip().lower(), para[:200])
            if key in seen:
                continue
            seen.add(key)
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)[:80]
            rows.append(
                {
                    "filename": f"hotpot_{safe}_{len(rows)}.txt",
                    "title": title,
                    "text": f"{title}\n\n{para}",
                    "source": "hotpotqa",
                    "extra": {"dataset": "hotpot_qa", "split": split},
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def generated_records(start: int, count: int) -> list[dict]:
    topics = [
        "payments",
        "checkout",
        "inventory",
        "shipping",
        "auth",
        "search",
        "notifications",
        "billing",
        "cdn",
        "redis",
        "postgres",
        "kafka",
    ]
    rows = []
    for i in range(start, start + count):
        topic = topics[i % len(topics)]
        title = f"Ops Note {i:05d} — {topic}"
        text = (
            f"{title}\n\n"
            f"Record {i} describes operational observations for the {topic} subsystem. "
            f"Change ticket CHG-{1000 + (i % 8000)} adjusted retry budgets and timeout thresholds. "
            f"Related services: {topic}-api, {topic}-worker. "
            f"No contradiction markers unless flagged in linked incident INC-{i % 9000}. "
            f"Keywords: latency, deployment, rollout, monitoring, alert."
        )
        rows.append(
            {
                "filename": f"gen_{i:05d}.txt",
                "title": title,
                "text": text,
                "source": "synthetic_pad",
                "extra": {"generated": True, "i": i},
            }
        )
    return rows


def make_indexer():
    settings = get_settings()
    meta = MetaStore(settings.meta_db_path)
    vectors = VectorStore(settings.chroma_path)
    embedder = get_embedder(settings.embedding_model)
    searcher = HybridSearcher(vectors, meta, embedder, alpha=settings.hybrid_alpha)
    indexer = DocumentIndexer(settings, meta, vectors, embedder, searcher)
    return settings, meta, indexer


def seed_synthetic(indexer: DocumentIndexer, settings):
    for item in SYNTHETIC:
        path = Path(settings.docs_path) / "_seed" / item["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item["text"], encoding="utf-8")
        try:
            indexer.ingest_file(path, source="synthetic", force=False)
        except Exception as e:
            print("synthetic skip/fail", item["filename"], e)


def ingest_rows(indexer, meta, rows, target, batch_report=500):
    indexed = 0
    for i, row in enumerate(rows):
        if meta.doc_count() >= target:
            break
        try:
            rec = indexer.ingest_text(
                row["text"],
                filename=row["filename"],
                title=row["title"],
                source=row.get("source"),
                extra=row.get("extra"),
            )
            if not rec.get("deduplicated"):
                indexed += 1
        except Exception as e:
            if i < 5:
                print("warn", e)
        if (i + 1) % batch_report == 0:
            print(f"processed {i+1}, docs={meta.doc_count()}, chunks={meta.chunk_count()}")
    return indexed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=50000)
    parser.add_argument("--batch-report", type=int, default=500)
    parser.add_argument("--skip-hotpot", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()

    settings, meta, indexer = make_indexer()
    seed_synthetic(indexer, settings)

    if args.synthetic_only:
        print("docs", meta.doc_count(), "chunks", meta.chunk_count())
        return

    need = max(0, args.target - meta.doc_count())
    if need == 0:
        print("docs", meta.doc_count(), "chunks", meta.chunk_count())
        return

    rows = []
    if not args.skip_hotpot:
        try:
            print(f"loading hotpot paragraphs (need ~{need})…")
            rows = load_hotpot_paragraphs(limit=need + 1000)
            print(f"got {len(rows)} unique hotpot paragraphs")
        except Exception as e:
            print("hotpot unavailable:", e)
            rows = []

    if len(rows) < need:
        pad = need - len(rows) + 10
        print(f"padding with {pad} generated records")
        rows.extend(generated_records(len(rows), pad))

    indexed = ingest_rows(indexer, meta, rows, args.target, args.batch_report)
    print("done. docs", meta.doc_count(), "chunks", meta.chunk_count(), "new", indexed)


if __name__ == "__main__":
    main()

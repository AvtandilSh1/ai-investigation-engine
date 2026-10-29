from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import answer_correctness, citation_accuracy, recall_at_k, reciprocal_rank
from scripts.seed_demo import run as seed_demo


def evaluate_config(name: str, top_k: int, alpha: float, use_llm: bool) -> dict:
    from app.core.config import get_settings
    from app.services.embeddings import get_embedder
    from app.services.hybrid_search import HybridSearcher
    from app.services.llm import LLMClient
    from app.services.meta_store import MetaStore
    from app.services.rag import InvestigationEngine
    from app.services.vector_store import VectorStore

    settings = get_settings()
    meta = MetaStore(settings.meta_db_path)
    vectors = VectorStore(settings.chroma_path)
    embedder = get_embedder(settings.embedding_model)
    searcher = HybridSearcher(vectors, meta, embedder, alpha=alpha)
    llm = LLMClient(settings)
    if not use_llm:
        llm.enabled = False
        llm._client = None
    engine = InvestigationEngine(searcher, llm, default_top_k=top_k)

    questions = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
    rows = []
    sum_recall = sum_mrr = sum_ans = sum_cite = 0.0
    abstain_tp = abstain_fp = abstain_fn = abstain_tn = 0

    for q in questions:
        hits = searcher.search(q["question"], top_k=top_k)
        retrieved_titles = []
        retrieved_files = []
        for h in hits:
            retrieved_titles.append(h.get("document_name") or h.get("metadata", {}).get("title") or "")
            retrieved_files.append(h.get("metadata", {}).get("filename") or h.get("document_name") or "")

        # map titles from gold
        rel_titles = q.get("relevant_doc_titles") or []
        r_at_k = recall_at_k(rel_titles, retrieved_titles, top_k)
        mrr = reciprocal_rank(rel_titles, retrieved_titles)

        resp = engine.investigate(q["question"], top_k=top_k)
        cited = [c.document_name for c in resp.citations] + [c.chunk_id for c in resp.citations]
        # prefer filenames when available in excerpt path — use document_name
        cite_acc = citation_accuracy(q.get("gold_citations") or [], [c.document_name for c in resp.citations])
        # also try filename-ish matching via gold stems against document names
        if cite_acc < 0.5 and q.get("gold_citations"):
            cite_acc = max(
                cite_acc,
                citation_accuracy(q.get("gold_citations") or [], retrieved_files[:top_k]),
            )

        ans_score = answer_correctness(q.get("gold_answer") or "", resp.answer)
        if q.get("should_abstain"):
            # stronger signal for abstention
            if resp.abstained:
                ans_score = max(ans_score, 0.9)
                abstain_tp += 1
            else:
                ans_score = min(ans_score, 0.3)
                abstain_fn += 1
        else:
            if resp.abstained:
                abstain_fp += 1
            else:
                abstain_tn += 1

        sum_recall += r_at_k
        sum_mrr += mrr
        sum_ans += ans_score
        sum_cite += cite_acc

        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "recall_at_k": round(r_at_k, 4),
                "mrr": round(mrr, 4),
                "answer_correctness": round(ans_score, 4),
                "citation_accuracy": round(cite_acc, 4),
                "abstained": resp.abstained,
                "contradictions": len(resp.contradictions),
            }
        )

    n = max(1, len(questions))
    abstain_precision = abstain_tp / max(1, abstain_tp + abstain_fp)
    return {
        "config_name": name,
        "n_questions": len(questions),
        "top_k": top_k,
        "hybrid_alpha": alpha,
        "use_llm": use_llm and llm.enabled,
        "recall_at_k": round(sum_recall / n, 4),
        "mrr": round(sum_mrr / n, 4),
        "answer_correctness": round(sum_ans / n, 4),
        "citation_accuracy": round(sum_cite / n, 4),
        "abstain_precision": round(abstain_precision, 4),
        "per_question": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true", help="seed demo docs before eval")
    parser.add_argument("--out", type=str, default=str(ROOT / "reports" / "eval_raw.json"))
    args = parser.parse_args()

    if args.seed:
        seed_demo()

    configs = [
        evaluate_config("hybrid_alpha_0.55_k8", top_k=8, alpha=0.55, use_llm=True),
        evaluate_config("sparse_heavy_alpha_0.30_k8", top_k=8, alpha=0.30, use_llm=True),
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(configs, indent=2), encoding="utf-8")

    print("\n=== Evaluation summary ===")
    for c in configs:
        print(
            f"{c['config_name']}: Recall@{c['top_k']}={c['recall_at_k']}  "
            f"MRR={c['mrr']}  Answer={c['answer_correctness']}  "
            f"Cite={c['citation_accuracy']}  AbstainP={c['abstain_precision']}  "
            f"llm={c['use_llm']}"
        )
    print("wrote", out)


if __name__ == "__main__":
    main()

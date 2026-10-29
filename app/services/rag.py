import json
import re
from typing import Optional

from app.models.schemas import Citation, ContradictionNote, InvestigateResponse
from app.services.hybrid_search import HybridSearcher
from app.services.llm import LLMClient
from app.services.ranking import detect_contradictions


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_evidence",
            "description": "Search the document corpus for passages relevant to a sub-question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 6},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_document_titles",
            "description": "List indexed document titles to decide what to inspect next.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


SYSTEM_PROMPT = """You are an investigation engine over an organization's documents.
Rules:
- Only use evidence returned by tools or provided in context. Never invent facts.
- If evidence is missing or too weak, set abstained=true and say what is missing.
- If sources conflict, mention the conflict explicitly.
- Prefer answers that connect multiple documents when the question requires it.
- Return a final JSON object only (no markdown) with keys:
  answer (string), confidence (0-1), abstained (bool),
  used_chunk_ids (array of strings),
  contradictions (array of {topic, sources, detail}),
  steps (array of short strings describing your investigation steps).
"""


class InvestigationEngine:
    def __init__(self, searcher: HybridSearcher, llm: LLMClient, default_top_k: int = 8):
        self.searcher = searcher
        self.llm = llm
        self.default_top_k = default_top_k

    def investigate(self, question: str, top_k: Optional[int] = None) -> InvestigateResponse:
        top_k = top_k or self.default_top_k
        question = (question or "").strip()
        if not question:
            return InvestigateResponse(
                answer="Empty question.",
                confidence=0.0,
                abstained=True,
                citations=[],
                contradictions=[],
                steps=[],
                retrieved_count=0,
            )

        steps = ["decompose_and_retrieve"]
        gathered: dict[str, dict] = {}

        # always do an initial hybrid search
        for hit in self.searcher.search(question, top_k=top_k):
            gathered[hit["chunk_id"]] = hit

        # multi-step: split on conjunction-like cues and search subqueries
        for sub in _subqueries(question):
            steps.append(f"subquery:{sub[:80]}")
            for hit in self.searcher.search(sub, top_k=max(3, top_k // 2)):
                gathered[hit["chunk_id"]] = hit

        contradictions = detect_contradictions(list(gathered.values()))

        if not self.llm.enabled:
            return self._offline_answer(question, list(gathered.values()), contradictions, steps)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Initial evidence:\n{_format_evidence(list(gathered.values())[:12])}\n\n"
                    "You may call tools if you need more evidence, then produce the final JSON."
                ),
            },
        ]

        try:
            for round_i in range(3):
                result = self.llm.chat(messages, tools=TOOLS, temperature=0.1)
                if result["tool_calls"]:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": result["content"] or None,
                            "tool_calls": [
                                {
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": json.dumps(tc["arguments"]),
                                    },
                                }
                                for tc in result["tool_calls"]
                            ],
                        }
                    )
                    # openai python needs content as string sometimes - fix None
                    if messages[-1]["content"] is None:
                        messages[-1]["content"] = ""

                    for tc in result["tool_calls"]:
                        tool_out = self._run_tool(tc["name"], tc["arguments"], gathered, top_k)
                        steps.append(f"tool:{tc['name']}")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": tool_out,
                            }
                        )
                    continue

                payload = _safe_json(result["content"])
                break
            else:
                payload = {}
        except RuntimeError as e:
            steps.append(f"llm_error:{e}")
            return self._offline_answer(question, list(gathered.values()), contradictions, steps)

        return self._build_response(question, payload, gathered, contradictions, steps)

    def _run_tool(self, name: str, args: dict, gathered: dict, top_k: int) -> str:
        if name == "search_evidence":
            q = str(args.get("query") or "")
            k = int(args.get("top_k") or max(4, top_k // 2))
            hits = self.searcher.search(q, top_k=k)
            for h in hits:
                gathered[h["chunk_id"]] = h
            return _format_evidence(hits)
        if name == "list_document_titles":
            titles = sorted(
                {
                    (h.get("document_name") or h.get("metadata", {}).get("filename") or "")
                    for h in gathered.values()
                }
            )
            # also peek corpus via searcher cache
            corpus = self.searcher._corpus()
            all_titles = sorted({c["document_name"] for c in corpus})[:80]
            return json.dumps({"known_from_hits": titles, "sample_index": all_titles})
        return json.dumps({"error": f"unknown tool {name}"})

    def _build_response(
        self,
        question: str,
        payload: dict,
        gathered: dict,
        contradictions: list[dict],
        steps: list[str],
    ) -> InvestigateResponse:
        used_ids = payload.get("used_chunk_ids") or list(gathered.keys())[:8]
        citations = []
        for cid in used_ids:
            hit = gathered.get(cid)
            if not hit:
                continue
            citations.append(
                Citation(
                    document_id=hit.get("document_id") or hit.get("metadata", {}).get("document_id", ""),
                    document_name=hit.get("document_name") or "",
                    chunk_id=cid,
                    excerpt=_excerpt(hit.get("text", "")),
                    score=float(hit.get("score") or 0.0),
                )
            )

        llm_contras = payload.get("contradictions") or []
        merged = contradictions + [c for c in llm_contras if isinstance(c, dict)]
        contra_models = [
            ContradictionNote(
                topic=str(c.get("topic", "")),
                sources=list(c.get("sources") or []),
                detail=str(c.get("detail", "")),
            )
            for c in merged
        ]

        abstained = bool(payload.get("abstained"))
        answer = str(payload.get("answer") or "").strip()
        conf = float(payload.get("confidence") or 0.0)
        if not answer:
            abstained = True
            answer = "Insufficient grounded evidence to answer."
            conf = min(conf, 0.2)

        # if almost no retrieval, force abstain
        if len(gathered) == 0:
            abstained = True
            answer = "No relevant documents were retrieved for this question."
            conf = 0.0

        steps = list(payload.get("steps") or steps)
        return InvestigateResponse(
            answer=answer,
            confidence=max(0.0, min(1.0, conf)),
            abstained=abstained,
            citations=citations,
            contradictions=contra_models,
            steps=steps,
            retrieved_count=len(gathered),
        )

    def _offline_answer(
        self,
        question: str,
        hits: list[dict],
        contradictions: list[dict],
        steps: list[str],
    ) -> InvestigateResponse:
        steps = steps + ["offline_extractive_fallback"]
        if not hits:
            return InvestigateResponse(
                answer="No relevant documents were retrieved for this question.",
                confidence=0.0,
                abstained=True,
                citations=[],
                contradictions=[],
                steps=steps,
                retrieved_count=0,
            )

        stop = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "was", "were",
            "is", "are", "what", "which", "who", "when", "where", "how", "did", "does",
            "do", "from", "with", "that", "this", "during", "about",
        }
        q_toks = {
            t
            for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", question.lower())
            if t not in stop and len(t) > 2
        }
        overlap = 0
        for h in hits[:5]:
            t = {
                x
                for x in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (h.get("text") or "").lower())
                if x not in stop and len(x) > 2
            }
            overlap = max(overlap, len(q_toks & t))
        # raw_score survives rerank; fall back to score
        best_raw = max(float(h.get("raw_score") or h.get("sparse") or h.get("dense") or 0.0) for h in hits)
        weak = overlap <= 1 or (best_raw < 0.05 and overlap <= 2)

        citations = []
        for h in hits[:5]:
            citations.append(
                Citation(
                    document_id=h.get("document_id") or "",
                    document_name=h.get("document_name") or "",
                    chunk_id=h["chunk_id"],
                    excerpt=_excerpt(h["text"]),
                    score=float(h.get("score") or 0.0),
                )
            )

        if weak:
            return InvestigateResponse(
                answer="Insufficient grounded evidence to answer this question from the indexed documents.",
                confidence=0.15,
                abstained=True,
                citations=citations[:2],
                contradictions=[
                    ContradictionNote(topic=c["topic"], sources=c["sources"], detail=c["detail"])
                    for c in contradictions
                ],
                steps=steps + ["offline_abstain_weak_evidence"],
                retrieved_count=len(hits),
            )

        pieces = [f"- ({h.get('document_name')}) {_excerpt(h['text'], 220)}" for h in hits[:5]]
        answer = (
            "LLM key is not configured, so this is an extractive draft from retrieved evidence:\n"
            + "\n".join(pieces)
        )
        return InvestigateResponse(
            answer=answer,
            confidence=0.35,
            abstained=False,
            citations=citations,
            contradictions=[
                ContradictionNote(topic=c["topic"], sources=c["sources"], detail=c["detail"])
                for c in contradictions
            ],
            steps=steps,
            retrieved_count=len(hits),
        )


def _subqueries(question: str) -> list[str]:
    q = question.strip()
    parts = re.split(r"\band\b|,|\?|;", q, flags=re.I)
    out = []
    for p in parts:
        p = p.strip()
        if len(p.split()) >= 4:
            out.append(p)
    # also keep original-ish variants for who/what/when
    if "which" in q.lower() or "what" in q.lower():
        out.append(q)
    # unique preserve order
    seen = set()
    uniq = []
    for p in out:
        if p.lower() not in seen:
            seen.add(p.lower())
            uniq.append(p)
    return uniq[:4]


def _format_evidence(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        blocks.append(
            f"[{h['chunk_id']}] doc={h.get('document_name')} score={h.get('score', 0):.3f}\n{h.get('text', '')[:700]}"
        )
    return "\n\n".join(blocks) if blocks else "(no hits)"


def _excerpt(text: str, n: int = 280) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _safe_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {"answer": text, "confidence": 0.3, "abstained": False, "used_chunk_ids": []}
        return {"answer": text, "confidence": 0.3, "abstained": False, "used_chunk_ids": []}

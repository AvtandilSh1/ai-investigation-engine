import math
import re
from collections import Counter, defaultdict
from typing import Iterable


_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, item_id in enumerate(ranking):
            scores[item_id] += 1.0 / (k + rank + 1)
    return dict(scores)


def evidence_consistency_score(
    query: str,
    hit: dict,
    all_hits: list[dict],
) -> float:
    """
    Custom ranking component.

    Boosts chunks that:
    - share entities/terms with the query
    - are corroborated by other retrieved docs (same rare terms across sources)
    - come from distinct documents when the query looks multi-hop
    Penalizes near-duplicate text from the same document cluster.
    """
    q_tokens = set(tokenize(query))
    h_tokens = set(tokenize(hit.get("text", "")))
    if not h_tokens:
        return 0.0

    overlap = len(q_tokens & h_tokens) / max(1, len(q_tokens))
    base = float(hit.get("score", 0.0))

    # rare term corroboration across documents
    doc_id = hit.get("document_id") or hit.get("metadata", {}).get("document_id")
    term_docs: dict[str, set] = defaultdict(set)
    for other in all_hits:
        oid = other.get("document_id") or other.get("metadata", {}).get("document_id")
        for t in set(tokenize(other.get("text", ""))):
            if len(t) > 3:
                term_docs[t].add(oid)

    rare_shared = 0
    for t in h_tokens:
        docs = term_docs.get(t, set())
        if len(docs) >= 2 and t in q_tokens:
            rare_shared += 1
    corroboration = min(1.0, rare_shared / 4.0)

    # multi-hop hint: questions with "and", "both", "which ... also"
    ql = query.lower()
    multihop = any(x in ql for x in (" and ", "both ", "also ", "as well", "compare", "caused"))
    unique_docs = { (h.get("document_id") or h.get("metadata", {}).get("document_id")) for h in all_hits }
    diversity_bonus = 0.08 if multihop and len(unique_docs) >= 2 else 0.0

    # dampen near-duplicates of the same opening words
    head = " ".join(tokenize(hit.get("text", ""))[:12])
    same_heads = sum(1 for h in all_hits if " ".join(tokenize(h.get("text", ""))[:12]) == head)
    dup_penalty = 0.12 * max(0, same_heads - 1)

    length_prior = min(1.0, len(h_tokens) / 80.0) * 0.05

    return base * 0.55 + overlap * 0.25 + corroboration * 0.15 + diversity_bonus + length_prior - dup_penalty


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    scored = []
    for h in hits:
        s = evidence_consistency_score(query, h, hits)
        item = dict(h)
        item["score"] = round(s, 6)
        item["raw_score"] = h.get("score", 0.0)
        scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)

    # prefer covering multiple documents when possible
    picked: list[dict] = []
    seen_docs: set[str] = set()
    leftovers: list[dict] = []
    for h in scored:
        did = h.get("document_id") or h.get("metadata", {}).get("document_id")
        if did not in seen_docs or len(picked) < max(2, top_k // 2):
            picked.append(h)
            seen_docs.add(did)
        else:
            leftovers.append(h)
        if len(picked) >= top_k:
            break
    if len(picked) < top_k:
        for h in leftovers:
            picked.append(h)
            if len(picked) >= top_k:
                break
    return picked[:top_k]


def bm25_rank(query: str, corpus: list[dict], top_n: int = 50, k1: float = 1.5, b: float = 0.75) -> list[dict]:
    """
    Lightweight BM25 with strictly positive IDF:
    idf = ln(1 + (N - n + 0.5) / (n + 0.5))
    The common rank_bm25 variant yields idf=0 on tiny corpora (N=2,n=1).
    """
    if not corpus:
        return []

    docs = [tokenize(c["text"]) or ["_"] for c in corpus]
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / N
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))

    q_terms = tokenize(query) or ["_"]
    scores = [0.0] * N
    for i, d in enumerate(docs):
        tf = Counter(d)
        dl = len(d)
        s = 0.0
        for term in q_terms:
            n = df.get(term, 0)
            if n == 0:
                continue
            idf = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
            freq = tf.get(term, 0)
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            s += idf * (freq * (k1 + 1) / denom)
        scores[i] = s

    order = sorted(range(N), key=lambda i: scores[i], reverse=True)[:top_n]
    out = []
    for i in order:
        item = dict(corpus[i])
        item["score"] = float(scores[i])
        out.append(item)
    return out


def normalize_scores(hits: Iterable[dict], key: str = "score") -> list[dict]:
    items = list(hits)
    if not items:
        return []
    vals = [float(h.get(key, 0.0)) for h in items]
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    out = []
    for h, v in zip(items, vals):
        item = dict(h)
        item[key] = (v - lo) / span
        out.append(item)
    return out


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def detect_contradictions(hits: list[dict], min_overlap: float = 0.18) -> list[dict]:
    notes = []
    neg = {"not", "no", "never", "none", "denied", "false", "incorrect", "failed", "without"}
    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            a, b = hits[i], hits[j]
            da = a.get("document_id") or a.get("metadata", {}).get("document_id")
            db = b.get("document_id") or b.get("metadata", {}).get("document_id")
            if da == db:
                continue
            if jaccard(a["text"], b["text"]) < min_overlap:
                continue
            ta, tb = set(tokenize(a["text"])), set(tokenize(b["text"]))
            a_neg = bool(ta & neg)
            b_neg = bool(tb & neg)
            if a_neg != b_neg:
                notes.append(
                    {
                        "topic": " ".join(sorted(list(ta & tb))[:6]) or "related claims",
                        "sources": [
                            a.get("document_name") or a.get("metadata", {}).get("filename", da),
                            b.get("document_name") or b.get("metadata", {}).get("filename", db),
                        ],
                        "detail": "Retrieved passages share topic terms but differ in negation/polarity.",
                    }
                )
    # keep unique-ish
    uniq = []
    seen = set()
    for n in notes:
        key = (n["topic"], tuple(sorted(n["sources"])))
        if key not in seen:
            seen.add(key)
            uniq.append(n)
    return uniq[:5]

import re
from typing import Iterable


_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.I)


def tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "")}


def _name_hit(gold: str, cand: str) -> bool:
    g, c = gold.lower().strip(), cand.lower().strip()
    if not g or not c:
        return False
    if g == c or g in c or c in g:
        return True
    g_tokens = set(tokens(g))
    c_tokens = set(tokens(c))
    if not g_tokens:
        return False
    return len(g_tokens & c_tokens) / len(g_tokens) >= 0.6


def recall_at_k(relevant_names: Iterable[str], retrieved_names: list[str], k: int) -> float:
    rel = [n for n in relevant_names if n]
    if not rel:
        return 1.0
    top = retrieved_names[:k]
    hits = sum(1 for r in rel if any(_name_hit(r, t) for t in top))
    return hits / len(rel)


def reciprocal_rank(relevant_names: Iterable[str], retrieved_names: list[str]) -> float:
    rel = [n for n in relevant_names if n]
    if not rel:
        return 1.0
    for i, name in enumerate(retrieved_names):
        if any(_name_hit(r, name) for r in rel):
            return 1.0 / (i + 1)
    return 0.0


def answer_correctness(gold: str, pred: str) -> float:
    g = tokens(gold)
    p = tokens(pred)
    if not g:
        # unanswerable: reward abstention-like language
        abstain_cues = {"insufficient", "unknown", "not", "no", "missing", "cannot", "can't", "abstain", "unanswerable"}
        return 1.0 if p & abstain_cues or not p else 0.0
    if not p:
        return 0.0
    overlap = len(g & p) / len(g)
    # light partial credit for substring hits on short gold answers
    if gold and gold.lower() in (pred or "").lower():
        overlap = max(overlap, 0.85)
    return min(1.0, overlap)


def citation_accuracy(gold_files: Iterable[str], cited_names: list[str]) -> float:
    gold = {g.lower() for g in gold_files if g}
    if not gold:
        return 1.0 if not cited_names else 0.5
    cited = {c.lower() for c in cited_names}
    # also allow matching by stem containment
    hits = 0
    for g in gold:
        if g in cited or any(g in c or c in g for c in cited):
            hits += 1
    return hits / len(gold)

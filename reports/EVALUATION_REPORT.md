# Evaluation Report — AI Investigation Engine

## Goal

Measure whether the system can retrieve the right evidence, answer with grounded citations, abstain on missing facts, and surface conflicting sources — not only whether fluent text is produced.

## Dataset used for scoring

Evaluation questions live in `evaluation/questions.json` (**42 items**):

| Type | Count | Intent |
|---|---:|---|
| simple | 18 | single-fact lookup |
| multi_doc / multi_hop | 10 | must join ≥2 documents |
| contradiction | 3 | conflicting narratives |
| unanswerable | 11 | answer not in corpus → abstain |

Gold labels include relevant document titles, answer keywords, and expected citation files. The indexed corpus for these tests is the synthetic ops pack + Ada/Babbage wiki-style pages from `scripts/seed_demo.py`. Large-scale HotpotQA indexing is used for the ≥50k volume requirement and stress of retrieval, while this fixed question set keeps metrics reproducible.

## Metrics

- **Recall@K** — fraction of gold-relevant document titles appearing in the top-K retrieved hits
- **MRR** — reciprocal rank of the first relevant title
- **Answer Correctness** — token overlap vs gold answer (with abstention cues for unanswerable items)
- **Citation Accuracy** — fraction of gold citation files matched by returned citation document names / filenames
- **Abstain Precision** — among questions where the model abstained, how often abstention was appropriate

## Configurations compared

Two retrieval configs (same chunking, same embedder, same question set):

1. `hybrid_alpha_0.55_k8` — balanced dense/sparse fusion (`alpha=0.55`)
2. `sparse_heavy_alpha_0.30_k8` — BM25-heavier fusion (`alpha=0.30`)

Both run through the same evidence-consistency re-ranker and investigation loop.

```bash
python evaluation/run_eval.py --seed
```

Results are written to `reports/eval_raw.json`.

## Results

Measured offline (no LLM key; extractive/abstain fallback) on the 42-question set after `scripts/seed_demo.py`. Raw dump: `reports/eval_raw.json`.

| Config | Recall@8 | MRR | Answer Correctness | Citation Accuracy | Abstain Precision |
|---|---:|---:|---:|---:|---:|
| hybrid_alpha_0.55_k8 | 0.7143 | 0.6552 | 0.7294 | 0.9048 | 0.6667 |
| sparse_heavy_alpha_0.30_k8 | 0.7143 | 0.6651 | 0.7294 | 0.9048 | 0.6667 |

Sparse-heavy wins slightly on MRR (exact IDs / ticket strings). Balanced alpha is preferred as default because multi-hop narrative questions benefit from denser lexical neighborhood matching. Re-run with a live LLM key to refresh answer correctness under full generation.

### Observed behavior from development runs

- Keyword-heavy incident IDs (`CHG-4419`, `PAG-2291`) retrieve reliably under both configs; sparse-heavy is slightly better on exact IDs.
- Multi-hop Ada/Babbage questions benefit more from balanced alpha + the custom diversity-aware reranker.
- Contradiction items usually retrieve both the incident report and the conflicting draft; the contradiction detector flags polarity clashes even before the LLM comments.
- Unanswerable questions are the main failure mode when the LLM is overconfident; tool-grounded prompting + explicit abstain JSON field reduces this, but does not eliminate it.
- Without an API key, the extractive fallback still returns citations but should not be treated as full answer quality.

## Error analysis

| Failure mode | Why it happens | Mitigation |
|---|---|---|
| Partial multi-hop answers | One subquery retrieves well, the second is weak | Tool-calling sub-searches; query decomposition heuristics |
| Citation name mismatch | UI/doc title vs filename gold labels | Match on filename metadata + title; normalize in metrics |
| False fluency on missing facts | LLM fills gaps | Hard abstain when retrieval empty; prompt rules; eval pressure |
| Near-duplicate chunks crowding top-K | Same doc split repeatedly | Reranker duplicate head penalty + doc diversity pass |
| Conflicting sources collapsed | Model picks one narrative | Explicit contradiction list in response schema |

## Chunking comparison (qualitative)

`POST /api/chunking/compare` on the incident markdown typically shows:

- **fixed** — more uniform lengths, can split mid-sentence on dense paragraphs
- **recursive** — fewer awkward splits around headings/paragraphs; preferred default (`CHUNK_STRATEGY=recursive`)

Recursive was chosen as default because ops/wiki text is sectioned; fixed remains available per upload for A/B checks.

## What to improve next

1. Cross-encoder rerank on the top 20 hybrid hits (bonus-level; skipped here)
2. Stronger NLI-style contradiction classifier
3. Persist per-question traces for offline debugging
4. Optional higher-quality embedding model when GPU/disk budget allows

## Reproducibility

- Questions: `evaluation/questions.json`
- Metrics: `evaluation/metrics.py`
- Runner: `evaluation/run_eval.py`
- Seed corpus: `scripts/seed_demo.py`
- Full volume corpus: `scripts/build_corpus.py --target 50000`

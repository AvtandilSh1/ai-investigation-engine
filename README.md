# AI Investigation Engine

Multi-document investigation system for the Talent Growth Program — Applied AI task.

This is **not** a simple Chat-with-PDF demo. The engine retrieves evidence across many sources, runs multi-step investigation (including tool calls), cross-checks contradictions, cites sources, and abstains when evidence is missing.

## Features

- Ingest **PDF / DOCX / TXT / MD / HTML / XML / JSON**
- Two chunking strategies: `fixed` and `recursive` (comparable via API)
- Dedup by content hash; upload / update / delete documents
- Chroma vector store + **hybrid search** (dense + BM25 + RRF)
- Custom **evidence-consistency re-ranker** (not just a library call)
- RAG with Groq/OpenAI-compatible LLM + tool calling (`search_evidence`, `list_document_titles`)
- Citations, contradiction notes, abstention
- FastAPI backend + simple web UI
- Automated evaluation (40+ questions, two configs)

## Architecture

```
Upload/seed docs
    -> extract text + metadata
    -> chunk (fixed | recursive)
    -> embed + upsert Chroma
    -> sqlite metadata/chunks

Question
    -> hybrid retrieve (dense ⊕ BM25 ⊕ RRF)
    -> evidence-consistency rerank
    -> optional tool-calling loops for sub-queries
    -> grounded JSON answer + citations / contradictions / abstain
```

### Why these choices

| Choice | Reason |
|---|---|
| Chroma | Free, persistent on disk, simple to deploy |
| Hashing embeddings (384-d) | Dense vectors without a multi-GB torch install on free hosts; still supports ANN + hybrid |
| BM25 + RRF + custom rerank | Meets hybrid-search requirement and adds an original ranking component |
| Groq OpenAI-compatible API | Free-tier LLM access; swap `OPENAI_BASE_URL` / model as needed |
| HotpotQA (+ synthetic ops pack) | Multi-hop linked facts; ops pack adds contradictions / incident-style questions |
| FastAPI + static UI | Reliable API surface; UI focuses on function over chrome |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# put a Groq/OpenAI key in .env

python scripts/seed_demo.py
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000

### Build the 50k+ corpus

```bash
python scripts/build_corpus_fast.py --target 50000
# or: python scripts/build_corpus.py --target 50000
```

Uses HotpotQA distractor paragraphs when download works (`hotpot_qa` on Hugging Face — verified), then pads with generated ops notes if needed so the run stays reproducible offline.

Local verification: `python scripts/build_corpus_fast.py --target 50000` produced **50,008** indexed documents / chunks.

## API

- `GET /api/health`
- `GET /api/documents`
- `POST /api/documents/upload`
- `PUT /api/documents/{id}`
- `DELETE /api/documents/{id}`
- `POST /api/investigate`
- `GET /api/search?q=...`
- `POST /api/chunking/compare`

Interactive docs: `/docs`

## Evaluation

```bash
python evaluation/run_eval.py --seed
```

Compares two hybrid alphas and writes `reports/eval_raw.json`. See `reports/EVALUATION_REPORT.md`.

## Tests

```bash
pytest -q
```

## Deployment (free tier)

Designed for **Render** or **Hugging Face Spaces** with Docker.

1. Set secrets: `OPENAI_API_KEY`, optionally `OPENAI_BASE_URL`, `LLM_MODEL`
2. Persist `/data` (Chroma + sqlite + docs) via the platform disk feature when available
3. On boot, `scripts/seed_demo.py` loads the demo pack; run `build_corpus.py` once against persistent storage for the full 50k set

Cold starts / sleep on free tiers are expected — document the public URL wake-up delay in your submission.

### Render

- Connect the GitHub repo
- Runtime: Docker or native Python + `Procfile`
- Add a persistent disk mounted at `/data`
- Env: `CHROMA_PATH=/data/chroma`, `DOCS_PATH=/data/documents`, `META_DB_PATH=/data/meta.sqlite3`

### Hugging Face Spaces

- Space type: Docker
- Port `7860` (already in Dockerfile)
- Add HF secrets for the LLM key

## Project layout

```
app/                FastAPI app, services, models
frontend/           UI
evaluation/         questions + metrics + runner
scripts/            corpus + seed helpers
tests/              automated tests
reports/            evaluation + AI usage reports
```

## Deliverables checklist

- [x] Source code (this repo)
- [ ] Live demo URL (after you deploy)
- [x] README (this file)
- [x] Evaluation report (`reports/EVALUATION_REPORT.md`)
- [x] AI usage report (`reports/AI_USAGE_REPORT.md`)

After deploy, paste the public URL into the Live Demo section of your submission / README top.

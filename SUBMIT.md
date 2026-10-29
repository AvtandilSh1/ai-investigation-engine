# What to upload / submit

## 1. GitHub repository
Push this folder (`ai-investigation-engine`) to a new public GitHub repo.

```bash
cd "ai-investigation-engine"
git init
git add .
git commit -m "AI Investigation Engine — Applied AI task"
# create repo on GitHub, then:
git branch -M main
git remote add origin git@github.com:<YOU>/<REPO>.git
git push -u origin main
```

Do **not** commit `.env` (already gitignored). Keep secrets only in the host dashboard.

## 2. Live demo
Deploy to Render or Hugging Face Spaces (Dockerfile included).

Required env vars:
- `OPENAI_API_KEY` — Groq or OpenAI key
- `OPENAI_BASE_URL` — e.g. `https://api.groq.com/openai/v1`
- `LLM_MODEL` — e.g. `llama-3.3-70b-versatile`
- `CHROMA_PATH`, `DOCS_PATH`, `META_DB_PATH` on a persistent disk if the platform provides one

After deploy:
1. Open the public URL
2. Confirm upload + investigate works
3. Paste the URL into your submission form and at the top of README

One-time on the server (or locally against mounted disk):

```bash
python scripts/build_corpus.py --target 50000
```

## 3. README
Already at `README.md` (architecture, run instructions, design choices).

## 4. Evaluation report
`reports/EVALUATION_REPORT.md` + raw numbers in `reports/eval_raw.json`

Re-generate after setting an LLM key:

```bash
python evaluation/run_eval.py --seed
```

## 5. AI usage report
`reports/AI_USAGE_REPORT.md`

## Local smoke check before upload

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your key
python scripts/seed_demo.py
pytest -q
uvicorn app.main:app --port 8000
```

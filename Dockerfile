FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=7860
ENV CHROMA_PATH=/data/chroma
ENV DOCS_PATH=/data/documents
ENV META_DB_PATH=/data/meta.sqlite3

EXPOSE 7860

CMD ["bash", "-lc", "python scripts/seed_demo.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

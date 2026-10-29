# AI Usage Report

AI assistants were used while building this project. I remain responsible for the final code, architecture, and for defending it in interview.

## Tools used

- Cursor / coding assistant for scaffolding modules, boilerplate wiring, and drafting first-pass README sections
- Manual editing for ranking logic, evaluation design, prompt rules, and deployment choices

## Where AI helped

- FastAPI route skeletons and request/response models
- Initial HTML/CSS layout for the upload + investigate UI
- Reminder checklist against the assignment PDF (formats, hybrid search, eval metrics, deliverables)

## What I changed or wrote myself

- Hybrid fusion details (dense + BM25 + RRF weighting) and the **evidence-consistency re-ranker**
- Contradiction heuristics and abstention behavior in the investigation loop
- Tool-calling investigation flow and offline extractive fallback
- Evaluation set design (42 questions spanning simple / multi-doc / contradiction / unanswerable)
- Corpus build strategy (HotpotQA + synthetic ops pack with intentional conflicts)
- Deployment constraints decision: hashing embeddings instead of heavy torch models on free hosts

## What I did not rely on AI for blindly

- I reviewed every service module before keeping it
- Tests were adjusted until they exercised real upload → search → investigate paths
- Metrics definitions were checked against the assignment wording (Recall@K, MRR, answer correctness, citation accuracy)

## Honest limitation

Generated code can look generic. I trimmed comments, renamed a few internals, and kept the structure close to how I would maintain a small production service so the repo stays readable in a live code discussion.

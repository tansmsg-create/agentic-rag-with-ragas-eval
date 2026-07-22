# CLAUDE.md — RAG Project Implementation Brief

## Objective

Build a full working local RAG (Retrieval-Augmented Generation) pipeline that lets a user query the internal policy documents in this repo via a Streamlit UI, using Ollama for embeddings and generation, backed by ChromaDB for vector storage.

## Scope (v1 — demo / proof of concept)

- Corpus (7 documents): `CODE_OF_CONDUCT.md`, `ACCEPTABLE_USE_POLICY.md`, `CONFIDENTIALITY_AGREEMENT.md`, `DATA_PROTECTION_POLICY.md`, `REMOTE_WORK_POLICY.md`, `IT_SECURITY_POLICY.md`, `WHISTLEBLOWING_POLICY.md` — all in the repo root
- No external data sources, no auth, single-user local demo
- Fictional company templates only — not real client data (the four newer documents are grounded in real Singapore regulatory frameworks — PDPA, TG-FWAR — for realistic content, but the company and named individuals throughout remain fictional)

## Tech Stack

- Python 3.x, LangChain, ChromaDB, Streamlit
- Generation model: `llama3.2` (via Ollama, local)
- Embedding model: `nomic-embed-text` (via Ollama, local, 768-dim)
- Reranker: Cohere Rerank, free tier (`rerank-english-v3.0`) — requires `COHERE_API_KEY`
- Evaluation: RAGAS, judged by Gemini — requires `GOOGLE_API_KEY`

**Assumption flagged for confirmation:** Gemini-as-judge is for offline evaluation only (scoring retrieval/generation quality in a test script), not part of the live query path a user waits on. If Gemini should sit in the runtime loop somewhere instead, this brief needs adjusting.

## Pipeline

```
query → embed (nomic-embed-text) → Chroma similarity search (top-k, e.g. 10)
      → Cohere rerank (top-n, e.g. 3–4)
      → llama3.2 generation (context = reranked chunks)
      → answer + source attribution → Streamlit UI
```

Offline, separate from the above: a RAGAS evaluation script scores the pipeline's outputs (faithfulness, answer relevancy, context precision/recall) using Gemini as the judge model.

## What to Build

### 1. `scripts/ingest.py`

- Load all `.md` policy files from the repo root
- Chunk using a markdown-aware splitter (e.g. LangChain's `MarkdownHeaderTextSplitter`) so section headers stay intact — these are structured policy documents, so splitting on headings will retrieve more coherently than fixed-size chunking
- Embed chunks with `nomic-embed-text` via Ollama (confirm the Chroma collection is created with 768 dimensions — that's `nomic-embed-text`'s native output size, don't let it default to something else)
- Persist to a local ChromaDB store under `data/chroma/`
- Should be re-runnable idempotently (clear and rebuild the collection, or detect changes)

### 2. `src/retriever.py` (new module)

- Given a query, embed it and run a Chroma similarity search for the top-k candidates (k configurable, default ~10)
- Pass candidates to Cohere Rerank (`rerank-english-v3.0`), keep the top-n (default ~3–4)
- Return reranked chunks with their source document/section metadata intact
- If `COHERE_API_KEY` is missing or the Cohere call fails, fall back to the raw Chroma similarity ranking rather than erroring out — log a warning so it's visible during local runs

### 3. `src/rag_chain.py` (new module)

- A `RAGChain` class or function: given a query, call `retriever.py` for reranked context, construct a prompt with that context, call Ollama (`llama3.2`) for generation
- Return the source document/section alongside the answer so responses are traceable
- Handle empty or low-relevance retrieval gracefully — the chain should say it doesn't know rather than fabricate an answer

### 4. `src/app.py` (replace placeholder)

- Wire the existing text input to `rag_chain`
- Display the answer plus which document(s) it drew from
- Optional: sidebar showing the active model name, chunk count, and a "rebuild index" button that calls `ingest.py`

### 5. `scripts/evaluate.py` (new — offline eval)

- Build a small test set of question/expected-answer pairs drawn from across the policy documents (5–10 questions is enough for a demo)
- Run each question through the full pipeline (retrieve → rerank → generate)
- Score with RAGAS metrics (faithfulness, answer relevancy, context precision, context recall)
- Output a simple results table (CSV or printed to console is fine for v1)

**Deviation from spec:** the judge LLM was meant to be Gemini (`GOOGLE_API_KEY`), per the Tech Stack section. Three problems ruled out the free-tier options in practice, in order encountered:
1. `ragas.evaluate()`'s built-in executor applies `nest_asyncio` and runs jobs through its own event-loop wrapper, which hung indefinitely against Gemini's async client in this environment (persisted even at `max_workers=1`, so it's specific to the executor rather than concurrency; direct calls to the same Gemini client outside that executor worked fine). Root cause unresolved — worked around by calling the RAGAS metric objects directly (`metric.single_turn_ascore(...)`) inside a single `asyncio.run()`, bypassing `ragas.evaluate()` entirely. This fix is retained regardless of judge provider.
2. Once that was fixed, Gemini's free tier turned out to cap at 20 requests/day *per model*, and `context_precision` alone burns roughly one call per retrieved chunk — a single question's full 4-metric evaluation can exhaust the daily quota by itself.
3. Switching to OpenRouter's free-tier `openai/gpt-oss-20b:free` avoided Gemini's per-model cap but hit OpenRouter's own free-tier cap (50 requests/day, platform-wide) partway through the 8-question run, and separately showed occasional `context_precision` parsing failures against that model (a `ragas`-internal schema mismatch, intermittent).

`scripts/evaluate.py` therefore uses **Claude** as judge (`ANTHROPIC_API_KEY`, requires API billing at console.anthropic.com — separate from a Claude Pro chat subscription) — no quota walls and no parsing issues across all four metrics in testing. `GOOGLE_API_KEY` is consequently unused by the current implementation.

### 6. Testing

- Smoke test with a question clearly answered in one document (e.g. "What should I do if I suspect a policy violation?") — confirm the retrieved chunk and answer are relevant
- Smoke test with an out-of-scope question (e.g. "What's the weather today?") — confirm the chain declines rather than fabricates
- Run `scripts/evaluate.py` once the pipeline is wired up and sanity-check the RAGAS scores aren't degenerate (e.g. faithfulness near zero would indicate a prompt or context-passing bug)

## Environment Variables

Needs a `.env` file (gitignored, see `.env.example`) with:

```
COHERE_API_KEY=
GOOGLE_API_KEY=
```

Ollama models run locally and need no API key, but must be pulled first: `ollama pull llama3.2` and `ollama pull nomic-embed-text`.

## Out of Scope for v1

- Multi-user support or authentication
- Real company/client documents
- Cloud deployment (local Streamlit run only)
- Model-selection UI beyond a simple sidebar note
- Gemini in the live query path (see assumption flagged above)

## Git / GitHub

- This repo already exists on GitHub — push to the existing remote, do not create a new one
- Suggested commit sequence:
  1. Confirm scaffold (docs, `requirements.txt`, placeholder `app.py`) is already committed
  2. Add `ingest.py` + `rag_chain.py` + updated `app.py` as one feature commit
  3. Add `data/chroma/` to `.gitignore` — the vector store shouldn't be versioned
- Confirm target branch (`main` vs a feature branch) before pushing

## Notes for Claude Code

- Verify `chromadb` and `sentence-transformers` install cleanly in the target environment before writing ingestion logic against them — memory-constrained installs have previously hit "Exit code 137"; install packages incrementally if that recurs
- British English throughout: comments, docstrings, and UI copy

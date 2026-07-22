# Internal Documentation RAG — Setup Guide

A local Retrieval-Augmented Generation (RAG) project for querying internal company policy documents, built with LangChain, ChromaDB, Ollama, and Streamlit.

## 1. Project Structure

```
your-project-folder/
├── src/
│   └── app.py
├── data/
├── scripts/
├── requirements.txt
├── CODE_OF_CONDUCT.md
├── ACCEPTABLE_USE_POLICY.md
└── CONFIDENTIALITY_AGREEMENT.md
```

## 2. Setup Instructions

### Step 1 — Install Ollama

Go to https://ollama.com/ and download the installer for your operating system, then follow the installation prompts.

### Step 2 — Download Ollama Models

Open your terminal and run:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

`nomic-embed-text` handles embeddings (768-dim). `llama3.2` handles generation.

### Step 2a — API Keys

This project also uses two hosted services on their free tiers:

- **Cohere Rerank** — improves retrieval quality by reranking candidate chunks before generation
- **Gemini** — used as the judge model for offline RAGAS evaluation (not part of the live query path)

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

### Step 3 — Set Up a Python Virtual Environment

From the project root:

```bash
python3 -m venv .venv
```

Activate it:

- **macOS/Linux:** `source .venv/bin/activate`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

You should see `(.venv)` prefixing your prompt once active.

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Installing `chromadb` and related packages can be resource-intensive. If you hit an "Exit code 137" error, this usually indicates insufficient memory during installation. Try installing packages one at a time (`pip install langchain`, then `pip install chromadb`, etc.) rather than all at once, and close other memory-hungry applications first.

### Step 5 — Run the Streamlit App

```bash
streamlit run src/app.py
```

This launches the placeholder UI at `http://localhost:8501`. The RAG chain itself (embedding the policy documents into ChromaDB and wiring up retrieval + generation via Ollama) still needs to be implemented in `src/app.py` — the current version just echoes back the question as a placeholder.

## 3. Next Steps

- Write an ingestion script in `scripts/` to chunk and embed the policy documents into ChromaDB.
- Wire up a LangChain retrieval chain in `src/app.py`, replacing the placeholder answer logic.
- Consider adding a model-selection dropdown and chunk-size controls to the Streamlit sidebar.

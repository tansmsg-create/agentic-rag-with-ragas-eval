"""Shared configuration for the RAG pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "data" / "chroma"
COLLECTION_NAME = "policy_docs"

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
GEN_MODEL = "llama3.2"

# Judge LLM for offline RAGAS evaluation only, never the live query path.
# CLAUDE.md specifies Gemini as judge; both Gemini's free tier (20 req/day
# per model) and OpenRouter's free tier (50 req/day platform-wide) proved
# too small for an 8-question x 4-metric run. Using Claude instead -- no
# quota walls, reliable structured output, no parsing issues across all
# four metrics. See CLAUDE.md "Deviation from spec".
JUDGE_MODEL = "claude-haiku-4-5-20251001"

RERANK_MODEL = "rerank-english-v3.0"
DEFAULT_TOP_K = 10
DEFAULT_TOP_N = 4

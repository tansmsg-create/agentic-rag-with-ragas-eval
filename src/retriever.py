"""Retrieve and rerank policy chunks for a query."""

import logging
import os

import chromadb
import ollama
from dotenv import load_dotenv

from config import CHROMA_DIR, COLLECTION_NAME, DEFAULT_TOP_K, DEFAULT_TOP_N, EMBED_MODEL, RERANK_MODEL

load_dotenv()
logger = logging.getLogger(__name__)


class RetrievedChunk:
    def __init__(self, text: str, source: str, section: str, score: float):
        self.text = text
        self.source = source
        self.section = section
        self.score = score


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def _embed_query(query: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=query)
    return list(response.embeddings[0])


def _chroma_search(query: str, top_k: int) -> list[RetrievedChunk]:
    collection = _get_collection()
    query_embedding = _embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        # Chroma returns a distance (lower = more similar); convert to a
        # similarity-style score so callers see "higher is better" throughout.
        score = 1.0 / (1.0 + distance)
        chunks.append(
            RetrievedChunk(
                text=text,
                source=metadata.get("source", "unknown"),
                section=metadata.get("section", "unknown"),
                score=score,
            )
        )
    return chunks


def _cohere_rerank(query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    import cohere

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is not set")

    client = cohere.ClientV2(api_key=api_key)
    response = client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=[c.text for c in candidates],
        top_n=top_n,
    )

    reranked = []
    for result in response.results:
        original = candidates[result.index]
        reranked.append(
            RetrievedChunk(
                text=original.text,
                source=original.source,
                section=original.section,
                score=result.relevance_score,
            )
        )
    return reranked


def retrieve(query: str, top_k: int = DEFAULT_TOP_K, top_n: int = DEFAULT_TOP_N) -> list[RetrievedChunk]:
    """Return the top-n reranked chunks for a query.

    Falls back to raw Chroma similarity ranking (top-n of top-k) if Cohere
    Rerank is unavailable or fails, rather than erroring out.
    """
    candidates = _chroma_search(query, top_k)
    if not candidates:
        return []

    try:
        return _cohere_rerank(query, candidates, top_n)
    except Exception as exc:
        logger.warning("Cohere rerank unavailable (%s); falling back to raw similarity ranking", exc)
        return candidates[:top_n]

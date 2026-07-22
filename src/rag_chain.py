"""RAG chain: retrieve reranked policy context, then generate a grounded answer."""

import ollama

from config import DEFAULT_TOP_K, DEFAULT_TOP_N, GEN_MODEL
from retriever import retrieve

# Below this top-chunk relevance score, treat the query as out of scope rather
# than risk the model fabricating an answer from weak context. Calibrated
# against Cohere relevance scores: on-topic queries scored >= 0.038, clearly
# off-topic queries (e.g. "what's the weather today?") scored <= 0.001.
RELEVANCE_THRESHOLD = 0.02

DECLINE_MESSAGE = (
    "I don't have information about that in the current policy documents. "
    "Please try rephrasing, or check with the relevant policy owner."
)

SYSTEM_PROMPT = (
    "You are an internal assistant that answers employee questions about company "
    "policy using only the provided context excerpts. Answer in British English. "
    "If the context does not contain enough information to answer confidently, "
    "say so plainly rather than guessing or inventing details. Keep answers concise "
    "and reference the relevant policy section by name where useful."
)


class RAGChain:
    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        top_n: int = DEFAULT_TOP_N,
        gen_model: str = GEN_MODEL,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
    ):
        self.top_k = top_k
        self.top_n = top_n
        self.gen_model = gen_model
        self.relevance_threshold = relevance_threshold

    def query(self, question: str) -> dict:
        chunks = retrieve(question, top_k=self.top_k, top_n=self.top_n)

        if not chunks or max(c.score for c in chunks) < self.relevance_threshold:
            return {"answer": DECLINE_MESSAGE, "sources": []}

        context = "\n\n".join(f"[{c.source} — {c.section}]\n{c.text}" for c in chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        response = ollama.chat(
            model=self.gen_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        sources = [
            {"source": c.source, "section": c.section, "score": c.score, "text": c.text}
            for c in chunks
        ]
        return {"answer": response.message.content, "sources": sources}

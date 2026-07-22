import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, GEN_MODEL  # noqa: E402
from rag_chain import RAGChain  # noqa: E402

st.title("Internal Documentation RAG")
st.write("Ask questions about company policies and procedures.")


def get_chunk_count():
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_collection(COLLECTION_NAME).count()
    except Exception:
        return None


with st.sidebar:
    st.header("Pipeline info")
    st.write(f"Generation model: `{GEN_MODEL}`")
    st.write(f"Embedding model: `{EMBED_MODEL}`")
    chunk_count = get_chunk_count()
    st.write(f"Indexed chunks: {chunk_count if chunk_count is not None else 'index not built yet'}")
    if st.button("Rebuild index"):
        with st.spinner("Rebuilding index from policy documents..."):
            import ingest

            ingest.main()
        st.success("Index rebuilt.")
        st.rerun()


@st.cache_resource
def get_chain():
    return RAGChain()


chain = get_chain()

user_question = st.text_input("Your Question:")

if user_question:
    with st.spinner("Thinking..."):
        result = chain.query(user_question)

    st.write("Answer:")
    st.write(result["answer"])

    if result["sources"]:
        st.write("Sources:")
        for source in result["sources"]:
            st.write(f"- {source['source']} — {source['section']} (relevance {source['score']:.3f})")

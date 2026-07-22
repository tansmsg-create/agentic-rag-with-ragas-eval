"""Ingest the policy documents into a local ChromaDB collection.

Chunks each Markdown policy file on its headers (so a section stays intact),
embeds each chunk with `nomic-embed-text` via Ollama, and persists the
result to data/chroma/. Re-running this script rebuilds the collection
from scratch, so it is safe to run repeatedly after editing the source docs.
"""

import re
import sys
from pathlib import Path

import chromadb
import ollama
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from config import CHROMA_DIR, COLLECTION_NAME, EMBED_DIM, EMBED_MODEL  # noqa: E402

POLICY_FILES = [
    "CODE_OF_CONDUCT.md",
    "ACCEPTABLE_USE_POLICY.md",
    "CONFIDENTIALITY_AGREEMENT.md",
    "DATA_PROTECTION_POLICY.md",
    "REMOTE_WORK_POLICY.md",
    "IT_SECURITY_POLICY.md",
    "WHISTLEBLOWING_POLICY.md",
]

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# These policy docs mark sections with bold numbered headers (e.g. "**3. Acceptable
# Use**") rather than real Markdown "##" headers, so a plain MarkdownHeaderTextSplitter
# only ever sees each file's single "#" title. Split on that bold-header convention
# instead so a chunk still corresponds to one coherent policy section.
BOLD_SECTION_RE = re.compile(r"^\*\*\d+\.\s*(.+?)\*\*\s*$", re.MULTILINE)


def split_by_bold_sections(text: str, doc_title: str):
    matches = list(BOLD_SECTION_RE.finditer(text))
    if not matches:
        return [Document(page_content=text.strip(), metadata={"h1": doc_title})]

    docs = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        docs.append(
            Document(page_content=body, metadata={"h1": doc_title, "h2": match.group(1).strip()})
        )
    return docs


def load_and_chunk():
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    chunks = []
    for filename in POLICY_FILES:
        path = ROOT_DIR / filename
        if not path.exists():
            print(f"WARNING: {filename} not found at {path}, skipping", file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8")

        title_match = re.match(r"^#\s+(.+)$", text, re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else filename

        top_level_docs = md_splitter.split_text(text)
        for doc in top_level_docs:
            for sub_doc in split_by_bold_sections(doc.page_content, doc.metadata.get("h1", doc_title)):
                section = " > ".join(
                    sub_doc.metadata[key] for key in ("h1", "h2", "h3") if key in sub_doc.metadata
                )
                sub_doc.metadata["source"] = filename
                sub_doc.metadata["section"] = section or filename
                chunks.append(sub_doc)
    return chunks


def embed(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    vector = response.embeddings[0]
    if len(vector) != EMBED_DIM:
        raise RuntimeError(
            f"Expected {EMBED_DIM}-dim embedding from {EMBED_MODEL}, got {len(vector)}"
        )
    return list(vector)


def build_collection(chunks):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"embedding_dim": EMBED_DIM}
    )

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"{chunk.metadata['source']}-{i}")
        embeddings.append(embed(chunk.page_content))
        documents.append(chunk.page_content)
        metadatas.append(chunk.metadata)

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return collection


def main():
    chunks = load_and_chunk()
    if not chunks:
        print("ERROR: no chunks produced, aborting ingest", file=sys.stderr)
        sys.exit(1)
    collection = build_collection(chunks)
    print(f"Ingested {len(chunks)} chunks from {len(POLICY_FILES)} documents "
          f"into collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()

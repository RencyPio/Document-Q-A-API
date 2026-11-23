"""The RAG pipeline: ingest a document, answer a question. No FastAPI here, so it is easy to test."""
import hashlib
import re
import time
from dataclasses import dataclass

from .chunking import chunk_text, extract_pages
from .embeddings import Embedder
from .llm import NO_ANSWER, SYSTEM_PROMPT, LLMClient, build_user_prompt
from .vector_store import RetrievedChunk, VectorStore

CITATION_RE = re.compile(r"\[([^\[\]]+?::\d+)\]")
NO_CONTEXT_ANSWER = "You haven't uploaded any documents yet. Upload a file first, then ask your question."


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    chunk_ids: list[str]


@dataclass
class Answer:
    answer: str
    citations: list[str]           # chunk IDs the model actually cited (and that we retrieved)
    sources: list[RetrievedChunk]  # everything we retrieved, for transparency
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str | None = None
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0


def make_doc_id(filename: str, data: bytes) -> str:
    """Stable ID: same file uploaded twice overwrites instead of duplicating."""
    digest = hashlib.sha256(data).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return f"{stem}-{digest}"


class RAGService:
    def __init__(self, embedder: Embedder, store: VectorStore, llm: LLMClient | None,
                 chunk_size: int = 800, chunk_overlap: int = 150, top_k: int = 4):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    def ingest(self, filename: str, data: bytes) -> IngestResult:
        chunks: list[str] = []
        pages: list[int | None] = []
        for page, text in extract_pages(filename, data):  # chunk per page so citations keep a page number
            for chunk in chunk_text(text, self.chunk_size, self.chunk_overlap):
                chunks.append(chunk)
                pages.append(page)
        if not chunks:
            raise ValueError(f"No text found in {filename}. If it's a scanned PDF, it needs OCR "
                             "(text recognition) before it can be searched.")
        doc_id = make_doc_id(filename, data)
        ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
        metas = [
            {"source": filename, "doc_id": doc_id, "chunk_index": i, "page": pages[i] or 0}  # Chroma disallows None
            for i in range(len(chunks))
        ]
        self.store.add(ids, chunks, self.embedder.embed(chunks), metas)
        return IngestResult(doc_id=doc_id, filename=filename, chunk_ids=ids)

    def list_documents(self) -> list[dict]:
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> int:
        return self.store.delete_document(doc_id)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        [query_vec] = self.embedder.embed([question])
        return self.store.search(query_vec, top_k or self.top_k)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        t0 = time.perf_counter()
        chunks = self.retrieve(question, top_k)
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
        if not chunks:
            return Answer(answer=NO_CONTEXT_ANSWER, citations=[], sources=[], retrieval_ms=retrieval_ms)
        if self.llm is None:
            raise RuntimeError(
                "Answers need an API key. Add GROQ_API_KEY (free at console.groq.com) to your .env file "
                "and restart the server. To use OpenAI instead, set LLM_PROVIDER=openai and OPENAI_API_KEY."
            )
        t1 = time.perf_counter()
        result = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, chunks))
        llm_ms = round((time.perf_counter() - t1) * 1000, 1)
        retrieved_ids = {c.chunk_id for c in chunks}
        cited = list(dict.fromkeys(m for m in CITATION_RE.findall(result.text) if m in retrieved_ids))
        return Answer(
            answer=result.text, citations=cited, sources=chunks,
            prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
            model=result.model, retrieval_ms=retrieval_ms, llm_ms=llm_ms,
        )

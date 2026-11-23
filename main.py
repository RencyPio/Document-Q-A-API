"""FastAPI app: POST /documents to upload and index, POST /ask to query."""
import logging
import time
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .llm import LLMError
from .monitoring import LatencyMiddleware, log_event
from .rag import RAGService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

DESCRIPTION = """
Upload your own PDFs or text files, then ask questions about them in plain English.
Every answer is written only from your documents and cites the passages it used.

**How to use it**
1. **Upload** one or more files with `POST /documents`.
2. **Ask** a question with `POST /ask`. You get the answer, the cited passages, and the token usage.

**How it works:** files are split into chunks, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB.
A question pulls back the closest chunks, and an LLM (Groq's free Llama models by default, or OpenAI)
answers from those chunks only.

Prefer a simple form? Open the [home page](/).
"""

TAGS = [
    {"name": "Documents", "description": "Upload and index your files."},
    {"name": "Questions", "description": "Ask questions and get cited answers."},
]

app = FastAPI(
    title="Document Q&A API",
    version="0.1.0",
    summary="Ask questions about your own documents and get answers with citations.",
    description=DESCRIPTION,
    openapi_tags=TAGS,
    # Keep /docs tidy: endpoints collapsed, no schema list at the bottom.
    swagger_ui_parameters={"defaultModelsExpandDepth": -1, "docExpansion": "list", "tryItOutEnabled": True},
)

app.add_middleware(LatencyMiddleware)


@lru_cache
def get_rag_service() -> RAGService:
    """Build the real service once. Tests override this dependency with fakes."""
    from .embeddings import SentenceTransformerEmbedder
    from .llm import OpenAIClient
    from .vector_store import VectorStore

    s: Settings = get_settings()
    c = s.llm
    llm = (OpenAIClient(c["api_key"], c["model"], base_url=c["base_url"], provider=c["provider"],
                        key_env=c["key_env"]) if c["api_key"] else None)
    return RAGService(
        embedder=SentenceTransformerEmbedder(s.embedding_model),
        store=VectorStore.persistent(s.chroma_dir, s.collection_name),
        llm=llm, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap, top_k=s.top_k,
    )


class IndexedFile(BaseModel):
    doc_id: str
    filename: str
    chunks: int


class SkippedFile(BaseModel):
    filename: str
    reason: str


class DocumentsResponse(BaseModel):
    indexed: list[IndexedFile]
    skipped: list[SkippedFile] = []
    total_chunks: int


class StoredDocument(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    pages: int | None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["What is the refund policy?"])
    top_k: int | None = Field(default=None, ge=1, le=10, description="How many passages to use (default 4).")


class Source(BaseModel):
    chunk_id: str
    source: str
    page: int | None
    score: float
    cited: bool = Field(description="True if the answer cites this passage.")
    text: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Latency(BaseModel):
    retrieval_ms: float
    llm_ms: float
    total_ms: float


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    sources: list[Source]
    model: str | None
    usage: Usage
    latency_ms: Latency


HOME_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return HOME_PAGE.read_text(encoding="utf-8")


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.post(
    "/documents", response_model=DocumentsResponse, status_code=201, tags=["Documents"],
    summary="Upload documents",
    description="Upload one or more PDF, .txt or .md files. Each file is split into chunks and indexed "
                "so you can ask questions about it. Uploading the same file again replaces it. "
                "Files that can't be read are listed under `skipped` with the reason; if none could be "
                "read, the request fails with 400.",
)
async def upload_documents(files: list[UploadFile] = File(...), rag: RAGService = Depends(get_rag_service)):
    max_mb = get_settings().max_upload_mb
    indexed: list[IndexedFile] = []
    skipped: list[SkippedFile] = []
    for f in files:
        name = f.filename or "upload.txt"
        data = await f.read()
        try:
            if not data:
                raise ValueError(f"{name} is empty.")
            if len(data) > max_mb * 1024 * 1024:
                raise ValueError(f"{name} is larger than the {max_mb} MB limit.")
            r = rag.ingest(name, data)
        except ValueError as e:
            skipped.append(SkippedFile(filename=name, reason=str(e)))
            continue
        indexed.append(IndexedFile(doc_id=r.doc_id, filename=r.filename, chunks=len(r.chunk_ids)))
    if not indexed:
        raise HTTPException(400, " ".join(s.reason for s in skipped))
    return DocumentsResponse(indexed=indexed, skipped=skipped, total_chunks=sum(i.chunks for i in indexed))


@app.get("/documents", response_model=list[StoredDocument], tags=["Documents"], summary="List documents",
         description="Every document that has been uploaded, with its number of chunks and pages.")
def list_documents(rag: RAGService = Depends(get_rag_service)):
    return rag.list_documents()


@app.delete("/documents/{doc_id}", status_code=204, tags=["Documents"], summary="Remove a document",
            description="Remove a document and all its chunks, so it is no longer used in answers.")
def delete_document(doc_id: str, rag: RAGService = Depends(get_rag_service)):
    if rag.delete_document(doc_id) == 0:
        raise HTTPException(404, "That document wasn't found. It may already have been removed.")


@app.post(
    "/ask", response_model=AskResponse, tags=["Questions"],
    summary="Ask a question",
    description="Ask a question about the uploaded documents. The answer uses only those documents and "
                "cites chunk IDs like `[report.pdf-1a2b3c4d::3]`. The response also lists the source "
                "passages, token usage and timings.",
)
def ask(body: AskRequest, request: Request, rag: RAGService = Depends(get_rag_service)):
    start = time.perf_counter()
    try:
        ans = rag.ask(body.question, body.top_k)
    except LLMError as e:
        raise HTTPException(502, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    total_ms = round((time.perf_counter() - start) * 1000, 1)
    usage = Usage(prompt_tokens=ans.prompt_tokens, completion_tokens=ans.completion_tokens,
                  total_tokens=ans.prompt_tokens + ans.completion_tokens)
    latency = Latency(retrieval_ms=ans.retrieval_ms, llm_ms=ans.llm_ms, total_ms=total_ms)
    log_event(event="llm_request", request_id=getattr(request.state, "request_id", None),
              model=ans.model, chunks_retrieved=len(ans.sources), citations=len(ans.citations),
              **usage.model_dump(), **latency.model_dump())
    return AskResponse(
        answer=ans.answer, citations=ans.citations, model=ans.model, usage=usage, latency_ms=latency,
        sources=[Source(chunk_id=c.chunk_id, source=c.source, page=c.page, score=round(c.score, 4),
                        cited=c.chunk_id in ans.citations, text=c.text)
                 for c in ans.sources],
    )

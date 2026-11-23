"""HTTP layer, with the RAG service wired to the fake embedder, in-memory Chroma and fake LLM."""
from app.main import get_rag_service, app
from app.rag import RAGService


def _upload(client, *files):
    return client.post("/documents", files=[("files", f) for f in files])


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_indexes_files(client):
    r = _upload(client, ("a.txt", b"Refunds within 30 days.", "text/plain"),
                ("b.md", b"# Shipping\nTakes 5 days.", "text/markdown"))
    assert r.status_code == 201
    body = r.json()
    assert [f["filename"] for f in body["indexed"]] == ["a.txt", "b.md"]
    assert body["total_chunks"] == sum(f["chunks"] for f in body["indexed"]) >= 2


def test_upload_rejects_when_nothing_is_readable(client):
    r = _upload(client, ("x.docx", b"...", "application/octet-stream"))
    assert r.status_code == 400
    assert "Unsupported" in r.json()["detail"]


def test_upload_skips_bad_files_but_indexes_good_ones(client):
    r = _upload(client, ("good.txt", b"Refunds within 30 days.", "text/plain"),
                ("bad.docx", b"...", "application/octet-stream"),
                ("empty.txt", b"   ", "text/plain"))
    assert r.status_code == 201
    body = r.json()
    assert [f["filename"] for f in body["indexed"]] == ["good.txt"]
    assert {s["filename"] for s in body["skipped"]} == {"bad.docx", "empty.txt"}
    assert all(s["reason"] for s in body["skipped"])


def test_list_and_delete_documents(client):
    _upload(client, ("a.txt", b"Refunds within 30 days.", "text/plain"),
            ("b.txt", b"Shipping takes 5 days.", "text/plain"))
    docs = client.get("/documents").json()
    assert [d["filename"] for d in docs] == ["a.txt", "b.txt"]

    assert client.delete(f"/documents/{docs[0]['doc_id']}").status_code == 204
    assert [d["filename"] for d in client.get("/documents").json()] == ["b.txt"]
    assert client.delete(f"/documents/{docs[0]['doc_id']}").status_code == 404


def test_ask_returns_answer_citations_usage_and_latency(client):
    _upload(client, ("refunds.txt", b"Customers can request a refund within 30 days.", "text/plain"))
    r = client.post("/ask", json={"question": "What is the refund window?", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] and body["citations"][0].startswith("refunds.txt-")
    assert body["sources"][0]["source"] == "refunds.txt"
    assert [s["cited"] for s in body["sources"]] == [s["chunk_id"] in body["citations"] for s in body["sources"]]
    assert body["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    assert set(body["latency_ms"]) == {"retrieval_ms", "llm_ms", "total_ms"}
    assert body["model"] == "fake-model"


def test_ask_sets_monitoring_headers(client):
    r = client.post("/ask", json={"question": "hi"})
    assert r.headers["X-Request-ID"]
    assert float(r.headers["X-Latency-Ms"]) >= 0


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "q", "top_k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "q", "top_k": 11}).status_code == 422


def test_ask_returns_503_without_llm(embedder, store):
    from fastapi.testclient import TestClient

    rag = RAGService(embedder=embedder, store=store, llm=None)
    rag.ingest("a.txt", b"some text")
    app.dependency_overrides[get_rag_service] = lambda: rag
    try:
        r = TestClient(app).post("/ask", json={"question": "q"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503


def test_home_page_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")


def test_ask_returns_502_when_openai_fails(client, fake_llm):
    from app.llm import LLMError

    def fail(system, user):
        raise LLMError("OpenAI's rate limit or quota was reached.")

    fake_llm.complete = fail
    _upload(client, ("a.txt", b"some text", "text/plain"))
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 502
    assert "rate limit" in r.json()["detail"]

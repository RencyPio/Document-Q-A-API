"""Retrieval: the vector store on its own, then RAGService.ingest + retrieve end to end."""
from app.rag import make_doc_id

DOCS = {
    "refunds.txt": [
        "Customers can request a refund within 30 days of purchase.",
        "Refunds are paid back to the original payment method.",
    ],
    "shipping.txt": [
        "Standard shipping takes 5 to 7 business days.",
        "Express shipping arrives in 2 business days for an extra fee.",
    ],
}


def _fill(store, embedder):
    for name, chunks in DOCS.items():
        ids = [f"{name}::{i}" for i in range(len(chunks))]
        metas = [{"source": name, "page": 0} for _ in chunks]
        store.add(ids, chunks, embedder.embed(chunks), metas)


# --- VectorStore -----------------------------------------------------------

def test_search_on_empty_store_returns_nothing(store, embedder):
    assert store.search(embedder.embed(["anything"])[0], top_k=3) == []


def test_search_returns_top_k(store, embedder):
    _fill(store, embedder)
    assert len(store.search(embedder.embed(["shipping time"])[0], top_k=3)) == 3


def test_top_k_larger_than_store_returns_everything(store, embedder):
    _fill(store, embedder)
    assert len(store.search(embedder.embed(["anything"])[0], top_k=50)) == 4


def test_most_relevant_chunk_ranks_first(store, embedder):
    _fill(store, embedder)
    results = store.search(embedder.embed(["express shipping extra fee"])[0], top_k=4)
    assert results[0].chunk_id == "shipping.txt::1"


def test_results_are_sorted_by_score(store, embedder):
    _fill(store, embedder)
    scores = [r.score for r in store.search(embedder.embed(["refund purchase"])[0], top_k=4)]
    assert scores == sorted(scores, reverse=True)


def test_result_fields_are_filled(store, embedder):
    _fill(store, embedder)
    top = store.search(embedder.embed(["refund within 30 days"])[0], top_k=1)[0]
    assert top.chunk_id == "refunds.txt::0"
    assert top.source == "refunds.txt"
    assert "30 days" in top.text
    assert 0.0 < top.score <= 1.0 + 1e-6
    assert top.page is None  # page 0 is stored for text files and read back as None


def test_upsert_with_same_ids_does_not_duplicate(store, embedder):
    _fill(store, embedder)
    _fill(store, embedder)
    assert store.count() == 4


# --- RAGService ingest + retrieve -------------------------------------------

def test_ingest_creates_chunk_ids_from_doc_id(rag):
    data = b"Refunds are accepted within 30 days. " * 20
    result = rag.ingest("policy.txt", data)
    assert result.doc_id == make_doc_id("policy.txt", data)
    assert result.chunk_ids == [f"{result.doc_id}::{i}" for i in range(len(result.chunk_ids))]
    assert len(result.chunk_ids) > 1  # chunk_size is 200 in the fixture


def test_doc_id_is_stable_and_safe():
    a = make_doc_id("my report (v2).pdf", b"abc")
    assert a == make_doc_id("my report (v2).pdf", b"abc")
    assert a != make_doc_id("my report (v2).pdf", b"abcd")
    assert " " not in a and "(" not in a


def test_reingesting_same_file_does_not_duplicate(rag, store):
    rag.ingest("policy.txt", b"Refunds within 30 days.")
    rag.ingest("policy.txt", b"Refunds within 30 days.")
    assert store.count() == 1


def test_ingest_rejects_empty_file(rag):
    import pytest

    with pytest.raises(ValueError, match="No text"):
        rag.ingest("empty.txt", b"   ")


def test_retrieve_finds_the_right_document(rag):
    rag.ingest("refunds.txt", " ".join(DOCS["refunds.txt"]).encode())
    rag.ingest("shipping.txt", " ".join(DOCS["shipping.txt"]).encode())
    top = rag.retrieve("How many business days does express shipping take?", top_k=1)[0]
    assert top.source == "shipping.txt"


def test_retrieve_uses_default_top_k(rag):
    for i in range(5):
        rag.ingest(f"doc{i}.txt", f"Document number {i} talks about topic {i}.".encode())
    assert len(rag.retrieve("topic")) == 3  # fixture top_k


# --- listing and deleting documents ----------------------------------------

def test_list_documents_groups_chunks_by_document(rag):
    a = rag.ingest("b_policy.txt", b"Refunds are accepted within 30 days. " * 20)
    b = rag.ingest("a_notes.txt", b"Short note.")
    docs = rag.list_documents()
    assert [d["filename"] for d in docs] == ["a_notes.txt", "b_policy.txt"]  # sorted by name
    by_id = {d["doc_id"]: d for d in docs}
    assert by_id[a.doc_id]["chunks"] == len(a.chunk_ids)
    assert by_id[b.doc_id]["chunks"] == 1
    assert by_id[b.doc_id]["pages"] is None  # text files have no pages


def test_list_documents_on_empty_store(rag):
    assert rag.list_documents() == []


def test_delete_document_removes_only_that_document(rag, store):
    keep = rag.ingest("keep.txt", b"Shipping takes 5 days.")
    drop = rag.ingest("drop.txt", b"Refunds within 30 days. " * 20)
    assert rag.delete_document(drop.doc_id) == len(drop.chunk_ids)
    assert store.count() == len(keep.chunk_ids)
    assert all(c.source == "keep.txt" for c in rag.retrieve("refunds", top_k=5))


def test_delete_unknown_document_returns_zero(rag):
    assert rag.delete_document("nope-00000000") == 0

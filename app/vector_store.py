"""ChromaDB wrapper: add chunks, search by vector."""
from dataclasses import dataclass

import chromadb


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    score: float
    page: int | None = None  # cosine similarity, higher is better


class VectorStore:
    def __init__(self, client: "chromadb.api.ClientAPI", collection_name: str):
        self.collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    @classmethod
    def persistent(cls, path: str, collection_name: str) -> "VectorStore":
        return cls(chromadb.PersistentClient(path=path), collection_name)

    @classmethod
    def in_memory(cls, collection_name: str | None = None) -> "VectorStore":
        """Non-persistent store for tests. Ephemeral clients share state within a process,
        so each call gets a fresh, uniquely named collection unless you pass a name."""
        import uuid

        return cls(chromadb.EphemeralClient(), collection_name or f"mem-{uuid.uuid4().hex[:12]}")

    def add(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def search(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(chunk_id=cid, text=doc, source=meta.get("source", ""), score=1 - dist,
                           page=meta.get("page") or None)
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def list_documents(self) -> list[dict]:
        """One entry per uploaded file: doc_id, filename, chunk count and page count."""
        docs: dict[str, dict] = {}
        for meta in self.collection.get(include=["metadatas"])["metadatas"]:
            d = docs.setdefault(meta["doc_id"], {"doc_id": meta["doc_id"], "filename": meta.get("source", ""),
                                                 "chunks": 0, "pages": set()})
            d["chunks"] += 1
            if meta.get("page"):
                d["pages"].add(meta["page"])
        return [{**d, "pages": len(d["pages"]) or None} for d in sorted(docs.values(), key=lambda d: d["filename"].lower())]

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of one document. Returns how many chunks were removed."""
        ids = self.collection.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self.collection.count()

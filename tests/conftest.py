"""Shared test fixtures. Everything runs offline: no model download and no OpenAI calls."""
import hashlib
import math

import pytest
from fastapi.testclient import TestClient

from app.llm import LLMResult
from app.main import app, get_rag_service
from app.rag import RAGService
from app.vector_store import VectorStore

STOPWORDS = {"a", "an", "the", "is", "are", "of", "to", "in", "on", "for", "and", "how", "what", "do", "does", "can", "i"}


class FakeEmbedder:
    """Deterministic bag-of-words embedder.

    Texts that share words get similar vectors, which is enough to check that
    retrieval ranks the relevant chunk first, without downloading a model.
    """

    dim = 256

    def __init__(self):
        self.calls: list[list[str]] = []

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for raw in text.lower().split():
            word = raw.strip(".,?!:;\"'()[]")
            if not word or word in STOPWORDS:
                continue
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:  # Chroma's cosine distance can't handle a zero vector
            v[0], norm = 1.0, 1.0
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]


class FakeLLM:
    """Stands in for OpenAIClient. Records prompts and returns a scripted reply.

    By default it cites the first chunk ID it sees in the prompt, like a well-behaved model.
    """

    def __init__(self, reply: str | None = None, prompt_tokens: int = 100, completion_tokens: int = 20):
        self.reply = reply
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        text = self.reply
        if text is None:
            first_id = user.split("[", 1)[1].split("]", 1)[0]
            text = f"Here is the answer [{first_id}]."
        return LLMResult(text=text, prompt_tokens=self.prompt_tokens,
                         completion_tokens=self.completion_tokens, model="fake-model")


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store() -> VectorStore:
    return VectorStore.in_memory()


@pytest.fixture
def rag(embedder, store, fake_llm) -> RAGService:
    return RAGService(embedder=embedder, store=store, llm=fake_llm, chunk_size=200, chunk_overlap=40, top_k=3)


@pytest.fixture
def client(rag):
    app.dependency_overrides[get_rag_service] = lambda: rag
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

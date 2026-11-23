"""Generation step with the LLM mocked: prompt building, OpenAI client wrapper, citation parsing."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app import llm as llm_module
import httpx
import openai

from app.llm import NO_ANSWER, SYSTEM_PROMPT, LLMError, OpenAIClient, build_user_prompt
from app.rag import NO_CONTEXT_ANSWER, RAGService
from app.vector_store import RetrievedChunk
from tests.conftest import FakeLLM

CHUNKS = [
    RetrievedChunk(chunk_id="policy-ab12cd34::0", text="Refunds are accepted within 30 days.", source="policy.txt", score=0.9),
    RetrievedChunk(chunk_id="policy-ab12cd34::1", text="Refunds go to the original card.", source="policy.txt", score=0.7),
]


# --- prompt template --------------------------------------------------------

def test_user_prompt_contains_question_and_tagged_chunks():
    prompt = build_user_prompt("What is the refund window?", CHUNKS)
    assert "Question: What is the refund window?" in prompt
    assert "[policy-ab12cd34::0]\nRefunds are accepted within 30 days." in prompt
    assert "[policy-ab12cd34::1]" in prompt


def test_system_prompt_asks_for_grounded_cited_answers():
    assert "ONLY" in SYSTEM_PROMPT
    assert "cite" in SYSTEM_PROMPT.lower()
    assert NO_ANSWER in SYSTEM_PROMPT  # the model is told the exact "not found" reply


# --- OpenAIClient with the SDK mocked ---------------------------------------

def _fake_completion(text="Answer [x::0]", prompt_tokens=120, completion_tokens=30):
    return SimpleNamespace(
        model="gpt-4o-mini-2024-07-18",
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


@pytest.fixture
def mock_openai(monkeypatch):
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = _fake_completion()
    monkeypatch.setattr("openai.OpenAI", MagicMock(return_value=sdk))
    return sdk


def test_openai_client_requires_api_key():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient(api_key=None, model="gpt-4o-mini")


def test_openai_client_sends_system_and_user_messages(mock_openai):
    OpenAIClient(api_key="sk-test", model="gpt-4o-mini").complete("SYS", "USER")
    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0
    assert kwargs["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]


def test_openai_client_returns_text_tokens_and_model(mock_openai):
    result = OpenAIClient(api_key="sk-test", model="gpt-4o-mini").complete("s", "u")
    assert result.text == "Answer [x::0]"
    assert (result.prompt_tokens, result.completion_tokens) == (120, 30)
    assert result.model == "gpt-4o-mini-2024-07-18"


def test_openai_client_handles_missing_usage_and_content(mock_openai):
    resp = _fake_completion(text=None)
    resp.usage = None
    mock_openai.chat.completions.create.return_value = resp
    result = OpenAIClient(api_key="sk-test", model="m").complete("s", "u")
    assert result.text == ""
    assert result.prompt_tokens == result.completion_tokens == 0


_REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(cls, code):
    return cls("boom", response=httpx.Response(code, request=_REQ), body=None)


@pytest.mark.parametrize("error,expected", [
    (lambda: _status_error(openai.AuthenticationError, 401), "API key"),
    (lambda: _status_error(openai.RateLimitError, 429), "rate limit"),
    (lambda: openai.RateLimitError("You exceeded your current quota", response=httpx.Response(429, request=_REQ),
                                   body={"code": "insufficient_quota"}), "no credit left"),
    (lambda: _status_error(openai.NotFoundError, 404), "model"),
    (lambda: openai.APIConnectionError(request=_REQ), "reach OpenAI"),
    (lambda: _status_error(openai.InternalServerError, 500), "try again"),
])
def test_openai_errors_become_friendly_llm_errors(mock_openai, error, expected):
    mock_openai.chat.completions.create.side_effect = error()
    with pytest.raises(LLMError, match=expected):
        OpenAIClient(api_key="sk-test", model="gpt-4o-mini").complete("s", "u")


# --- RAGService.ask with a fake LLM -----------------------------------------

@pytest.fixture
def loaded_rag(rag):
    rag.ingest("refunds.txt", b"Customers can request a refund within 30 days of purchase.")
    rag.ingest("shipping.txt", b"Standard shipping takes 5 to 7 business days.")
    return rag


def test_ask_sends_retrieved_chunks_to_llm(loaded_rag, fake_llm):
    loaded_rag.ask("How long is the refund window?")
    system, user = fake_llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "How long is the refund window?" in user
    assert "refund within 30 days" in user


def test_ask_returns_citations_usage_and_timings(loaded_rag):
    ans = loaded_rag.ask("How long is the refund window?", top_k=1)
    assert len(ans.citations) == 1
    assert ans.citations[0] == ans.sources[0].chunk_id
    assert (ans.prompt_tokens, ans.completion_tokens, ans.model) == (100, 20, "fake-model")
    assert ans.retrieval_ms >= 0 and ans.llm_ms >= 0


def test_ask_drops_citations_that_were_not_retrieved(loaded_rag, fake_llm):
    real_id = loaded_rag.retrieve("refund", top_k=1)[0].chunk_id
    fake_llm.reply = f"Within 30 days [{real_id}] [invented-doc::7] [{real_id}]."
    ans = loaded_rag.ask("refund", top_k=1)
    assert ans.citations == [real_id]  # made-up ID dropped, duplicate collapsed


def test_ask_with_no_documents_skips_the_llm(rag, fake_llm):
    ans = rag.ask("Anything?")
    assert ans.answer == NO_CONTEXT_ANSWER
    assert ans.citations == [] and fake_llm.calls == []


def test_ask_without_llm_raises(embedder, store):
    rag = RAGService(embedder=embedder, store=store, llm=None)
    rag.ingest("a.txt", b"some text")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        rag.ask("question")


# --- provider settings (Groq by default, OpenAI optional) -------------------

def _settings(**env):
    from app.config import Settings

    return Settings(_env_file=None, **env)


def test_groq_is_the_default_provider():
    c = _settings(groq_api_key="gsk-test").llm
    assert c["provider"] == "Groq" and c["api_key"] == "gsk-test" and c["key_env"] == "GROQ_API_KEY"
    assert c["base_url"] == "https://api.groq.com/openai/v1"
    assert c["model"] == "openai/gpt-oss-120b"


def test_openai_provider_uses_its_own_key_and_default_url():
    c = _settings(llm_provider="openai", openai_api_key="sk-test", groq_api_key="gsk-ignored").llm
    assert c["provider"] == "OpenAI" and c["api_key"] == "sk-test"
    assert c["base_url"] is None and c["model"] == "gpt-4o-mini"


def test_generic_overrides_win():
    c = _settings(llm_api_key="k", llm_base_url="http://localhost:11434/v1", llm_model="m").llm
    assert (c["api_key"], c["base_url"], c["model"]) == ("k", "http://localhost:11434/v1", "m")


def test_client_passes_base_url_and_names_provider_in_errors(monkeypatch):
    ctor = MagicMock()
    monkeypatch.setattr("openai.OpenAI", ctor)
    OpenAIClient(api_key="gsk", model="llama", base_url="https://api.groq.com/openai/v1", provider="Groq")
    assert ctor.call_args.kwargs["base_url"] == "https://api.groq.com/openai/v1"
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        OpenAIClient(api_key=None, model="llama", provider="Groq", key_env="GROQ_API_KEY")


def test_unknown_model_error_suggests_available_models(mock_openai):
    mock_openai.chat.completions.create.side_effect = _status_error(openai.NotFoundError, 404)
    mock_openai.models.list.return_value = SimpleNamespace(data=[
        SimpleNamespace(id="whisper-large-v3"), SimpleNamespace(id="openai/gpt-oss-20b"),
        SimpleNamespace(id="openai/gpt-oss-120b")])
    with pytest.raises(LLMError, match="LLM_MODEL=openai/gpt-oss-120b") as exc:
        OpenAIClient(api_key="gsk", model="llama-old", provider="Groq").complete("s", "u")
    assert "whisper" not in str(exc.value)

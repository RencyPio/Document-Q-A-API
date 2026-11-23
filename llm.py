"""Prompt template and the LLM call (any OpenAI-compatible API: Groq by default, or OpenAI)."""
from dataclasses import dataclass
from typing import Protocol

from .vector_store import RetrievedChunk

NO_ANSWER = "I couldn't find the answer to that in your documents."


class LLMError(Exception):
    """The LLM call failed. The message is written for the end user."""


SYSTEM_PROMPT = (
    "You answer questions using ONLY the context passages provided. "
    "Each passage starts with its chunk ID in square brackets. "
    "After every claim, cite the chunk ID(s) it came from, like [report.pdf::3]. "
    f"If the context does not contain the answer, reply exactly: \"{NO_ANSWER}\" "
    "Do not use outside knowledge. Write clear, short sentences a non-expert can follow."
)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> LLMResult: ...


class OpenAIClient:
    """Chat client for any OpenAI-compatible API. Groq works by passing its base_url."""

    def __init__(self, api_key: str | None, model: str, base_url: str | None = None,
                 provider: str = "OpenAI", key_env: str = "OPENAI_API_KEY"):
        from openai import OpenAI

        if not api_key:
            raise RuntimeError(
                f"Answers need a {provider} API key. Add {key_env} to your .env file and restart the server."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.provider = provider
        self.key_env = key_env

    def _suggest_models(self) -> str:
        """Models get retired over time, so on a 'model not found' error, list what is available now."""
        try:
            ids = sorted(m.id for m in self.client.models.list().data)
        except Exception:
            return "Set LLM_MODEL in your .env file to a model your provider currently offers."
        skip = ("whisper", "tts", "guard", "embed", "orpheus", "playai")
        chat = [i for i in ids if not any(w in i.lower() for w in skip)][:5]
        if not chat:
            return "Set LLM_MODEL in your .env file to a model your provider currently offers."
        return f"Set LLM_MODEL in your .env file to one it offers now, e.g. LLM_MODEL={chat[0]} (others: {', '.join(chat[1:]) or 'none'})."

    def complete(self, system: str, user: str) -> LLMResult:
        import openai

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except openai.AuthenticationError as e:
            raise LLMError(f"{self.provider} rejected the API key. Check {self.key_env} in your .env file.") from e
        except openai.RateLimitError as e:
            # 429 means one of two things: no credit left (OpenAI), or too many requests (any provider).
            if getattr(e, "code", None) == "insufficient_quota" or "insufficient_quota" in str(e):
                raise LLMError(f"Your {self.provider} account has no credit left. Add credit on its billing "
                               "page, or switch to Groq's free tier (LLM_PROVIDER=groq).") from e
            raise LLMError(f"{self.provider}'s rate limit was reached (too many requests at once). "
                           "Wait a minute and try again.") from e
        except openai.NotFoundError as e:
            raise LLMError(f"{self.provider} doesn't recognise the model \"{self.model}\". "
                           f"{self._suggest_models()}") from e
        except openai.APIConnectionError as e:
            raise LLMError(f"Couldn't reach {self.provider}. Check your internet connection and try again.") from e
        except openai.APIError as e:
            raise LLMError(f"{self.provider} had a problem answering. Please try again in a moment.") from e
        usage = resp.usage
        return LLMResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=resp.model,
        )

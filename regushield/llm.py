"""Groq chat completions with streaming, time-to-first-token and token usage."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol


class LLMError(Exception):
    """The LLM call failed; the message is safe to show to the user."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Completion:
    """Iterate to receive text deltas; afterwards ``text``, ``usage`` and ``ttft_ms`` are set."""

    def __init__(self, deltas: Iterator[tuple[str, Usage | None]], prompt_chars: int) -> None:
        self._deltas = deltas
        self._prompt_chars = prompt_chars
        self._started = time.perf_counter()
        self.text = ""
        self.usage: Usage | None = None
        self.ttft_ms: float | None = None

    def __iter__(self) -> Iterator[str]:
        for delta, usage in self._deltas:
            if usage is not None:
                self.usage = usage
            if delta:
                if self.ttft_ms is None:
                    self.ttft_ms = (time.perf_counter() - self._started) * 1000
                self.text += delta
                yield delta
        if self.usage is None:  # provider sent no usage: rough estimate, flagged as such
            self.usage = Usage(self._prompt_chars // 4, len(self.text) // 4, estimated=True)


class ChatModel(Protocol):
    model: str

    def stream(self, messages: list[dict[str, str]]) -> Completion: ...


def _usage_from(chunk: Any) -> Usage | None:
    for usage in (
        getattr(chunk, "usage", None),
        getattr(getattr(chunk, "x_groq", None), "usage", None),
    ):
        if usage is not None and getattr(usage, "prompt_tokens", None) is not None:
            return Usage(int(usage.prompt_tokens), int(usage.completion_tokens or 0))
    return None


class GroqChat:
    def __init__(self, api_key: str, model: str, reasoning_effort: str | None = "low") -> None:
        from groq import Groq

        self.model = model
        self._client = Groq(api_key=api_key, timeout=60.0, max_retries=1)
        # Only the gpt-oss family accepts reasoning_effort; other models reject the parameter.
        self._extra = (
            {"reasoning_effort": reasoning_effort}
            if "gpt-oss" in model and reasoning_effort
            else {}
        )

    def stream(self, messages: list[dict[str, str]]) -> Completion:
        prompt_chars = sum(len(m["content"]) for m in messages)
        return Completion(self._deltas(messages), prompt_chars)

    def _deltas(self, messages: list[dict[str, str]]) -> Iterator[tuple[str, Usage | None]]:
        import groq

        try:
            stream = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0, stream=True, **self._extra
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                yield delta or "", _usage_from(chunk)
        except groq.AuthenticationError as exc:
            raise LLMError("GROQ_API_KEY tidak valid atau sudah dicabut.") from exc
        except groq.RateLimitError as exc:
            raise LLMError("Batas request Groq tercapai. Coba lagi sebentar lagi.") from exc
        except groq.NotFoundError as exc:
            raise LLMError(f"Model '{self.model}' tidak tersedia di Groq.") from exc
        except groq.APITimeoutError as exc:
            raise LLMError("Groq tidak merespons (timeout).") from exc
        except groq.APIConnectionError as exc:
            raise LLMError("Tidak dapat terhubung ke Groq.") from exc
        except groq.APIStatusError as exc:
            raise LLMError(f"Groq mengembalikan error HTTP {exc.status_code}.") from exc

from typing import Protocol


class LLMProviderError(Exception):
    pass


class LLMProvider(Protocol):
    def complete_json(self, prompt: str, payload: dict) -> dict: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...

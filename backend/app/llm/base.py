from typing import Protocol


class LLMProviderError(Exception):
    pass


class LLMProvider(Protocol):
    async def complete_json(self, prompt: str, payload: dict, *, agent: str, max_output_tokens: int) -> dict: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...

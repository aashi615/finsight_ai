from typing import Protocol


class LLMProviderError(Exception):
    def __init__(self, message: str, *, category: str = "llm_provider_error"):
        super().__init__(message)
        self.category = category


class LLMProvider(Protocol):
    async def complete_json(self, prompt: str, payload: dict, *, agent: str, max_output_tokens: int) -> dict: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...

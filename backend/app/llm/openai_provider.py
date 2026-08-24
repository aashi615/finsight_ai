import json
import httpx
from app.core.config import settings
from app.llm.base import LLMProviderError


class OpenAIProvider:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else settings.openai_api_key

    def _post(self, path: str, payload: dict) -> dict:
        if not self.api_key:
            raise LLMProviderError("LLM provider is not configured.")
        try:
            response = httpx.post(f"https://api.openai.com/v1{path}", headers={"Authorization": f"Bearer {self.api_key}"}, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError
            return data
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMProviderError("LLM provider is unavailable or returned malformed data.") from exc

    def complete_json(self, prompt: str, payload: dict) -> dict:
        data = self._post("/chat/completions", {"model": settings.openai_model, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload, default=str)}]})
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError
            return parsed
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMProviderError("LLM provider returned malformed structured output.") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/embeddings", {"model": settings.openai_embedding_model, "input": texts})
        try:
            vectors = [item["embedding"] for item in data["data"]]
            if len(vectors) != len(texts) or not all(isinstance(vector, list) for vector in vectors):
                raise ValueError
            return vectors
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMProviderError("LLM provider returned malformed embeddings.") from exc

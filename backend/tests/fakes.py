from datetime import datetime, timezone
from decimal import Decimal

from app.llm.base import LLMProviderError
from app.providers.base import CompanyData, MarketBarData, NewsArticleData


class FakeLLMProvider:
    """Deterministic provider used by Day 3 tests; it never performs network I/O."""

    def __init__(self, *, malformed: bool = False, fail: bool = False):
        self.malformed = malformed
        self.fail = fail
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise LLMProviderError("fake provider failure")
        return [[float(len(text)), float(sum(map(ord, text)) % 97), 1.0] for text in texts]

    async def complete_json(self, prompt: str, payload: dict, *, agent: str, max_output_tokens: int) -> dict:
        if self.fail:
            raise LLMProviderError("fake provider failure")
        self.calls.append(prompt)
        if self.malformed:
            return {"not": "an analysis"}
        evidence = payload.get("evidence") or payload.get("allowed_evidence") or []
        claim = [{"claim": "Based on available data, this is a monitored signal.", "evidence": evidence[:1]}] if evidence else []
        if "market analyst" in prompt:
            return {"agent": "market_analyst", "summary": "Recent prices were reviewed.", "metrics": {}, "signals": claim, "evidence": evidence}
        if "news analyst" in prompt:
            return {"agent": "news_analyst", "summary": "Recent news was reviewed.", "themes": ["company update"], "signals": claim, "evidence": evidence}
        if "document analyst" in prompt:
            return {"agent": "document_rag_agent", "summary": "Relevant tenant documents were reviewed.", "findings": claim, "evidence": evidence}
        market_summary = payload.get("market", {}).get("summary", "")
        market_analysis = "Historical market-price data is unavailable; no historical price performance was assessed." if "Historical market-price data is unavailable" in market_summary else "Market data suggests recent movement."
        return {"executive_summary": "Based on available data, the evidence indicates monitored developments.", "company_overview": "Company overview is based on available data.", "market_analysis": market_analysis, "news_analysis": "News coverage indicates recent themes.", "key_risks": claim, "key_opportunities": claim, "evidence": evidence, "confidence": 0.6, "generated_at": datetime.now(timezone.utc).isoformat()}


class FakeResearchProvider:
    def get_company(self, ticker: str) -> CompanyData:
        return CompanyData(ticker=ticker, name="NVIDIA Corporation", exchange="NASDAQ", country="US", sector="Technology", industry="Semiconductors")

    def get_market_data(self, ticker, from_date, to_date):
        return [
            MarketBarData(timestamp=datetime.combine(from_date, datetime.min.time(), tzinfo=timezone.utc), open=Decimal("100"), high=Decimal("110"), low=Decimal("99"), close=Decimal("105"), volume=1000, source="fake-market"),
            MarketBarData(timestamp=datetime.combine(to_date, datetime.min.time(), tzinfo=timezone.utc), open=Decimal("105"), high=Decimal("115"), low=Decimal("104"), close=Decimal("112"), volume=1500, source="fake-market"),
        ]

    def get_news(self, ticker, from_date, to_date):
        return [NewsArticleData(title="NVIDIA update", summary="A notable update.", url="https://example.test/nvidia-update", published_at=datetime.combine(to_date, datetime.min.time(), tzinfo=timezone.utc), source_name="Fake News", author=None)]

import time
from collections import defaultdict, deque
from threading import Lock
from fastapi import Depends
from app.api.deps import get_current_user
from app.core.config import settings
from app.core.exceptions import api_error
from app.models.user import User


class InMemoryResearchLimiter:
    """Per-process guard for expensive requests; not a distributed quota system."""
    def __init__(self):
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            entries = self._requests[key]
            while entries and now - entries[0] >= 60:
                entries.popleft()
            if len(entries) >= settings.research_rate_limit_per_minute:
                raise api_error(429, "RESEARCH_RATE_LIMITED", "Too many research requests. Please try again shortly.")
            entries.append(now)


limiter = InMemoryResearchLimiter()


def limit_research(current_user: User = Depends(get_current_user)) -> User:
    limiter.check(str(current_user.organization_id))
    return current_user

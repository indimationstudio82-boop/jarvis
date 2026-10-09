"""Unit tests for the search_web tool.

They pin two behaviors: the blocking search call never runs on the event
loop, and search backend failures (timeouts, rate limits, unexpected errors)
return a graceful fallback message instead of raising. Network behavior of the
real backend is intentionally not exercised here.
"""

import threading

from ddgs.exceptions import RatelimitException, TimeoutException

import search_backend
import tools
from tools import search_web


async def test_search_runs_off_the_event_loop(monkeypatch) -> None:
    """The synchronous search must execute in a worker thread, not the loop."""
    seen_threads: list[threading.Thread] = []

    def fake_search(query: str) -> str:
        seen_threads.append(threading.current_thread())
        return f"results for {query}"

    monkeypatch.setattr(tools, "_search_blocking", fake_search)

    result = await search_web(None, "livekit agents")

    assert result == "results for livekit agents"
    assert seen_threads, "search was never invoked"
    assert seen_threads[0] is not threading.main_thread()


async def test_search_blocking_delegates_to_search_backend(monkeypatch) -> None:
    """The throttle, retries and headers all live in search_backend.search."""
    calls: list[str] = []

    def fake_search(query: str) -> str:
        calls.append(query)
        return "canned results"

    monkeypatch.setattr(search_backend, "search", fake_search)

    assert await search_web(None, "some query") == "canned results"
    assert calls == ["some query"]


async def test_rate_limit_returns_graceful_fallback(monkeypatch) -> None:
    def fake_search(query: str) -> str:
        raise RatelimitException("429 Too Many Requests")

    monkeypatch.setattr(tools, "_search_blocking", fake_search)

    result = await search_web(None, "some query")

    assert isinstance(result, str)
    assert "unavailable" in result
    assert "rate-limiting" in result


async def test_timeout_returns_graceful_fallback(monkeypatch) -> None:
    def fake_search(query: str) -> str:
        raise TimeoutException("request timed out")

    monkeypatch.setattr(tools, "_search_blocking", fake_search)

    result = await search_web(None, "some query")

    assert isinstance(result, str)
    assert "unavailable" in result
    assert "timed out" in result


async def test_unexpected_error_returns_graceful_fallback(monkeypatch) -> None:
    def fake_search(query: str) -> str:
        raise ValueError("boom")

    monkeypatch.setattr(tools, "_search_blocking", fake_search)

    result = await search_web(None, "some query")

    assert isinstance(result, str)
    assert "unavailable" in result
    assert "boom" in result

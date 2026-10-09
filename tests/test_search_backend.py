"""Unit tests for the search backend (src/search_backend.py).

They pin the behaviors that keep the ``search_web`` tool working when search
engines start rate-limiting: consecutive searches are spaced 3-5 seconds apart,
transient failures are retried across providers with backoff, and every request
carries browser-like headers. Real network behavior is not exercised here.
"""

import time

import pytest
from ddgs.exceptions import DDGSException, TimeoutException

import search_backend


@pytest.fixture(autouse=True)
def patch_pacing(monkeypatch):
    """Record sleeps instead of taking them, and reset the throttle state."""
    sleeps: list[float] = []
    monkeypatch.setattr(search_backend, "_sleep", sleeps.append)
    monkeypatch.setattr(search_backend, "_last_search_at", None)
    return sleeps


@pytest.fixture(autouse=True)
def stable_attempts(monkeypatch):
    """Make the retry loop deterministic no matter what the environment sets."""
    monkeypatch.setattr(search_backend, "_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(search_backend, "_TOTAL_BUDGET_SECONDS", 60.0)
    monkeypatch.setattr(search_backend, "_BACKENDS", "yahoo,duckduckgo")


@pytest.fixture
def no_gap(monkeypatch):
    """Zero the inter-search throttle so retry tests record only backoffs."""
    monkeypatch.setattr(search_backend, "_MIN_GAP_SECONDS", 0.0)
    monkeypatch.setattr(search_backend, "_MAX_GAP_SECONDS", 0.0)


# --------------------------------------------------------------------------- #
# Throttle
# --------------------------------------------------------------------------- #


def test_gap_defaults_to_three_to_five_seconds() -> None:
    assert search_backend._MIN_GAP_SECONDS == 3.0
    assert search_backend._MAX_GAP_SECONDS == 5.0


def test_first_search_is_not_delayed(patch_pacing) -> None:
    search_backend._wait_for_slot()
    assert patch_pacing == []


def test_consecutive_search_waits_the_gap(patch_pacing) -> None:
    # Pretend a search just happened and confirm the next one waits ~3-5s,
    # which is the rate-limit window the engines enforce per client.
    search_backend._last_search_at = time.monotonic()
    search_backend._wait_for_slot()
    assert len(patch_pacing) == 1
    assert 2.9 <= patch_pacing[0] <= 5.1


# --------------------------------------------------------------------------- #
# Retries
# --------------------------------------------------------------------------- #


def _fail_then(monkeypatch, *errors) -> dict:
    """Fake ``_fetch`` that raises each error in turn, then returns a result."""
    state = {"calls": []}
    failures = list(errors)

    def fake_fetch(query: str, backend: str) -> list[dict]:
        state["calls"].append(backend)
        if failures:
            raise failures.pop(0)
        return [{"title": "T", "href": "https://x", "body": "bing bong"}]

    monkeypatch.setattr(search_backend, "_fetch", fake_fetch)
    return state


def test_retry_tries_each_backend_in_turn(monkeypatch, patch_pacing, no_gap) -> None:
    state = _fail_then(
        monkeypatch,
        TimeoutException("slow"),
        TimeoutException("slow"),
        TimeoutException("slow"),
    )
    with pytest.raises(TimeoutException):
        search_backend.search("q")

    assert state["calls"] == ["yahoo", "duckduckgo", "yahoo"]
    # Exponential backoff between attempts: 3s then 9s.
    assert patch_pacing == [3.0, 9.0]


def test_retry_recovers_after_transient_failure(monkeypatch, no_gap) -> None:
    state = _fail_then(monkeypatch, TimeoutException("slow"))
    assert search_backend.search("q") == "bing bong"
    assert state["calls"] == ["yahoo", "duckduckgo"]


def test_no_results_retried_then_raised(monkeypatch, no_gap) -> None:
    state = {"calls": []}

    def fake_fetch(query: str, backend: str) -> list[dict]:
        state["calls"].append(backend)
        return []

    monkeypatch.setattr(search_backend, "_fetch", fake_fetch)
    with pytest.raises(DDGSException, match="no results"):
        search_backend.search("q")
    assert state["calls"] == ["yahoo", "duckduckgo", "yahoo"]


def test_total_budget_stops_further_retries(monkeypatch, patch_pacing, no_gap) -> None:
    _fail_then(monkeypatch, TimeoutException("slow"), TimeoutException("slow"))
    monkeypatch.setattr(search_backend, "_TOTAL_BUDGET_SECONDS", 0.0)

    with pytest.raises(TimeoutException):
        search_backend.search("q")

    # The budget is already spent, so only the first attempt runs.
    assert patch_pacing == []


# --------------------------------------------------------------------------- #
# Headers
# --------------------------------------------------------------------------- #


def test_declared_user_agent_is_browser_like() -> None:
    ua = search_backend._BROWSER_HEADERS["user-agent"]
    assert ua.startswith("Mozilla/5.0")
    assert "Chrome/" in ua
    assert "Windows NT" in ua
    assert "python" not in ua.casefold()


def test_search_requests_send_browser_headers() -> None:
    client = search_backend._BrowserHttpClient().client
    headers = dict(client.headers)

    assert headers["user-agent"].startswith("Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    assert "Chrome/" in headers["user-agent"]
    assert headers["accept"].startswith("text/html")
    assert headers["sec-fetch-mode"] == "navigate"
    ua = headers["user-agent"].casefold()
    assert not any(token in ua for token in ("python", "requests", "curl", "ddgs"))


def test_dropped_profile_falls_back_to_a_browser(monkeypatch) -> None:
    # primp retires Chrome versions between releases; when ours is gone the
    # client must still build and still send a realistic browser User-Agent.
    monkeypatch.setattr(search_backend, "_IMPERSONATE_PROFILE", "chrome_1")
    ua = dict(search_backend._new_client(None, 6, True).headers)["user-agent"]
    assert ua.startswith("Mozilla/5.0")

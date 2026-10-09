"""Resilient web search behind the ``search_web`` tool.

The original implementation forwarded queries to DuckDuckGo through langchain's
wrapper using the HTTP library's default headers. Engines treat that as
automation: they answer with CAPTCHA pages, HTTP 403/429, or connection
timeouts, and the tool has nothing left to report.

Four things keep the requests working:

* **Provider** - ``SEARCH_BACKENDS`` names the engines to try, one per attempt.
  The default order is Yahoo then DuckDuckGo: both answered automated queries
  from this host when Google replied 403, Brave 429 and Mojeek a CAPTCHA page.
* **Headers** - every request carries the desktop Chrome header set below on
  top of primp's browser impersonation, so the User-Agent, the sec-ch-ua hints
  and the TLS fingerprint all agree instead of looking like a scripted client.
* **Throttle** - :func:`_wait_for_slot` keeps 3-5 seconds between consecutive
  searches (``SEARCH_MIN_GAP_SECONDS``/``SEARCH_MAX_GAP_SECONDS``), which is
  what engines rate limit on.
* **Retries** - a timed out or rate limited attempt is retried on the next
  backend with exponential backoff, bounded by ``SEARCH_TOTAL_TIMEOUT_SECONDS``
  so a voice turn never stalls indefinitely.
"""

import logging
import os
import random
import threading
import time
from typing import Any

import primp
from ddgs import DDGS
from ddgs.base import BaseSearchEngine
from ddgs.exceptions import DDGSException
from ddgs.http_client import HttpClient
from dotenv import load_dotenv

logger = logging.getLogger("search_backend")

load_dotenv(".env.local")


# --------------------------------------------------------------------------- #
# Configuration (all optional, read once at import time)
# --------------------------------------------------------------------------- #


def _env_number(name: str, default: float) -> float:
    """Read a numeric env var, falling back to ``default`` when unset or invalid."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring %s=%r: not a number, using %s", name, raw, default)
        return default


# Providers in preference order; every attempt uses the next one in the list.
_BACKENDS = os.getenv("SEARCH_BACKENDS", "yahoo,duckduckgo")
# Seconds a single attempt may spend on the network before it counts as a timeout.
_TIMEOUT_SECONDS = _env_number("SEARCH_TIMEOUT_SECONDS", 6)
# How many attempts one search gets before it reports failure.
_MAX_ATTEMPTS = int(_env_number("SEARCH_MAX_ATTEMPTS", 3))
# Wall-clock ceiling for a whole search, retries included.
_TOTAL_BUDGET_SECONDS = _env_number("SEARCH_TOTAL_TIMEOUT_SECONDS", 20)
# Range for the pause between two consecutive searches, in seconds.
_MIN_GAP_SECONDS = _env_number("SEARCH_MIN_GAP_SECONDS", 3)
_MAX_GAP_SECONDS = _env_number("SEARCH_MAX_GAP_SECONDS", 5)
# Exponential backoff between attempts: 3s, then 9s on top of the pause above.
_BACKOFF_BASE_SECONDS = _env_number("SEARCH_BACKOFF_BASE_SECONDS", 3)

# Kept at five results and one year of age to match what langchain's
# DuckDuckGoSearchRun returned before this module replaced it.
_MAX_RESULTS = 5
_REGION = "wt-wt"
_TIME_LIMIT = "y"


# --------------------------------------------------------------------------- #
# Browser-like request headers
# --------------------------------------------------------------------------- #

# Current desktop Chrome on Windows. ddgs sends requests through primp, which
# impersonates the browser's TLS/HTTP2 fingerprint, so this User-Agent is paired
# with the profile the client is built with in _new_client below.
_BROWSER_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
    ),
    "accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "accept-language": "en-US,en;q=0.9",
    "upgrade-insecure-requests": "1",
    "sec-fetch-site": "none",
    "sec-fetch-mode": "navigate",
    "sec-fetch-user": "?1",
    "sec-fetch-dest": "document",
}

# primp profile the client is pinned to. It releases a new Chrome every few
# weeks and drops old ones, so _new_client falls back to primp's random profile
# when this name disappears from a future version.
_IMPERSONATE_PROFILE = "chrome_146"

# Headers primp derives from the profile it picked. They are skipped on that
# fallback so a Firefox or Safari profile never ends up claiming to be Chrome.
_PROFILE_HEADERS = ("user-agent", "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform")


def _new_client(
    proxy: str | None, timeout: float | None, verify: bool | str
) -> primp.Client:
    """Build a primp client that looks like a current desktop browser."""
    kwargs: dict[str, Any] = {
        "proxy": proxy,
        "timeout": timeout,
        "impersonate_os": "windows",
        "verify": verify if isinstance(verify, bool) else True,
        "ca_cert_file": verify if isinstance(verify, str) else None,
    }
    try:
        client = primp.Client(impersonate=_IMPERSONATE_PROFILE, **kwargs)
        headers = _BROWSER_HEADERS
    except primp.BuilderError:
        logger.warning(
            "primp no longer knows the %s profile; using its random browser profile instead",
            _IMPERSONATE_PROFILE,
        )
        client = primp.Client(impersonate="random", **kwargs)
        headers = {
            k: v for k, v in _BROWSER_HEADERS.items() if k not in _PROFILE_HEADERS
        }

    client.headers_update(headers)
    return client


class _BrowserHttpClient(HttpClient):
    """ddgs's HTTP client, built from :func:`_new_client` instead of ddgs's own.

    Only the constructor changes: ddgs hard-codes ``impersonate="random"``,
    which picks an unpredictable OS and browser per process and would leave the
    Chrome User-Agent above paired with, say, a Firefox fingerprint. Request
    handling and error translation are inherited unchanged.
    """

    def __init__(
        self,
        proxy: str | None = None,
        timeout: float | None = 10,
        *,
        verify: bool | str = True,
    ) -> None:
        self.client = _new_client(proxy, timeout, verify)


class BrowserDDGS(DDGS):
    """``DDGS`` whose search engines send the headers above.

    ddgs builds and caches its engines in ``_get_engines``, which is the only
    place that lets us swap the HTTP client they talk to. Duck-typed on purpose:
    engines only use ``http_client.request(...)`` and ``http_client.client``,
    both of which :class:`_BrowserHttpClient` provides.
    """

    def _get_engines(self, category: str, backend: str) -> list[BaseSearchEngine[Any]]:
        engines = super()._get_engines(category, backend)
        for engine in engines:
            if isinstance(engine.http_client, _BrowserHttpClient):
                continue
            engine.http_client = _BrowserHttpClient(
                proxy=self._proxy,
                timeout=self._timeout,
                verify=self._verify,
            )
            # BaseSearchEngine.__init__ applied the engine's own headers to the
            # client it constructed, so they have to be re-applied to ours.
            engine.http_client.client.headers_update(engine.headers_update)
        return engines


# --------------------------------------------------------------------------- #
# Throttle
# --------------------------------------------------------------------------- #

_gap_lock = threading.Lock()
_last_search_at: float | None = None


def _sleep(seconds: float) -> None:
    """Sleep; a function so tests can record delays without patching ``time``."""
    if seconds > 0:
        time.sleep(seconds)


def _wait_for_slot() -> None:
    """Block until this search may start: 3-5 seconds after the previous one.

    The first search of a session is not delayed at all, so an isolated lookup
    stays fast. The lock covers the sleep, which serializes concurrent callers
    instead of letting them fire at the same instant.
    """
    global _last_search_at

    with _gap_lock:
        if _last_search_at is not None:
            wait = (
                _last_search_at
                + random.uniform(_MIN_GAP_SECONDS, _MAX_GAP_SECONDS)
                - time.monotonic()
            )
            if wait > 0:
                logger.debug("throttling search for %.2fs", wait)
                _sleep(wait)
        _last_search_at = time.monotonic()


# --------------------------------------------------------------------------- #
# Searching
# --------------------------------------------------------------------------- #


def _fetch(query: str, backend: str) -> list[dict[str, Any]]:
    """Run one attempt against a single backend, bounded by its timeout.

    A fresh instance per attempt: ddgs caches parse state on the instance and
    its lxml parser is not safe to share between the agent's worker threads.
    """
    return BrowserDDGS(timeout=_TIMEOUT_SECONDS).text(
        query,
        backend=backend,
        max_results=_MAX_RESULTS,
        region=_REGION,
        safesearch="moderate",
        timelimit=_TIME_LIMIT,
    )


def _format_results(results: list[dict[str, Any]]) -> str:
    """Join result snippets into the plain text blob the tool hands to the LLM."""
    snippets = [str(result["body"]).strip() for result in results if result.get("body")]
    return " ".join(snippets)


def search(query: str) -> str:
    """Search the web for ``query`` and return the result snippets as text.

    Attempts one backend per try and rotates through ``SEARCH_BACKENDS`` while
    failures keep coming, waiting between tries: the inter-search throttle
    first, then exponential backoff on top of it, until the attempts run out or
    the total time budget is spent.

    Raises:
        DDGSException: The failure of the last attempt, once every attempt has
            failed, so callers can map it to a user-facing message.
    """
    backends = [name.strip() for name in _BACKENDS.split(",") if name.strip()] or [
        "yahoo"
    ]
    started = time.monotonic()
    last_error: DDGSException | None = None

    for attempt in range(max(_MAX_ATTEMPTS, 1)):
        backend = backends[attempt % len(backends)]
        _wait_for_slot()
        try:
            results = _fetch(query, backend)
        except DDGSException as exc:
            # Covers timeouts, rate limits, CAPTCHA pages and parse failures.
            last_error = exc
            logger.warning(
                "search attempt %d/%d via %s failed: %s",
                attempt + 1,
                _MAX_ATTEMPTS,
                backend,
                exc,
            )
        else:
            text = _format_results(results)
            if text:
                logger.info("search result for %r via %s: %s", query, backend, text)
                return text
            last_error = DDGSException(f"{backend} returned no results")
            logger.warning(
                "search attempt %d/%d via %s returned no results",
                attempt + 1,
                _MAX_ATTEMPTS,
                backend,
            )

        if attempt + 1 >= _MAX_ATTEMPTS:
            break
        remaining = _TOTAL_BUDGET_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            logger.warning(
                "giving up on %r: %ss budget spent", query, _TOTAL_BUDGET_SECONDS
            )
            break
        # The next attempt throttles itself, so only wait here when the
        # backoff is the longer of the two: rate limits need more than the gap.
        _sleep(min(_BACKOFF_BASE_SECONDS ** (attempt + 1), remaining))

    if last_error is None:  # only reachable when _MAX_ATTEMPTS was non-positive
        last_error = DDGSException("search did not run")
    raise last_error

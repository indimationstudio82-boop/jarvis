"""Tools exposed to the JARVIS LLM."""

import asyncio
import logging

from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from livekit.agents import RunContext, function_tool

# Pre-warm the heavy native modules at worker startup instead of on the first
# search call. search_backend pulls in `ddgs`, which lazy-loads its real
# implementation (which pulls in `lxml.html` and every search engine module) on
# the first `DDGS()` call, and that import used to run inside the first tool
# call, blocking the event loop for over 12 seconds. Importing it here moves
# that cost to process boot, before any session is served.
import search_backend

logger = logging.getLogger("tools")

_FALLBACK_TEMPLATE = (
    "Web search is currently unavailable: {reason}. Do not claim to have "
    "found any results. Tell Sir the search could not be completed right now "
    "and offer to try again later or to look it up with the browser tools."
)


def _search_blocking(query: str) -> str:
    """Run the synchronous web search; must not run on the event loop.

    The browser headers, the delay between consecutive searches and the retries
    on transient failures all live in search_backend.search.
    """
    return search_backend.search(query)


@function_tool
async def search_web(context: RunContext, query: str) -> str:
    """Use this tool to search the web for information related to the given query.

    Returns the search results as text, or a note that the search failed when
    the search backend times out or rate-limits the request.
    """
    try:
        # The search (including any first-use imports behind it) is fully
        # synchronous, so run it in a worker thread to keep the event loop
        # responsive while it waits on network I/O.
        result = await asyncio.to_thread(_search_blocking, query)
    except RatelimitException as e:
        logger.warning("Web search rate-limited for %r: %s", query, e)
        return _FALLBACK_TEMPLATE.format(
            reason="the search service is rate-limiting requests (too many requests)"
        )
    except TimeoutException as e:
        logger.warning("Web search timed out for %r: %s", query, e)
        return _FALLBACK_TEMPLATE.format(reason="the search service timed out")
    except DDGSException as e:
        logger.warning("Web search failed for %r: %s", query, e)
        return _FALLBACK_TEMPLATE.format(
            reason=f"the search service returned an error ({e})"
        )
    except Exception as e:
        logger.error("Unexpected error during web search for %r", query, exc_info=True)
        return _FALLBACK_TEMPLATE.format(reason=f"an unexpected error occurred ({e})")

    logger.info("Web search result for query '%s': %s", query, result)
    return result

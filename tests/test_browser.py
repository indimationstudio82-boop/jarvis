"""Unit tests for the browser layer.

These run against the mock browser (BROWSER_MOCK=1), so no Chromium is
launched. Browser behavior in real conversations is covered by the
simulations in scenarios.yaml.
"""

import pytest
from livekit.agents.llm import ToolError

from browser import close_browser, get_browser, normalize_url
from browser_tools import BROWSER_TOOLS


@pytest.fixture
async def mock_browser(monkeypatch):
    monkeypatch.setenv("BROWSER_MOCK", "1")
    await close_browser()
    browser = get_browser()
    yield browser
    await close_browser()


# --------------------------------------------------------------------- #
# URL normalization
# --------------------------------------------------------------------- #


def test_normalize_url_keeps_full_https_url() -> None:
    assert normalize_url("https://example.com/page?q=1") == (
        "https://example.com/page?q=1"
    )


def test_normalize_url_adds_https_to_bare_domain() -> None:
    assert normalize_url("example.com") == "https://example.com"
    assert (
        normalize_url("  docs.livekit.io/agents  ") == "https://docs.livekit.io/agents"
    )


def test_normalize_url_uses_http_for_local_hosts() -> None:
    assert normalize_url("localhost:3000") == "http://localhost:3000"
    assert normalize_url("127.0.0.1:8080/path") == "http://127.0.0.1:8080/path"


def test_normalize_url_rejects_search_phrases() -> None:
    with pytest.raises(ToolError):
        normalize_url("livekit agents docs")


def test_normalize_url_rejects_domainless_input() -> None:
    with pytest.raises(ToolError):
        normalize_url("intranet")


def test_normalize_url_rejects_empty_input() -> None:
    with pytest.raises(ToolError):
        normalize_url("   ")


def test_normalize_url_rejects_non_http_schemes() -> None:
    for url in ("file:///etc/passwd", "javascript:alert(1)", "ftp://x.com"):
        with pytest.raises(ToolError):
            normalize_url(url)


# --------------------------------------------------------------------- #
# Mock browser behavior
# --------------------------------------------------------------------- #


async def test_open_and_read_known_mock_page(mock_browser) -> None:
    result = await mock_browser.open_url("example.com")
    assert "Opened https://example.com" in result
    assert "Example Domain" in result

    snapshot = await mock_browser.snapshot()
    assert "Page: Example Domain" in snapshot
    assert "[e1] link" in snapshot


async def test_open_unknown_url_gets_generic_mock_page(mock_browser) -> None:
    result = await mock_browser.open_url("https://weather.example.com")
    assert "Mock content" in result
    assert "[e1] textbox" in result


async def test_click_follows_link_element(mock_browser) -> None:
    await mock_browser.open_url("example.com")
    result = await mock_browser.click("e1")
    assert "URL: https://www.iana.org/domains/example" in result


async def test_click_rejects_malformed_ref(mock_browser) -> None:
    await mock_browser.open_url("example.com")
    with pytest.raises(ToolError):
        await mock_browser.click("button-1; rm -rf /")


async def test_click_unknown_ref_suggests_snapshot(mock_browser) -> None:
    await mock_browser.open_url("example.com")
    with pytest.raises(ToolError, match="snapshot"):
        await mock_browser.click("e99")


async def test_type_and_submit_reports_enter(mock_browser) -> None:
    await mock_browser.open_url("https://news.example.com")
    result = await mock_browser.type_text("e1", "hello there", submit=True)
    assert "'hello there'" in result
    assert "Enter" in result


async def test_go_back_returns_previous_mock_page(mock_browser) -> None:
    await mock_browser.open_url("example.com")
    await mock_browser.open_url("https://weather.example.com")
    result = await mock_browser.go_back()
    assert "Went back to https://example.com" in result
    assert "Example Domain" in result


async def test_go_back_without_history_is_graceful(mock_browser) -> None:
    result = await mock_browser.go_back()
    assert "no previous page" in result


async def test_close_tab_shuts_down_mock_browser(mock_browser) -> None:
    await mock_browser.open_url("example.com")
    result = await mock_browser.close_tab()
    assert result == "Closed the browser."


async def test_run_js_is_mocked(mock_browser) -> None:
    result = await mock_browser.run_js("document.title")
    assert result.startswith("(mock)")


# --------------------------------------------------------------------- #
# Tool registration
# --------------------------------------------------------------------- #


def test_browser_tools_have_unique_ids() -> None:
    ids = [tool.id for tool in BROWSER_TOOLS]
    assert len(ids) == len(set(ids)) == 9

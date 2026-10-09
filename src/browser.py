"""Playwright browser lifecycle for the JARVIS agent.

BrowserManager owns a single lazily started Chromium instance per agent job.
It exposes page-level operations (navigate, snapshot, click, type, ...) and
leaves tool schemas and LLM-facing wording to browser_tools.py.

Set BROWSER_MOCK=1 (e.g. when running simulations or unit tests) to return
deterministic canned pages instead of launching a real browser.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from livekit.agents.llm import ToolError
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger("browser")

load_dotenv(".env.local")

# Snapshot of interactive elements + visible text, rendered for the LLM.
_SNAPSHOT_JS = r"""
() => {
  const SELECTOR = [
    "a[href]", "button", "input", "textarea", "select", "summary",
    "[role=button]", "[role=link]", "[role=tab]", "[role=menuitem]",
    "[role=checkbox]", "[role=radio]", "[role=switch]", "[role=option]",
    "[contenteditable=true]", "[onclick]",
  ].join(",");

  const labelOf = (el) => {
    let text = el.getAttribute("aria-label") || "";
    if (!text && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) {
      text = el.placeholder || el.name || el.id || "";
      if (el.type === "submit" || el.type === "button") text = el.value || text;
    }
    if (!text) text = el.innerText || el.title || "";
    if (!text && el.tagName === "A") text = el.getAttribute("href") || "";
    return (text || "").trim().replace(/\s+/g, " ").slice(0, 120);
  };

  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    if (style.visibility === "hidden" || style.display === "none") return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };

  const elements = [];
  let n = 0;
  for (const el of document.querySelectorAll(SELECTOR)) {
    if (!isVisible(el)) continue;
    n += 1;
    const ref = "e" + n;
    el.setAttribute("data-jarvis-ref", ref);
    const entry = { ref, tag: el.tagName.toLowerCase(), label: labelOf(el) };
    const role = el.getAttribute("role");
    if (role) entry.role = role;
    if (el.tagName === "A") entry.href = (el.getAttribute("href") || "").slice(0, 160);
    if (el.tagName === "INPUT") entry.type = el.type;
    if (el.disabled) entry.disabled = true;
    if (el.type === "checkbox" || el.type === "radio") entry.checked = !!el.checked;
    elements.push(entry);
  }

  return {
    url: location.href,
    title: document.title,
    elements: elements.slice(0, 80),
    element_count: elements.length,
    text: (document.body ? document.body.innerText : "").slice(0, 24000),
  };
}
"""

# Body text snapshot used to detect whether a submit action changed anything.
_BODY_TEXT_JS = (
    "() => document.body ? document.body.innerText.slice(0, 6000).trim() : ''"
)

_REF_RE = re.compile(r"^e\d{1,4}$")

# Text snapshot cap, in characters. Keeps tool results small enough for voice
# latency while still describing the page well.
SNAPSHOT_TEXT_LIMIT = 6000

# Canned pages used when BROWSER_MOCK=1 (simulations, unit tests, CI).
_MOCK_PAGES: dict[str, dict] = {
    "https://example.com": {
        "title": "Example Domain",
        "elements": [
            {
                "ref": "e1",
                "tag": "a",
                "role": "link",
                "label": "More information...",
                "href": "https://www.iana.org/domains/example",
            },
        ],
        "text": (
            "Example Domain\n\n"
            "This domain is for use in illustrative examples in documents. "
            "You may use this domain in literature without prior coordination "
            "or asking for permission.\n\n"
            "More information..."
        ),
    },
}


def normalize_url(raw: str) -> str:
    """Validate and normalize a URL supplied by the LLM.

    Raises:
        ToolError: If the input cannot be interpreted as an http(s) URL.
    """
    s = raw.strip()
    if not s:
        raise ToolError("The URL is empty. Provide a URL such as https://example.com.")
    if any(c.isspace() for c in s):
        raise ToolError(
            f"'{s}' looks like a search phrase, not a URL. Use the search web "
            "tool first to find the right page, then open its full URL."
        )

    if "://" in s:
        scheme = urlsplit(s).scheme.lower()
        if scheme not in ("http", "https"):
            raise ToolError(
                f"Only http and https URLs are supported, not '{scheme}://'."
            )
        return s

    host = s.split("/", 1)[0].split("?", 1)[0].split(":", 1)[0]
    if host not in ("localhost", "127.0.0.1"):
        if "." not in host:
            raise ToolError(
                f"'{s}' is missing a domain ending. Provide a full URL such "
                "as https://example.com, or use the search web tool to find it."
            )
        return "https://" + s
    return "http://" + s


def _format_snapshot(data: dict) -> str:
    """Render a snapshot payload as compact text for the LLM."""
    lines = [f"Page: {data.get('title') or '(untitled)'}", f"URL: {data.get('url')}"]

    elements = data.get("elements") or []
    total = data.get("element_count", len(elements))
    shown = (
        f"{len(elements)} of {total} shown"
        if total > len(elements)
        else str(len(elements))
    )
    lines.append("")
    lines.append(f"Interactive elements ({shown}):")
    if not elements:
        lines.append("(none found)")
    for el in elements:
        kind = el.get("role") or el["tag"]
        bits = [f"[{el['ref']}]", kind]
        if el.get("label"):
            bits.append(f'"{el["label"]}"')
        if el.get("href"):
            bits.append(f"-> {el['href']}")
        if el.get("type"):
            bits.append(f"(type={el['type']})")
        if el.get("disabled"):
            bits.append("(disabled)")
        if "checked" in el:
            bits.append("(checked)" if el["checked"] else "(unchecked)")
        lines.append(" ".join(bits))

    text = (data.get("text") or "").strip()
    lines.append("")
    lines.append("Visible text:")
    if not text:
        lines.append("(no visible text)")
    elif len(text) > SNAPSHOT_TEXT_LIMIT:
        lines.append(text[:SNAPSHOT_TEXT_LIMIT])
        lines.append("... (truncated, scroll or re-snapshot for more)")
    else:
        lines.append(text)

    return "\n".join(lines)


def _truncate(value: str, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"... (truncated, {len(value)} characters total)"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _profile_dir() -> Path:
    return Path.home() / ".jarvis" / "browser-profile"


class BrowserManager:
    """One Chromium instance per agent job, started on first use."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self._mock = _env_flag("BROWSER_MOCK", False)
        # Headless set to False by default so the browser opens visibly on screen
        self._headless = _env_flag("BROWSER_HEADLESS", False)
        try:
            self._timeout_ms = int(os.getenv("BROWSER_TIMEOUT_MS", "15000"))
        except ValueError:
            self._timeout_ms = 15000

        self._pw = None
        self._context = None

        # Mock-mode state.
        self._mock_url: str | None = None
        self._mock_data: dict | None = None
        self._mock_history: list[str] = []

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def _ensure_started(self) -> Page:
        if self._mock:
            if self._mock_data is None:
                self._mock_url = "about:blank"
                self._mock_data = {
                    "title": "Blank",
                    "url": "about:blank",
                    "elements": [],
                    "text": "",
                }
            return None  # type: ignore[return-value]

        if self._context is None:
            logger.info("starting headless=%s chromium", self._headless)
            self._pw = await async_playwright().start()
            self._context = await self._pw.chromium.launch_persistent_context(
                user_data_dir=str(_profile_dir()),
                headless=self._headless,
                viewport={"width": 1280, "height": 800},
            )
            self._context.set_default_navigation_timeout(self._timeout_ms)
            if not self._context.pages:
                await self._context.new_page()
        return self._page()

    def _page(self) -> Page:
        pages = self._context.pages if self._context else []
        if not pages:
            raise ToolError("The browser has no open page. Use browser_open first.")
        return pages[-1]

    async def close(self) -> None:
        """Shut the browser down and release all resources."""
        async with self.lock:
            await self._shutdown_unlocked()

    async def _shutdown_unlocked(self) -> None:
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                logger.exception("error closing browser context")
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                logger.exception("error stopping playwright")
        self._pw = None
        self._context = None
        self._mock_url = None
        self._mock_data = None
        self._mock_history = []

    # ------------------------------------------------------------------ #
    # Real browser helpers
    # ------------------------------------------------------------------ #

    async def _render_snapshot(self) -> str:
        page = self._page()
        try:
            data = await page.evaluate(_SNAPSHOT_JS)
        except PlaywrightError as err:
            raise ToolError(f"Could not read the page: {err}") from err
        return _format_snapshot(data)

    async def _goto(self, url: str) -> str:
        page = self._page()
        try:
            await page.goto(
                url, wait_until="domcontentloaded", timeout=self._timeout_ms
            )
        except PlaywrightTimeoutError:
            logger.warning("navigation to %s timed out; showing partial content", url)
        except PlaywrightError as err:
            raise ToolError(f"Could not open {url}: {err}") from err
        # Give JS-heavy pages a short window to settle; not fatal if they don't.
        with contextlib.suppress(PlaywrightTimeoutError):
            await page.wait_for_load_state("networkidle", timeout=3000)
        return await self._render_snapshot()

    async def _after_action(self, action: str, page: Page, url_before: str) -> str:
        await self._settle(page, url_before)
        return await self._with_snapshot(action)

    async def _with_snapshot(self, action: str) -> str:
        return f"{action}\n\n{await self._render_snapshot()}"

    async def _settle(
        self, page: Page, url_before: str, expect_nav: bool = False
    ) -> None:
        """Let a navigation or DOM update finish before snapshotting."""
        if expect_nav or page.url != url_before:
            with contextlib.suppress(PlaywrightTimeoutError):
                await page.wait_for_load_state("networkidle", timeout=5000)
        else:
            await asyncio.sleep(0.3)

    # ------------------------------------------------------------------ #
    # Mock helpers
    # ------------------------------------------------------------------ #

    def _mock_snapshot(self) -> str:
        if self._mock_data is None:
            self._mock_url = "about:blank"
            self._mock_data = {
                "title": "Blank",
                "url": "about:blank",
                "elements": [],
                "text": "",
            }
        return _format_snapshot(self._mock_data)

    async def _mock_goto(self, url: str) -> str:
        self._mock_history.append(self._mock_url or "about:blank")
        self._mock_url = url
        page = dict(_MOCK_PAGES.get(url) or self._generic_mock_page(url))
        page["url"] = url
        self._mock_data = page
        return self._mock_snapshot()

    @staticmethod
    def _generic_mock_page(url: str) -> dict:
        return {
            "title": f"Mock page ({url})",
            "elements": [
                {"ref": "e1", "tag": "input", "role": "textbox", "label": "Search"},
                {"ref": "e2", "tag": "button", "role": "button", "label": "Search"},
            ],
            "text": (
                f"Mock content for {url}.\n\n"
                "This page is simulated because BROWSER_MOCK is enabled. "
                "It contains a search box and a Search button."
            ),
        }

    def _mock_element(self, ref: str) -> dict | None:
        if not self._mock_data:
            return None
        for el in self._mock_data.get("elements", []):
            if el["ref"] == ref:
                return el
        return None

    # ------------------------------------------------------------------ #
    # Public operations
    # ------------------------------------------------------------------ #

    async def open_url(self, raw_url: str) -> str:
        url = normalize_url(raw_url)
        async with self.lock:
            if self._mock:
                snapshot = await self._mock_goto(url)
            else:
                await self._ensure_started()
                snapshot = await self._goto(url)
            logger.info("browser opened %s", url)
            return _truncate(f"Opened {url}.\n\n{snapshot}", 2500)

    async def snapshot(self) -> str:
        async with self.lock:
            if self._mock:
                return self._mock_snapshot()
            await self._ensure_started()
            return await self._render_snapshot()

    async def click(self, ref: str) -> str:
        self._validate_ref(ref)
        async with self.lock:
            if self._mock:
                el = self._mock_element(ref)
                if el is None:
                    raise ToolError(
                        f"No element '{ref}' on the current mock page. "
                        "Take a fresh snapshot."
                    )
                href = el.get("href", "")
                if href.startswith("http"):
                    return await self._mock_goto(href)
                return await self._after_action_mock(
                    f"Clicked [{ref}] {el['label']!r}."
                )
            await self._ensure_started()
            page = self._page()
            url_before = page.url
            locator = page.locator(f'[data-jarvis-ref="{ref}"]')
            try:
                await locator.first.click(timeout=self._timeout_ms)
            except PlaywrightTimeoutError as err:
                raise ToolError(
                    f"Element '{ref}' was not clickable (it may be stale after a "
                    "page change). Take a fresh snapshot and try again."
                ) from err
            logger.info("browser clicked %s", ref)
            return await self._after_action(f"Clicked [{ref}].", page, url_before)

    async def _after_action_mock(self, action: str) -> str:
        return f"{action}\n\n{self._mock_snapshot()}"

    async def type_text(self, ref: str, text: str, submit: bool) -> str:
        self._validate_ref(ref)
        async with self.lock:
            if self._mock:
                el = self._mock_element(ref)
                if el is None:
                    raise ToolError(
                        f"No element '{ref}' on the current mock page. "
                        "Take a fresh snapshot."
                    )
                action = f"Typed {text!r} into [{ref}] {el['label']!r}."
                if submit:
                    action += " Pressed Enter to submit."
                return await self._after_action_mock(action)
            await self._ensure_started()
            page = self._page()
            url_before = page.url
            locator = page.locator(f'[data-jarvis-ref="{ref}"]')
            action = f"Typed {text!r} into [{ref}]."
            try:
                await locator.first.fill(text, timeout=self._timeout_ms)
                if submit:
                    text_before = await page.evaluate(_BODY_TEXT_JS)
                    action += " Pressed Enter to submit."
                    await locator.first.press("Enter")
            except PlaywrightTimeoutError as err:
                raise ToolError(
                    f"Could not type into '{ref}' (it may be stale after a page "
                    "change). Take a fresh snapshot and try again."
                ) from err
            logger.info("browser typed into %s", ref)
            if not submit:
                return await self._after_action(action, page, url_before)
            await self._settle(page, url_before, expect_nav=True)
            snapshot = await self._render_snapshot()
            if page.url == url_before and await page.evaluate(_BODY_TEXT_JS) == (
                text_before
            ):
                action += (
                    " WARNING: nothing on the page changed, so the form did not "
                    "submit. The element may be wrong; take a fresh snapshot and "
                    "use the form's visible submit or search button instead."
                )
            return f"{action}\n\n{snapshot}"

    async def press_key(self, key: str) -> str:
        async with self.lock:
            if self._mock:
                return await self._after_action_mock(f"Pressed the {key!r} key.")
            await self._ensure_started()
            page = self._page()
            url_before = page.url
            try:
                await page.keyboard.press(key)
            except PlaywrightError as err:
                raise ToolError(f"Could not press key '{key}': {err}") from err
            return await self._after_action(
                f"Pressed the {key!r} key.", page, url_before
            )

    async def go_back(self) -> str:
        async with self.lock:
            if self._mock:
                if not self._mock_history:
                    return await self._after_action_mock(
                        "There is no previous page to go back to."
                    )
                self._mock_url = self._mock_history.pop()
                page = dict(
                    _MOCK_PAGES.get(self._mock_url)
                    or self._generic_mock_page(self._mock_url)
                )
                page["url"] = self._mock_url
                self._mock_data = page
                return await self._after_action_mock(f"Went back to {self._mock_url}.")
            await self._ensure_started()
            page = self._page()
            try:
                response = await page.go_back(
                    wait_until="domcontentloaded", timeout=self._timeout_ms
                )
            except PlaywrightTimeoutError:
                return await self._with_snapshot(
                    "Going back timed out; the previous page is still loading."
                )
            if response is None:
                return await self._with_snapshot(
                    "There is no previous page to go back to."
                )
            return await self._with_snapshot(f"Went back to {page.url}.")

    async def close_tab(self) -> str:
        async with self.lock:
            if self._mock:
                await self._shutdown_unlocked()
                return "Closed the browser."
            await self._ensure_started()
            if len(self._context.pages) > 1:
                closed = self._page()
                await closed.close()
                return await self._with_snapshot("Closed the tab.")
            await self._shutdown_unlocked()
            return "Closed the browser."

    async def run_js(self, code: str) -> str:
        async with self.lock:
            if self._mock:
                return f"(mock) JavaScript result for {code[:80]!r}: null"
            await self._ensure_started()
            page = self._page()
            try:
                result = await page.evaluate(code)
            except PlaywrightError as err:
                raise ToolError(f"JavaScript failed: {err}") from err
            rendered = (
                result if isinstance(result, str) else json.dumps(result, default=str)
            )
            return _truncate(rendered)

    async def screenshot(self) -> str:
        async with self.lock:
            if self._mock:
                return "(mock) Screenshot skipped because BROWSER_MOCK is enabled."
            await self._ensure_started()
            page = self._page()
            directory = Path(tempfile.gettempdir()) / "jarvis-screenshots"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"page-{int(time.time() * 1000)}.png"
            try:
                await page.screenshot(path=str(path), full_page=False)
            except PlaywrightError as err:
                raise ToolError(f"Could not take a screenshot: {err}") from err
            return f"Screenshot saved to {path}"

    @staticmethod
    def _validate_ref(ref: str) -> None:
        if not _REF_RE.match(ref):
            raise ToolError(
                f"'{ref}' is not a valid element reference. Use a ref from the "
                "latest snapshot, such as e3."
            )


_manager: BrowserManager | None = None


def get_browser() -> BrowserManager:
    """Return the job's browser manager, creating it if needed."""
    global _manager
    if _manager is None:
        _manager = BrowserManager()
    return _manager


async def close_browser() -> None:
    """Shut down the job's browser, if one is running."""
    global _manager
    if _manager is not None:
        await _manager.close()
        _manager = None
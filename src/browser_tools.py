"""Browser tools exposed to the JARVIS LLM.

Each tool is a thin, voice-friendly wrapper around BrowserManager (browser.py).
Mutating actions that submit forms run with interruptions disabled so user
speech cannot cut them off halfway.
"""

import logging

from livekit.agents import RunContext, function_tool

from browser import get_browser

logger = logging.getLogger("browser_tools")


@function_tool()
async def browser_open(context: RunContext, url: str) -> str:
    """Open a website in the browser and report what loaded.

    Args:
        url: The full URL to open, such as "https://example.com". Requires a
            domain with an ending like .com; use search web first if the
            destination is unknown.
    """
    return await get_browser().open_url(url)


@function_tool()
async def browser_snapshot(context: RunContext) -> str:
    """Read the current page: its URL, title, interactive elements, and text.

    Returns interactive elements as references like e1, e2. Take a snapshot
    whenever you need to know what is on screen, especially after a page change.
    """
    return await get_browser().snapshot()


@function_tool()
async def browser_click(context: RunContext, ref: str) -> str:
    """Click the element with the given reference from the latest snapshot.

    Args:
        ref: The element reference, such as e3, exactly as shown in a snapshot.
    """
    return await get_browser().click(ref)


@function_tool()
async def browser_type(
    context: RunContext, ref: str, text: str, submit: bool = False
) -> str:
    """Type text into the input field with the given reference.

    Args:
        ref: The element reference, such as e2, exactly as shown in a snapshot.
        text: The text to type into the field.
        submit: Set true to press Enter after typing. Always confirm with Sir
            before submitting a form that sends or commits data.
    """
    if submit:
        context.disallow_interruptions()
    return await get_browser().type_text(ref, text, submit)


@function_tool()
async def browser_press_key(context: RunContext, key: str) -> str:
    """Press a keyboard key on the page, such as Enter, Tab, Escape, or ArrowDown.

    Args:
        key: The key name, for example "Enter", "Tab", "Escape", "ArrowDown",
            or a combination like "Control+a".
    """
    return await get_browser().press_key(key)


@function_tool()
async def browser_go_back(context: RunContext) -> str:
    """Go back to the previously viewed page."""
    return await get_browser().go_back()


@function_tool()
async def browser_close_tab(context: RunContext) -> str:
    """Close the current browser tab. Closes the browser entirely if it is the last tab."""
    return await get_browser().close_tab()


@function_tool()
async def browser_eval(context: RunContext, code: str) -> str:
    """Run JavaScript on the current page and return its result.

    Use only for reading page state or performing actions no other browser
    tool supports. Never use it to send data to external servers.

    Args:
        code: JavaScript to evaluate, for example "document.title".
    """
    return await get_browser().run_js(code)


@function_tool()
async def browser_screenshot(context: RunContext) -> str:
    """Take a screenshot of the current viewport and return the saved file path."""
    return await get_browser().screenshot()


BROWSER_TOOLS = [
    browser_open,
    browser_snapshot,
    browser_click,
    browser_type,
    browser_press_key,
    browser_go_back,
    browser_close_tab,
    browser_eval,
    browser_screenshot,
]

__all__ = [
    "BROWSER_TOOLS",
    "browser_click",
    "browser_close_tab",
    "browser_eval",
    "browser_go_back",
    "browser_open",
    "browser_press_key",
    "browser_screenshot",
    "browser_snapshot",
    "browser_type",
]

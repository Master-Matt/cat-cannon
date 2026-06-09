"""Lightweight Discord webhook text notifications.

This is intentionally dependency-free (stdlib ``urllib`` only) so the heartbeat
watchdog and the process guardian can post status messages without pulling in
the heavier event-video recorder.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request

_ALLOWED_HOSTS = {"discord.com", "discordapp.com"}


def validate_discord_webhook_url(webhook_url: str) -> str:
    """Return a normalized webhook URL or raise ``ValueError`` if it is unsafe."""
    parsed = urllib.parse.urlparse(webhook_url.strip())
    if parsed.scheme != "https":
        raise ValueError("Discord webhook URL must use https")
    hostname = (parsed.hostname or "").lower()
    if hostname not in _ALLOWED_HOSTS:
        raise ValueError("Discord webhook URL must point to discord.com")
    if not parsed.path.startswith("/api/webhooks/"):
        raise ValueError("Discord webhook URL must be an /api/webhooks/ URL")
    return urllib.parse.urlunparse(parsed)


def build_discord_message_request(
    *,
    webhook_url: str,
    content: str,
    username: str = "Cat Cannon",
) -> urllib.request.Request:
    """Build a JSON POST request that posts a plain-text Discord message."""
    safe_url = validate_discord_webhook_url(webhook_url)
    payload = {
        "content": content[:1900],
        "username": username,
        "allowed_mentions": {"parse": []},
    }
    body = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        safe_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "cat-cannon-heartbeat",
        },
    )


def post_discord_message(
    webhook_url: str,
    content: str,
    *,
    username: str = "Cat Cannon",
    timeout: float = 15.0,
    urlopen=urllib.request.urlopen,
) -> bool:
    """Post a text message to a Discord webhook.

    Returns ``True`` on success and ``False`` on any failure (network error,
    bad status, invalid URL). Never raises — notifications must not crash the
    watchdog that is trying to report a problem.
    """
    if not webhook_url.strip():
        return False
    try:
        request = build_discord_message_request(
            webhook_url=webhook_url,
            content=content,
            username=username,
        )
    except ValueError as exc:
        print(f"[notify] Discord message skipped: {exc}", flush=True)
        return False
    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 204))
            if 200 <= status < 300:
                return True
            print(f"[notify] Discord message failed with status {status}", flush=True)
            return False
    except (urllib.error.URLError, OSError) as exc:
        print(f"[notify] Discord message failed: {exc}", flush=True)
        return False


def post_discord_message_async(
    webhook_url: str,
    content: str,
    *,
    username: str = "Cat Cannon",
    urlopen=urllib.request.urlopen,
) -> threading.Thread | None:
    """Post a Discord message on a background thread; return the thread.

    Returns ``None`` immediately when no webhook is configured.
    """
    if not webhook_url.strip():
        return None
    thread = threading.Thread(
        target=post_discord_message,
        args=(webhook_url, content),
        kwargs={"username": username, "urlopen": urlopen},
        daemon=True,
    )
    thread.start()
    return thread

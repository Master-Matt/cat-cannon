import json

import pytest

from cat_cannon.app import notify


class _FakeResponse:
    def __init__(self, status=204):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capturing_urlopen(captured, status=204):
    def _urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(status)

    return _urlopen


def test_validate_rejects_non_https():
    with pytest.raises(ValueError):
        notify.validate_discord_webhook_url("http://discord.com/api/webhooks/1/abc")


def test_validate_rejects_foreign_host():
    with pytest.raises(ValueError):
        notify.validate_discord_webhook_url("https://evil.example/api/webhooks/1/abc")


def test_validate_rejects_non_webhook_path():
    with pytest.raises(ValueError):
        notify.validate_discord_webhook_url("https://discord.com/api/other/1/abc")


def test_validate_accepts_good_url():
    url = "https://discord.com/api/webhooks/123/token"
    assert notify.validate_discord_webhook_url(url) == url


def test_post_message_success_sends_json_payload():
    captured = {}
    ok = notify.post_discord_message(
        "https://discord.com/api/webhooks/123/token",
        "hello world",
        urlopen=_capturing_urlopen(captured),
    )
    assert ok is True
    body = json.loads(captured["request"].data.decode("utf-8"))
    assert body["content"] == "hello world"
    assert body["allowed_mentions"] == {"parse": []}


def test_post_message_empty_webhook_is_noop():
    called = {"n": 0}

    def _urlopen(request, timeout=None):
        called["n"] += 1
        return _FakeResponse()

    assert notify.post_discord_message("", "hi", urlopen=_urlopen) is False
    assert called["n"] == 0


def test_post_message_invalid_url_returns_false():
    assert notify.post_discord_message(
        "https://evil.example/api/webhooks/1/abc",
        "hi",
        urlopen=_capturing_urlopen({}),
    ) is False


def test_post_message_non_2xx_returns_false():
    assert notify.post_discord_message(
        "https://discord.com/api/webhooks/123/token",
        "hi",
        urlopen=_capturing_urlopen({}, status=500),
    ) is False


def test_post_message_network_error_returns_false():
    def _urlopen(request, timeout=None):
        raise OSError("boom")

    assert notify.post_discord_message(
        "https://discord.com/api/webhooks/123/token",
        "hi",
        urlopen=_urlopen,
    ) is False

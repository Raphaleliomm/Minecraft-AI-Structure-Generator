from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import scrape_structures
from scrape_structures import HttpClient


def test_retry_after_seconds_and_http_date():
    response = SimpleNamespace(headers={"Retry-After": "3"})
    assert HttpClient._retry_after(response, 0) == 3
    response.headers["Retry-After"] = format_datetime(
        datetime.now(timezone.utc) + timedelta(seconds=5), usegmt=True
    )
    assert 3 <= HttpClient._retry_after(response, 0) <= 5
    response.headers["Retry-After"] = format_datetime(
        datetime.now(timezone.utc) - timedelta(seconds=5), usegmt=True
    )
    assert HttpClient._retry_after(response, 0) == 0


def test_zero_retry_after_does_not_use_backoff(monkeypatch):
    client = HttpClient(delay=0, jitter=0, timeout=1, retries=1)
    failed = SimpleNamespace(status_code=429, headers={"Retry-After": "0"}, close=lambda: None)
    success = SimpleNamespace(status_code=200)
    responses = iter([failed, success])
    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: next(responses))
    sleeps = []
    monkeypatch.setattr(scrape_structures.time, "sleep", sleeps.append)
    assert client.get("https://example.com") is success
    assert sleeps == [0]


def test_short_retry_is_disabled_after_six_failures_in_last_100_requests(monkeypatch):
    client = HttpClient(delay=0, jitter=0, timeout=1, retries=2)
    sleeps = []
    monkeypatch.setattr(scrape_structures.time, "sleep", sleeps.append)

    def response(status):
        return SimpleNamespace(status_code=status, headers={}, close=lambda: None)

    # Each sequence fails after the 5-second retry, then succeeds after 10.
    for _ in range(6):
        responses = iter([response(429), response(429), response(200)])
        monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: next(responses))
        client.get("https://example.com")
    assert sleeps == [5, 10] * 6

    responses = iter([response(429), response(200)])
    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: next(responses))
    client.get("https://example.com")
    assert sleeps[-1] == 10

    # Successful requests eventually push the old failures out of the window.
    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: response(200))
    for _ in range(100):
        client.get("https://example.com")
    responses = iter([response(429), response(200)])
    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: next(responses))
    client.get("https://example.com")
    assert sleeps[-1] == 5

"""Errors raised while talking to the Mistral API."""

from __future__ import annotations

from importlib import import_module

from mistralai.client.errors import MistralError, NoResponseError


def _transport_errors() -> tuple[type[Exception], ...]:
    """Return the base transport error of every HTTP library that may be in play.

    Home Assistant's shared client and the mistralai SDK moved from httpx to
    httpx2 at different times, so a network failure can surface from either.
    """
    found: list[type[Exception]] = []
    for name in ("httpx", "httpx2"):
        try:
            error = import_module(name).HTTPError
        except ImportError:
            continue
        if error not in found:
            found.append(error)
    return tuple(found)


# Mistral answered with an error status. Carries status_code and message.
MistralAPIError = MistralError

# No usable answer at all: network failure, timeout or an empty response.
MISTRAL_CONNECTION_ERRORS: tuple[type[Exception], ...] = (
    NoResponseError,
    *_transport_errors(),
)

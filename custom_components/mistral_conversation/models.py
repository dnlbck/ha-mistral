"""Access to the Mistral model list with per-entry caching."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.core import callback
from homeassistant.util import dt as dt_util

from .const import DOMAIN

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from . import MistralConfigEntry

_LOGGER = logging.getLogger(__name__)

# How long a fetched model list stays fresh before the next call refetches it
MODEL_LIST_TTL = timedelta(hours=1)

# hass.data[DOMAIN]["model_cards"]: entry_id -> (fetched at, model cards)
MODEL_CARDS_STORE = "model_cards"


async def async_get_model_cards(
    hass: HomeAssistant, entry: MistralConfigEntry, *, force_refresh: bool = False
) -> list[Any]:
    """Return the account's model cards, cached per config entry.

    The list is refreshed when older than MODEL_LIST_TTL or when force_refresh
    is set. API and network errors are raised to the caller, which decides how
    to degrade.
    """
    store: dict[str, tuple[datetime, list[Any]]] = hass.data.setdefault(
        DOMAIN, {}
    ).setdefault(MODEL_CARDS_STORE, {})
    cached = store.get(entry.entry_id)
    now = dt_util.utcnow()
    if not force_refresh and cached is not None and now - cached[0] < MODEL_LIST_TTL:
        return cached[1]

    client = entry.runtime_data
    response = await client.models.list_async()
    cards = list(response.data or [])
    store[entry.entry_id] = (now, cards)
    return cards


@callback
def async_forget_model_cards(hass: HomeAssistant, entry: MistralConfigEntry) -> None:
    """Drop the cached model cards of a config entry."""
    hass.data.get(DOMAIN, {}).get(MODEL_CARDS_STORE, {}).pop(entry.entry_id, None)


async def async_model_supports_vision(
    hass: HomeAssistant, entry: MistralConfigEntry, model_id: str
) -> bool | None:
    """Return whether the model accepts image input, None when unknown."""
    try:
        cards = await async_get_model_cards(hass, entry)
    except Exception:
        _LOGGER.warning(
            "Could not fetch models from Mistral to check vision support for %s",
            model_id,
            exc_info=True,
        )
        return None
    for card in cards:
        if model_id in {card.id, *(getattr(card, "aliases", None) or [])}:
            return bool(getattr(getattr(card, "capabilities", None), "vision", False))
    return None

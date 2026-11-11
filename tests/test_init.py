"""Tests for setting up the Mistral AI integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from homeassistant.config_entries import SOURCE_REAUTH, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from mistralai.client import models
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_conversation.const import DOMAIN

from .conftest import api_error


async def test_setup_and_unload(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    """One entity is created per subentry and the entry unloads cleanly."""
    assert setup_integration.state is ConfigEntryState.LOADED
    for entity_id in (
        "conversation.mistral_ai",
        "ai_task.mistral_ai_task",
        "stt.mistral_stt",
        "tts.mistral_tts",
    ):
        assert hass.states.get(entity_id) is not None, entity_id

    assert await hass.config_entries.async_unload(setup_integration.entry_id)
    await hass.async_block_till_done()
    assert setup_integration.state is ConfigEntryState.NOT_LOADED


async def test_setup_starts_reauth_on_invalid_key(
    hass: HomeAssistant, mock_client: MagicMock, mock_config_entry: MockConfigEntry
) -> None:
    """A rejected API key asks the user to re-authenticate."""
    mock_client.models.list_async.side_effect = api_error(401, "Unauthorized")

    assert not await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR
    flows = hass.config_entries.flow.async_progress_by_handler(DOMAIN)
    assert [flow["context"]["source"] for flow in flows] == [SOURCE_REAUTH]


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("No route to host"), api_error(503, "Unavailable")],
    ids=["network down", "server error"],
)
async def test_setup_retries_when_mistral_is_unreachable(
    hass: HomeAssistant,
    mock_client: MagicMock,
    mock_config_entry: MockConfigEntry,
    error: Exception,
) -> None:
    """Setup is retried later instead of failing for good."""
    mock_client.models.list_async.side_effect = error

    assert not await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_generate_content(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """The generate_content action returns the model's answer."""
    mock_client.chat.complete_async = AsyncMock(
        return_value=models.ChatCompletionResponse.model_validate(
            {
                "id": "cmpl-1",
                "object": "chat.completion",
                "model": "mistral-large-4",
                "created": 0,
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Bonjour"},
                    }
                ],
            }
        )
    )

    response = await hass.services.async_call(
        DOMAIN,
        "generate_content",
        {"config_entry": setup_integration.entry_id, "prompt": "Say hello"},
        blocking=True,
        return_response=True,
    )

    assert response == {"text": "Bonjour"}
    assert mock_client.chat.complete_async.call_args.kwargs["model"] == (
        "mistral-large-4"
    )


async def test_generate_content_reports_network_errors(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """A network failure becomes a readable action error."""
    mock_client.chat.complete_async = AsyncMock(
        side_effect=httpx.ConnectError("No route to host")
    )

    with pytest.raises(HomeAssistantError, match="Could not reach Mistral"):
        await hass.services.async_call(
            DOMAIN,
            "generate_content",
            {"config_entry": setup_integration.entry_id, "prompt": "Say hello"},
            blocking=True,
            return_response=True,
        )


async def test_generate_content_needs_a_loaded_entry(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    """Calling the action for an unloaded entry is rejected cleanly."""
    assert await hass.config_entries.async_unload(setup_integration.entry_id)
    await hass.async_block_till_done()

    with pytest.raises(ServiceValidationError, match="not loaded"):
        await hass.services.async_call(
            DOMAIN,
            "generate_content",
            {"config_entry": setup_integration.entry_id, "prompt": "Say hello"},
            blocking=True,
            return_response=True,
        )

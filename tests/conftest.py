"""Fixtures for the Mistral AI integration tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component
from mistralai.client import models
from mistralai.client.errors import SDKError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_conversation.const import DOMAIN


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Let Home Assistant load the integration under test."""


def model_card(
    model_id: str,
    *,
    chat: bool = False,
    vision: bool = False,
    stt: bool = False,
    tts: bool = False,
    aliases: list[str] | None = None,
) -> models.BaseModelCard:
    """Build a model card as returned by the Mistral model list."""
    return models.BaseModelCard(
        id=model_id,
        name=model_id,
        aliases=aliases or [],
        capabilities=models.ModelCapabilities(
            completion_chat=chat,
            vision=vision,
            audio_transcription=stt,
            audio_speech=tts,
        ),
    )


def api_error(status_code: int, message: str = "API error") -> SDKError:
    """Build the error the SDK raises for an HTTP error response."""
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    return SDKError(message, httpx.Response(status_code, request=request))


def delta_event(
    content: Any = None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: tuple[int, int] | None = None,
) -> models.CompletionEvent:
    """Build one event of a streamed chat completion."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    chunk: dict[str, Any] = {
        "id": "cmpl-1",
        "model": "mistral-large-4",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": sum(usage),
        }
    return models.CompletionEvent.model_validate({"data": chunk})


async def event_stream(*events: Any) -> AsyncGenerator[Any]:
    """Mimic the event stream of a streamed answer."""
    for event in events:
        yield event


def text_stream(text: str) -> AsyncGenerator[models.CompletionEvent]:
    """Mimic a streamed plain text answer."""
    return event_stream(delta_event(text, finish_reason="stop", usage=(10, 5)))


def conversation_event(event_type: str, **data: Any) -> models.ConversationEvents:
    """Build one event of a streamed Conversations API response."""
    return models.ConversationEvents.model_validate(
        {"event": event_type, "data": {"type": event_type, **data}}
    )


def web_search_stream(*texts: str) -> AsyncGenerator[models.ConversationEvents]:
    """Mimic an answer for which Mistral searched the web."""
    return event_stream(
        conversation_event("conversation.response.started", conversation_id="conv-1"),
        conversation_event(
            "tool.execution.started", id="exec-1", name="web_search", arguments="{}"
        ),
        conversation_event("tool.execution.done", id="exec-1", name="web_search"),
        *(
            conversation_event("message.output.delta", id="msg-1", content=text)
            for text in texts
        ),
        conversation_event(
            "conversation.response.done",
            usage={"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
        ),
    )


@pytest.fixture
def mock_client() -> Generator[MagicMock]:
    """Stand in for the Mistral SDK client."""
    client = MagicMock()
    client.models.list_async = AsyncMock(
        return_value=models.ModelList(
            data=[
                model_card(
                    "mistral-large-4",
                    chat=True,
                    vision=True,
                    aliases=["mistral-large-latest"],
                ),
                model_card("zai-glm-5-3", chat=True),
                model_card("voxtral-mini-latest", stt=True),
                model_card("voxtral-mini-tts-2603", tts=True),
            ]
        )
    )
    client.audio.voices.list_async = AsyncMock(
        return_value=SimpleNamespace(
            items=[SimpleNamespace(id="voice-1", name="Jessica - Neutral")],
            total=1,
        )
    )
    with (
        patch("custom_components.mistral_conversation.Mistral", return_value=client),
        patch(
            "custom_components.mistral_conversation.config_flow.Mistral",
            return_value=client,
        ),
    ):
        yield client


@pytest.fixture
def conversation_options() -> dict[str, Any]:
    """Options of the conversation subentry; override to test other setups."""
    return {
        "recommended": True,
        "chat_model": "mistral-large-4",
        CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
        "prompt": "You are a helpful assistant.",
    }


@pytest.fixture
async def core_integration(hass: HomeAssistant) -> None:
    """Set up the core integration, which the conversation dependency relies on."""
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def mock_config_entry(
    hass: HomeAssistant, core_integration: None, conversation_options: dict[str, Any]
) -> MockConfigEntry:
    """Add a config entry with one subentry of each type."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Mistral AI",
        data={CONF_API_KEY: "test-key"},
        version=2,
        minor_version=6,
        subentries_data=[
            {
                "data": conversation_options,
                "subentry_type": "conversation",
                "title": "Mistral AI",
                "unique_id": None,
            },
            {
                "data": {"recommended": True, "chat_model": "mistral-large-4"},
                "subentry_type": "ai_task_data",
                "title": "Mistral AI Task",
                "unique_id": None,
            },
            {
                "data": {},
                "subentry_type": "stt",
                "title": "Mistral STT",
                "unique_id": None,
            },
            {
                "data": {"chat_model": "voxtral-mini-tts-2603"},
                "subentry_type": "tts",
                "title": "Mistral TTS",
                "unique_id": None,
            },
        ],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def setup_integration(
    hass: HomeAssistant, mock_client: MagicMock, mock_config_entry: MockConfigEntry
) -> MockConfigEntry:
    """Set up the integration with a mocked client."""
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry

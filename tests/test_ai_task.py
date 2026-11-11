"""Tests for the Mistral AI task entity."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import voluptuous as vol
from homeassistant.components import ai_task
from homeassistant.components.ai_task.const import DATA_COMPONENT
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import selector
from homeassistant.helpers.chat_session import async_get_chat_session
from mistralai.client import models
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .conftest import api_error, text_stream

ENTITY_ID = "ai_task.mistral_ai_task"

STRUCTURE = vol.Schema(
    {
        vol.Required("temperature", description="Outside temperature"): (
            selector.selector({"number": {"min": -40, "max": 60}})
        ),
        vol.Optional("summary"): selector.selector({"text": {}}),
    }
)


async def _generate_structured(hass: HomeAssistant) -> ai_task.GenDataTaskResult:
    return await ai_task.async_generate_data(
        hass,
        task_name="Weather report",
        entity_id=ENTITY_ID,
        instructions="Report the weather",
        structure=STRUCTURE,
    )


async def test_generate_text(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """Without a structure the answer is returned as text."""
    mock_client.chat.stream_async = AsyncMock(return_value=text_stream("Sunny."))

    result = await ai_task.async_generate_data(
        hass, task_name="Weather", entity_id=ENTITY_ID, instructions="Weather?"
    )

    assert result.data == "Sunny."
    assert "response_format" not in mock_client.chat.stream_async.call_args.kwargs


async def test_structure_is_sent_as_json_schema(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """The requested structure reaches the model as a strict JSON schema."""
    mock_client.chat.stream_async = AsyncMock(
        return_value=text_stream('{"temperature": 21.5, "summary": "Mild"}')
    )

    result = await _generate_structured(hass)

    assert result.data == {"temperature": 21.5, "summary": "Mild"}
    kwargs = mock_client.chat.stream_async.call_args.kwargs
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "weather_report",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "number",
                        "minimum": -40.0,
                        "maximum": 60.0,
                        "description": "Outside temperature",
                    },
                    "summary": {"type": "string"},
                },
                "required": ["temperature"],
                "additionalProperties": False,
            },
        },
    }
    # The SDK validates every request like this before sending it
    models.ChatCompletionStreamRequest(**kwargs)


async def test_structure_falls_back_to_json_mode(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """A model that refuses the schema is asked again in plain JSON mode."""
    requests: list[dict[str, Any]] = []

    async def stream_async(**kwargs: Any) -> Any:
        requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if len(requests) == 1:
            raise api_error(400, "json_schema is not supported by this model")
        return text_stream('{"temperature": 3}')

    mock_client.chat.stream_async = AsyncMock(side_effect=stream_async)

    result = await _generate_structured(hass)

    assert result.data == {"temperature": 3}
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[1]["response_format"] == {"type": "json_object"}
    system_message = requests[1]["messages"][0]
    assert system_message["role"] == "system"
    assert '"temperature"' in system_message["content"]
    assert [message["role"] for message in requests[1]["messages"]].count("system") == 1


async def test_invalid_json_is_an_error(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """An answer that is not JSON fails the task instead of returning garbage."""
    mock_client.chat.stream_async = AsyncMock(return_value=text_stream("not json"))

    with pytest.raises(HomeAssistantError, match="structured response"):
        await _generate_structured(hass)


async def test_generate_image(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """An image is generated with the image tool and downloaded."""
    mock_client.beta.conversations.start_async = AsyncMock(
        return_value=models.ConversationResponse.model_validate(
            {
                "conversation_id": "conv-1",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
                "outputs": [
                    {
                        "type": "message.output",
                        "content": [
                            {"type": "text", "text": "Here is your image."},
                            {
                                "type": "tool_file",
                                "tool": "image_generation",
                                "file_id": "file-1",
                                "file_type": "png",
                            },
                        ],
                    }
                ],
            }
        )
    )
    mock_client.files.download_async = AsyncMock(
        return_value=SimpleNamespace(aread=AsyncMock(return_value=b"PNG-DATA"))
    )

    entity = hass.data[DATA_COMPONENT].get_entity(ENTITY_ID)
    with async_get_chat_session(hass) as session:
        result = await entity.internal_async_generate_image(
            session,
            ai_task.GenImageTask(name="Lighthouse", instructions="A lighthouse"),
        )

    assert result.image_data == b"PNG-DATA"
    assert result.mime_type == "image/png"
    assert result.model == "mistral-large-4"
    kwargs = mock_client.beta.conversations.start_async.call_args.kwargs
    assert kwargs["tools"] == [{"type": "image_generation"}]
    assert kwargs["inputs"] == "A lighthouse"
    mock_client.files.download_async.assert_awaited_once_with(file_id="file-1")

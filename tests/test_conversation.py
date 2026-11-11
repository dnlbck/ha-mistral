"""Tests for the Mistral AI conversation agent."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from homeassistant.components import conversation
from homeassistant.config_entries import SOURCE_REAUTH
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent, llm
from mistralai.client import models
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_conversation.const import DOMAIN
from custom_components.mistral_conversation.entity import (
    _convert_content_to_entries,
    _convert_content_to_messages,
    _transform_conversation_stream,
    _transform_stream,
)

from .conftest import (
    api_error,
    conversation_event,
    delta_event,
    event_stream,
    text_stream,
    web_search_stream,
)

AGENT_ID = "conversation.mistral_ai"


async def _converse(hass: HomeAssistant, text: str = "hello") -> Any:
    return await conversation.async_converse(
        hass, text, None, Context(), agent_id=AGENT_ID
    )


async def _deltas(*events: models.CompletionEvent) -> tuple[list[Any], MagicMock]:
    chat_log = MagicMock()
    stream = _transform_stream(chat_log, event_stream(*events))
    return [delta async for delta in stream], chat_log


async def test_tools_are_sent_as_json_schema(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """Home Assistant's tools reach Mistral with a valid parameter schema.

    Regression test for tool parameters serialized as UNSUPPORTED, which made
    every request fail when "Control Home Assistant" was enabled.
    """
    mock_client.chat.stream_async = AsyncMock(return_value=text_stream("Hello!"))

    result = await _converse(hass)

    assert result.response.speech["plain"]["speech"] == "Hello!"
    kwargs = mock_client.chat.stream_async.call_args.kwargs
    assert kwargs["tools"]
    for tool in kwargs["tools"]:
        assert tool["function"]["parameters"]["type"] == "object", tool
    # The SDK validates every request like this before sending it
    models.ChatCompletionStreamRequest(**kwargs)
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1] == {"role": "user", "content": "hello"}


async def test_tool_call_round_trip(
    hass: HomeAssistant,
    mock_client: MagicMock,
    setup_integration: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool call is executed and its result is sent back to the model."""
    requests: list[list[dict[str, Any]]] = []
    tool_name = ""

    async def stream_async(**kwargs: Any) -> Any:
        nonlocal tool_name
        requests.append(list(kwargs["messages"]))
        if len(requests) > 1:
            return text_stream("Done.")
        tool_name = kwargs["tools"][0]["function"]["name"]
        return event_stream(
            delta_event(
                tool_calls=[
                    {
                        "id": "abc123XYZ",
                        "index": 0,
                        "function": {"name": tool_name, "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
            )
        )

    mock_client.chat.stream_async = AsyncMock(side_effect=stream_async)

    result = await _converse(hass)

    assert result.response.speech["plain"]["speech"] == "Done."
    assert len(requests) == 2
    assistant_message, tool_message = requests[1][-2:]
    assert assistant_message == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "abc123XYZ",
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }
        ],
    }
    assert tool_message["role"] == "tool"
    assert tool_message["name"] == tool_name
    assert tool_message["tool_call_id"] == "abc123XYZ"
    assert isinstance(json.loads(tool_message["content"]), dict)
    assert "ToolResultContent.tool_result" not in caplog.text


@pytest.mark.parametrize(
    "conversation_options",
    [{"recommended": True, "prompt": "Be brief.", CONF_LLM_HASS_API: ["assist"]}],
)
async def test_agents_without_a_stored_model_keep_their_old_default(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """An agent saved before 1.3.0 is not moved to the new recommended model."""
    mock_client.chat.stream_async = AsyncMock(return_value=text_stream("Hello!"))

    await _converse(hass)

    assert mock_client.chat.stream_async.call_args.kwargs["model"] == (
        "mistral-small-latest"
    )


WEB_SEARCH_OPTIONS = {
    "recommended": False,
    "chat_model": "mistral-large-4",
    "prompt": "Be brief.",
    "max_tokens": 400,
    "web_search": True,
    CONF_LLM_HASS_API: ["assist"],
}


@pytest.mark.parametrize("conversation_options", [WEB_SEARCH_OPTIONS])
async def test_web_search_uses_the_conversations_api(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """With web search on, the answer comes from the Conversations API."""
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        return_value=web_search_stream("It is ", "sunny.")
    )

    result = await _converse(hass, "How is the weather in Paris?")

    assert result.response.speech["plain"]["speech"] == "It is sunny."
    mock_client.chat.stream_async.assert_not_called()
    kwargs = mock_client.beta.conversations.start_stream_async.call_args.kwargs
    assert kwargs["model"] == "mistral-large-4"
    assert kwargs["store"] is False
    assert kwargs["instructions"].startswith("Be brief.")
    assert kwargs["inputs"][0] == {
        "type": "message.input",
        "role": "user",
        "content": "How is the weather in Paris?",
    }
    assert kwargs["tools"][-1] == {"type": "web_search"}
    assert all(tool["type"] == "function" for tool in kwargs["tools"][:-1])
    assert len(kwargs["tools"]) > 1
    assert kwargs["completion_args"]["max_tokens"] == 400
    # The SDK validates every request like this before sending it
    models.ConversationStreamRequest(**kwargs)


@pytest.mark.parametrize("conversation_options", [WEB_SEARCH_OPTIONS])
async def test_web_search_tool_call_round_trip(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """Home Assistant tools keep working next to web search."""
    requests: list[list[dict[str, Any]]] = []
    tool_name = ""

    async def start_stream_async(**kwargs: Any) -> Any:
        nonlocal tool_name
        requests.append(list(kwargs["inputs"]))
        models.ConversationStreamRequest(**kwargs)
        if len(requests) > 1:
            return web_search_stream("Done.")
        tool_name = kwargs["tools"][0]["function"]["name"]
        call = {"id": "fc-1", "name": tool_name, "tool_call_id": "abc123XYZ"}
        return event_stream(
            conversation_event("function.call.delta", **call, arguments="{"),
            conversation_event("function.call.delta", **call, arguments="}"),
            conversation_event(
                "conversation.response.done",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            ),
        )

    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=start_stream_async
    )

    result = await _converse(hass)

    assert result.response.speech["plain"]["speech"] == "Done."
    assert len(requests) == 2
    call_entry, result_entry = requests[1][-2:]
    assert call_entry == {
        "type": "function.call",
        "tool_call_id": "abc123XYZ",
        "name": tool_name,
        "arguments": "{}",
    }
    assert result_entry["type"] == "function.result"
    assert result_entry["tool_call_id"] == "abc123XYZ"
    assert isinstance(json.loads(result_entry["result"]), dict)


@pytest.mark.parametrize("conversation_options", [WEB_SEARCH_OPTIONS])
async def test_web_search_falls_back_to_plain_chat(
    hass: HomeAssistant,
    mock_client: MagicMock,
    setup_integration: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model the Conversations API refuses still answers, without web search."""
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=api_error(400, "Model does not support connectors")
    )
    mock_client.chat.stream_async = AsyncMock(return_value=text_stream("Hello!"))

    result = await _converse(hass)

    assert result.response.speech["plain"]["speech"] == "Hello!"
    kwargs = mock_client.chat.stream_async.call_args.kwargs
    assert {"type": "web_search"} not in kwargs["tools"]
    models.ChatCompletionStreamRequest(**kwargs)
    assert "answering without web search" in caplog.text


@pytest.mark.parametrize("conversation_options", [WEB_SEARCH_OPTIONS])
async def test_web_search_reports_stream_errors(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """An error event in the stream becomes an error response."""
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        return_value=event_stream(
            conversation_event(
                "conversation.response.error", message="Search failed", code=500
            )
        )
    )

    result = await _converse(hass)

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert "Search failed" in result.response.speech["plain"]["speech"]


async def test_web_search_stream_skips_references() -> None:
    """Source references between text chunks do not end up in the answer."""
    chat_log = MagicMock()
    stream = _transform_conversation_stream(
        chat_log,
        event_stream(
            conversation_event(
                "message.output.delta",
                id="msg-1",
                content={"type": "text", "text": "Sunny"},
            ),
            conversation_event(
                "message.output.delta",
                id="msg-1",
                content={
                    "type": "tool_reference",
                    "tool": "web_search",
                    "title": "Weather",
                    "url": "https://example.com",
                },
            ),
            conversation_event("message.output.delta", id="msg-1", content="."),
            conversation_event(
                "conversation.response.done",
                usage={"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
            ),
        ),
    )

    assert [delta async for delta in stream] == [
        {"role": "assistant"},
        {"content": "Sunny"},
        {"content": "."},
    ]
    chat_log.async_trace.assert_called_once_with(
        {"stats": {"input_tokens": 30, "output_tokens": 8}}
    )


async def test_rate_limit_is_retried(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """A 429 answer is retried after a pause."""
    mock_client.chat.stream_async = AsyncMock(
        side_effect=[api_error(429, "Too many requests"), text_stream("Hello!")]
    )

    with patch(
        "custom_components.mistral_conversation.entity.asyncio.sleep"
    ) as mock_sleep:
        result = await _converse(hass)

    assert result.response.speech["plain"]["speech"] == "Hello!"
    assert mock_sleep.await_count == 1
    assert mock_client.chat.stream_async.await_count == 2


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (api_error(500, "Internal error"), "Error talking to Mistral"),
        (httpx.ConnectError("No route to host"), "Could not reach Mistral"),
        (api_error(401, "Unauthorized"), "Authentication error with Mistral"),
    ],
    ids=["server error", "network down", "invalid key"],
)
async def test_errors_become_error_responses(
    hass: HomeAssistant,
    mock_client: MagicMock,
    setup_integration: MockConfigEntry,
    error: Exception,
    message: str,
) -> None:
    """API and network failures are reported to the user, not raised."""
    mock_client.chat.stream_async = AsyncMock(side_effect=error)

    result = await _converse(hass)

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert message in result.response.speech["plain"]["speech"]
    reauth_flows = [
        flow
        for flow in hass.config_entries.flow.async_progress_by_handler(DOMAIN)
        if flow["context"]["source"] == SOURCE_REAUTH
    ]
    assert bool(reauth_flows) is ("Authentication" in message)


async def test_stream_starts_with_the_assistant_role() -> None:
    """Assist only streams to text-to-speech once it has seen the role."""
    deltas, chat_log = await _deltas(
        delta_event("Hel"),
        delta_event("lo", finish_reason="stop", usage=(12, 3)),
    )

    assert deltas == [{"role": "assistant"}, {"content": "Hel"}, {"content": "lo"}]
    chat_log.async_trace.assert_called_once_with(
        {"stats": {"input_tokens": 12, "output_tokens": 3}}
    )


async def test_stream_separates_reasoning_from_the_answer() -> None:
    """Reasoning models send chunk lists instead of plain strings."""
    deltas, _ = await _deltas(
        delta_event(
            [{"type": "thinking", "thinking": [{"type": "text", "text": "Hmm. "}]}]
        ),
        delta_event([{"type": "text", "text": "42"}], finish_reason="stop"),
    )

    assert deltas == [
        {"role": "assistant"},
        {"thinking_content": "Hmm. "},
        {"content": "42"},
    ]


@pytest.mark.parametrize("finish_reason", ["tool_calls", "length"])
async def test_stream_assembles_tool_calls(finish_reason: str) -> None:
    """Tool call fragments are joined; unfinished tool calls are not run."""
    deltas, _ = await _deltas(
        delta_event(
            tool_calls=[
                {
                    "id": "call00001",
                    "index": 0,
                    "function": {"name": "HassTurnOn", "arguments": '{"name": '},
                }
            ]
        ),
        delta_event(
            tool_calls=[
                {"index": 0, "function": {"name": "", "arguments": '"lamp"}'}},
                {
                    "id": "call00002",
                    "index": 1,
                    "function": {"name": "GetLiveContext", "arguments": {}},
                },
            ],
            finish_reason=finish_reason,
        ),
    )

    assert deltas[0] == {"role": "assistant"}
    if finish_reason == "length":
        assert len(deltas) == 1
        return
    tool_calls = deltas[1]["tool_calls"]
    assert [(call.id, call.tool_name, call.tool_args) for call in tool_calls] == [
        ("call00001", "HassTurnOn", {"name": "lamp"}),
        ("call00002", "GetLiveContext", {}),
    ]
    assert len(deltas) == 2


def test_history_is_converted_to_mistral_messages() -> None:
    """Chat log content maps to Mistral messages with paired tool call IDs."""
    tool_call = llm.ToolInput(
        id="01JABCDEFGHJKMNPQRSTVWXYZ0", tool_name="HassTurnOn", tool_args={"a": 1}
    )
    messages = _convert_content_to_messages(
        [
            conversation.SystemContent(content="Be brief."),
            conversation.UserContent(content="Turn on the lamp"),
            conversation.AssistantContent(agent_id=AGENT_ID, content="On it."),
            conversation.AssistantContent(agent_id=AGENT_ID, tool_calls=[tool_call]),
            conversation.ToolResultContent(
                agent_id=AGENT_ID,
                tool_call_id=tool_call.id,
                tool_name="HassTurnOn",
                result=llm.ToolResult(data={"success": True}),
            ),
            conversation.AssistantContent(agent_id=AGENT_ID, content="Done."),
        ]
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "tool",
        "assistant",
    ]
    # The first answer keeps its text and does not inherit the later tool call
    assert messages[2] == {"role": "assistant", "content": "On it."}
    call_id = messages[3]["tool_calls"][0]["id"]
    assert len(call_id) == 9
    assert call_id.isalnum()
    assert messages[4] == {
        "role": "tool",
        "name": "HassTurnOn",
        "content": '{"success":true}',
        "tool_call_id": call_id,
    }


def test_history_is_converted_to_conversation_entries() -> None:
    """The Conversations API gets the same history as entries."""
    tool_call = llm.ToolInput(
        id="01JABCDEFGHJKMNPQRSTVWXYZ0", tool_name="HassTurnOn", tool_args={"a": 1}
    )
    entries = _convert_content_to_entries(
        [
            conversation.SystemContent(content="Be brief."),
            conversation.UserContent(content="Turn on the lamp"),
            conversation.AssistantContent(
                agent_id=AGENT_ID, content="On it.", tool_calls=[tool_call]
            ),
            conversation.ToolResultContent(
                agent_id=AGENT_ID,
                tool_call_id=tool_call.id,
                tool_name="HassTurnOn",
                result=llm.ToolResult(data={"success": True}),
            ),
        ]
    )

    call_id = entries[2]["tool_call_id"]
    assert entries == [
        {"type": "message.input", "role": "user", "content": "Turn on the lamp"},
        {"type": "message.input", "role": "assistant", "content": "On it."},
        {
            "type": "function.call",
            "tool_call_id": call_id,
            "name": "HassTurnOn",
            "arguments": '{"a": 1}',
        },
        {
            "type": "function.result",
            "tool_call_id": call_id,
            "result": '{"success":true}',
        },
    ]
    # The SDK accepts the history as the input of a new conversation
    models.ConversationStreamRequest(model="mistral-large-4", inputs=entries)

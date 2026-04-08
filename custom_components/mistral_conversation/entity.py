"""Base entity for Mistral."""

from __future__ import annotations

import asyncio
import json
import random
import string
from collections.abc import AsyncGenerator, Callable, Iterable
from typing import TYPE_CHECKING, Any

from mistralai.client.errors import SDKError
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.json import json_dumps

from .const import (
    BACKOFF_INITIAL_DELAY,
    BACKOFF_MAX_DELAY,
    BACKOFF_MAX_RETRIES,
    BACKOFF_MULTIPLIER,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_SAFE_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_SAFE_PROMPT,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_TTS_MODEL,
)

if TYPE_CHECKING:
    from . import MistralConfigEntry


# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_backoff(attempt: int, context: str = "API") -> None:
    """Shared exponential backoff for 429 rate limits."""
    delay = min(
        BACKOFF_INITIAL_DELAY * (BACKOFF_MULTIPLIER**attempt),
        BACKOFF_MAX_DELAY,
    )
    LOGGER.warning(
        "Rate limited by Mistral %s (429), retrying in %.1f seconds (attempt %d/%d)",
        context,
        delay,
        attempt + 1,
        BACKOFF_MAX_RETRIES,
    )
    await asyncio.sleep(delay)


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format tool specification for Mistral function calling."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
        },
    }


def _mistral_tool_call_id(ha_id: str, id_map: dict[str, str]) -> str:
    """Convert an HA tool call ID to a Mistral-compatible 9-char alphanumeric ID."""
    if ha_id in id_map:
        return id_map[ha_id]
    # Filter to alphanumeric only
    alnum = "".join(c for c in ha_id if c.isalnum())
    if len(alnum) == 9:
        short = alnum
    elif len(alnum) > 9:
        short = alnum[:9]
    else:
        # Pad if too short
        chars = string.ascii_letters + string.digits
        short = alnum + "".join(random.choices(chars, k=9 - len(alnum)))
    # Deduplicate: if collision, append counter chars
    while short in id_map.values():
        short = "".join(random.choices(string.ascii_letters + string.digits, k=9))
    id_map[ha_id] = short
    return short


def _convert_content_to_messages(
    chat_content: Iterable[conversation.Content],
) -> list[dict[str, Any]]:
    """Convert HA chat content to Mistral message format."""
    messages: list[dict[str, Any]] = []
    # id_map is local per call — safe because each call converts a complete
    # message history, so tool_call/tool_result pairs always appear together.
    id_map: dict[str, str] = {}

    for content in chat_content:
        if isinstance(content, conversation.ToolResultContent):
            messages.append(
                {
                    "role": "tool",
                    "name": content.tool_name,
                    "content": json_dumps(content.tool_result),
                    "tool_call_id": _mistral_tool_call_id(content.tool_call_id, id_map),
                }
            )
            continue

        if content.content:
            role: str = content.role
            if role == "system":
                role = "system"
            messages.append({"role": role, "content": content.content})

        if isinstance(content, conversation.AssistantContent):
            if content.tool_calls:
                tool_calls = []
                for tool_call in content.tool_calls:
                    tool_calls.append(
                        {
                            "id": _mistral_tool_call_id(tool_call.id, id_map),
                            "type": "function",
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": json.dumps(tool_call.tool_args),
                            },
                        }
                    )
                # If we already appended a message for this content, add
                # tool_calls to it; otherwise create a new assistant message
                if messages and messages[-1].get("role") == "assistant":
                    messages[-1]["tool_calls"] = tool_calls
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.content or "",
                            "tool_calls": tool_calls,
                        }
                    )

    return messages


async def _transform_stream(
    chat_log: conversation.ChatLog,
    result: Any,
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
]:
    """Transform a Mistral delta stream into HA format."""
    current_tool_calls: dict[int, dict[str, Any]] = {}

    async for event in result:
        chunk = event.data
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if delta.content:
            yield {"content": delta.content}

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index if hasattr(tc, "index") else 0
                if idx not in current_tool_calls:
                    current_tool_calls[idx] = {
                        "id": tc.id or "",
                        "name": (
                            tc.function.name if tc.function and tc.function.name else ""
                        ),
                        "arguments": "",
                    }
                if tc.id:
                    current_tool_calls[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        current_tool_calls[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        current_tool_calls[idx]["arguments"] += tc.function.arguments

        finish_reason = chunk.choices[0].finish_reason
        if finish_reason == "tool_calls" and current_tool_calls:
            tool_inputs = []
            for tc_data in current_tool_calls.values():
                try:
                    args = json.loads(tc_data["arguments"])
                except json.JSONDecodeError:
                    args = {}
                tool_inputs.append(
                    llm.ToolInput(
                        id=tc_data["id"],
                        tool_name=tc_data["name"],
                        tool_args=args,
                    )
                )
            yield {"tool_calls": tool_inputs}
            current_tool_calls.clear()

        if finish_reason == "stop" and chunk.usage is not None:
            chat_log.async_trace(
                {
                    "stats": {
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens,
                    }
                }
            )


class MistralBaseLLMEntity(Entity):
    """Mistral base LLM entity."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(self, entry: MistralConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        default_model = RECOMMENDED_CHAT_MODEL
        if subentry.subentry_type == "stt":
            default_model = RECOMMENDED_STT_MODEL
        elif subentry.subentry_type == "tts":
            default_model = RECOMMENDED_TTS_MODEL
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Mistral AI",
            model=subentry.data.get(CONF_CHAT_MODEL, default_model),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        """Generate an answer for the chat log."""
        options = self.subentry.data

        messages = _convert_content_to_messages(chat_log.content)

        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        model_args: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            "safe_prompt": options.get(CONF_SAFE_PROMPT, RECOMMENDED_SAFE_PROMPT),
        }

        tools: list[dict[str, Any]] = []
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        if structure and structure_name:
            model_args["response_format"] = {"type": "json_object"}

        if tools:
            model_args["tools"] = tools
            model_args["tool_choice"] = "auto"

        client = self.entry.runtime_data

        retries = 0
        for _iteration in range(max_iterations):
            try:
                result = await client.chat.stream_async(**model_args)

                new_messages = _convert_content_to_messages(
                    [
                        content
                        async for content in chat_log.async_add_delta_content_stream(
                            self.entity_id,
                            _transform_stream(chat_log, result),
                        )
                    ]
                )
                messages.extend(new_messages)
                model_args["messages"] = messages
                retries = 0

            except SDKError as err:
                if err.status_code == 429:
                    if retries >= BACKOFF_MAX_RETRIES:
                        raise HomeAssistantError(
                            "Rate limited by Mistral after maximum retries"
                        ) from err
                    await async_backoff(retries, "chat")
                    retries += 1
                    continue
                if err.status_code == 401:
                    LOGGER.error("Authentication error with Mistral: %s", err)
                    raise HomeAssistantError(
                        "Authentication error with Mistral"
                    ) from err
                LOGGER.error("Error talking to Mistral: %s", err)
                raise HomeAssistantError(
                    f"Error talking to Mistral: {err.message}"
                ) from err

            if not chat_log.unresponded_tool_results:
                break

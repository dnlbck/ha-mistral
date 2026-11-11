"""Base entity for Mistral."""

from __future__ import annotations

import asyncio
import base64
import json
import random
import string
from collections.abc import AsyncGenerator, Callable, Iterable
from json import JSONDecodeError
from mimetypes import guess_file_type
from pathlib import Path
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.json import json_dumps
from homeassistant.util import slugify
from homeassistant.util.json import json_loads
from mistralai.client.models import (
    FunctionCallEvent,
    MessageOutputEvent,
    ResponseDoneEvent,
    ResponseErrorEvent,
    TextChunk,
    ThinkChunk,
)

from .const import (
    BACKOFF_INITIAL_DELAY,
    BACKOFF_MAX_DELAY,
    BACKOFF_MAX_RETRIES,
    BACKOFF_MULTIPLIER,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_REASONING_EFFORT,
    CONF_SAFE_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_WEB_SEARCH,
    DOMAIN,
    LEGACY_CHAT_MODEL,
    LOGGER,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_SAFE_PROMPT,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_WEB_SEARCH,
)
from .errors import MISTRAL_CONNECTION_ERRORS, MistralAPIError
from .models import async_model_supports_vision

try:
    # Home Assistant 2026.9+ builds its schemas with probatio. Its serializers
    # hand back probatio's UNSUPPORTED sentinel, which voluptuous-openapi does
    # not recognize, so the schema has to be rendered by probatio as well.
    from probatio import to_openapi as _schema_to_openapi
except ImportError:
    # Older Home Assistant releases ship voluptuous-openapi instead
    from voluptuous_openapi import convert as _schema_to_openapi

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
            "parameters": _schema_to_openapi(
                tool.parameters, custom_serializer=custom_serializer
            ),
        },
    }


def _close_objects(node: Any) -> Any:
    """Forbid undeclared keys on every object, as strict JSON schemas require."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
        for value in node.values():
            _close_objects(value)
    elif isinstance(node, list):
        for value in node:
            _close_objects(value)
    return node


def _format_structured_output(
    name: str, structure: vol.Schema, llm_api: llm.APIInstance | None
) -> dict[str, Any]:
    """Describe the expected answer as a strict JSON schema response format."""
    schema = _schema_to_openapi(
        structure,
        custom_serializer=(
            llm_api.custom_serializer if llm_api else llm.selector_serializer
        ),
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": slugify(name)[:64] or "response",
            "schema": _close_objects(schema),
            "strict": True,
        },
    }


def _use_json_mode(model_args: dict[str, Any]) -> None:
    """Swap a JSON schema response format for JSON mode plus written instructions."""
    schema = model_args["response_format"]["json_schema"]["schema"]
    instructions = (
        "Respond with a single JSON object that follows this JSON schema:\n"
        f"{json.dumps(schema)}"
    )
    messages: list[dict[str, Any]] = model_args["messages"]
    if messages and messages[0]["role"] == "system":
        messages[0] = {
            **messages[0],
            "content": f"{messages[0]['content']}\n\n{instructions}",
        }
    else:
        messages.insert(0, {"role": "system", "content": instructions})
    model_args["response_format"] = {"type": "json_object"}


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


async def async_prepare_files_for_prompt(
    hass: HomeAssistant, files: list[tuple[Path, str | None]]
) -> list[dict[str, Any]]:
    """Convert file attachments to Mistral vision content items."""

    def _prepare() -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for file_path, mime_type in files:
            if not file_path.exists():
                raise HomeAssistantError(f"`{file_path}` does not exist")
            if mime_type is None:
                mime_type = guess_file_type(file_path)[0]
            if not mime_type or not mime_type.startswith("image/"):
                raise HomeAssistantError(
                    "Only images are supported by the Mistral Vision API, "
                    f"`{file_path}` is not an image file"
                )
            base64_data = base64.b64encode(file_path.read_bytes()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{base64_data}",
                }
            )
        return content

    return await hass.async_add_executor_job(_prepare)


def _tool_result_data(content: conversation.ToolResultContent) -> Any:
    """Return what a tool call produced."""
    # Home Assistant 2026.10 moved the payload to `result` and deprecated
    # the old `tool_result` attribute.
    if (result := getattr(content, "result", None)) is not None:
        return result.data
    return content.tool_result


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
                    "content": json_dumps(_tool_result_data(content)),
                    "tool_call_id": _mistral_tool_call_id(content.tool_call_id, id_map),
                }
            )
            continue

        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": content.content or "",
                    "tool_calls": [
                        {
                            "id": _mistral_tool_call_id(tool_call.id, id_map),
                            "type": "function",
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": json.dumps(tool_call.tool_args),
                            },
                        }
                        for tool_call in content.tool_calls
                    ],
                }
            )
            continue

        if content.content:
            messages.append({"role": content.role, "content": content.content})

    return messages


def _convert_content_to_entries(
    chat_content: Iterable[conversation.Content],
) -> list[dict[str, Any]]:
    """Convert HA chat content to Mistral Conversations API entries.

    System prompts are left out, that API takes them as instructions.
    """
    entries: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}

    for content in chat_content:
        if isinstance(content, conversation.ToolResultContent):
            entries.append(
                {
                    "type": "function.result",
                    "tool_call_id": _mistral_tool_call_id(content.tool_call_id, id_map),
                    "result": json_dumps(_tool_result_data(content)),
                }
            )
            continue

        if isinstance(content, conversation.SystemContent):
            continue

        if content.content:
            entries.append(
                {
                    "type": "message.input",
                    "role": content.role,
                    "content": content.content,
                }
            )

        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            entries.extend(
                {
                    "type": "function.call",
                    "tool_call_id": _mistral_tool_call_id(tool_call.id, id_map),
                    "name": tool_call.tool_name,
                    "arguments": json.dumps(tool_call.tool_args),
                }
                for tool_call in content.tool_calls
            )

    return entries


def _split_delta_content(content: Any) -> tuple[str, str]:
    """Split streamed content into (answer text, thinking text).

    Plain answers arrive as a string. Reasoning models send a list of chunks
    in which thinking chunks carry the reasoning and text chunks the answer.
    """
    if isinstance(content, str):
        return content, ""
    if not isinstance(content, list):
        return "", ""

    text: list[str] = []
    thinking: list[str] = []
    for chunk in content:
        if isinstance(chunk, TextChunk):
            text.append(chunk.text)
        elif isinstance(chunk, ThinkChunk):
            thinking.extend(
                part.text for part in chunk.thinking if isinstance(part, TextChunk)
            )
    return "".join(text), "".join(thinking)


def _tool_inputs(pending: dict[Any, dict[str, Any]]) -> list[llm.ToolInput]:
    """Turn the accumulated tool call fragments into tool inputs."""
    tool_inputs: list[llm.ToolInput] = []
    for tool_call in pending.values():
        arguments = tool_call["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json_loads(arguments) if arguments else {}
            except JSONDecodeError:
                LOGGER.warning(
                    "Mistral sent arguments for %s that are not valid JSON: %s",
                    tool_call["name"],
                    arguments,
                )
                arguments = {}
        tool_inputs.append(
            llm.ToolInput(
                id=tool_call["id"],
                tool_name=tool_call["name"],
                tool_args=arguments,
            )
        )
    return tool_inputs


async def _transform_stream(
    chat_log: conversation.ChatLog,
    result: Any,
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
]:
    """Transform a Mistral delta stream into HA format."""
    pending_tool_calls: dict[int, dict[str, Any]] = {}
    started = False

    async for event in result:
        chunk = event.data

        if chunk.usage is not None:
            chat_log.async_trace(
                {
                    "stats": {
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens,
                    }
                }
            )

        if not chunk.choices:
            continue

        if not started:
            # Marks the start of a new assistant message. Assist needs it to
            # stream the answer to text-to-speech and to keep the answer after
            # a tool call apart from the text before it.
            started = True
            yield {"role": "assistant"}

        choice = chunk.choices[0]
        delta = choice.delta

        text, thinking = _split_delta_content(delta.content)
        if reasoning := getattr(delta, "reasoning_content", None):
            thinking += reasoning
        if thinking:
            yield {"thinking_content": thinking}
        if text:
            yield {"content": text}

        for tool_call in delta.tool_calls or []:
            pending = pending_tool_calls.setdefault(
                tool_call.index or 0, {"id": "", "name": "", "arguments": ""}
            )
            # The SDK fills a missing ID with the string "null"
            if tool_call.id and tool_call.id != "null":
                pending["id"] = tool_call.id
            if function := tool_call.function:
                if function.name:
                    pending["name"] = function.name
                if isinstance(function.arguments, dict):
                    pending["arguments"] = function.arguments
                elif function.arguments:
                    pending["arguments"] += function.arguments

        # Only run tool calls the model finished. A reply cut off by the
        # token limit can leave one behind with half its arguments.
        if choice.finish_reason == "tool_calls" and pending_tool_calls:
            yield {"tool_calls": _tool_inputs(pending_tool_calls)}
            pending_tool_calls.clear()


async def _transform_conversation_stream(
    chat_log: conversation.ChatLog,
    result: Any,
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
]:
    """Transform a Mistral Conversations API event stream into HA format.

    Web searches run on Mistral's side and only show up as answer text here.
    """
    pending_tool_calls: dict[str, dict[str, Any]] = {}
    started = False

    async for event in result:
        data = event.data

        if isinstance(data, ResponseErrorEvent):
            raise HomeAssistantError(f"Error talking to Mistral: {data.message}")

        if isinstance(data, ResponseDoneEvent):
            chat_log.async_trace(
                {
                    "stats": {
                        "input_tokens": data.usage.prompt_tokens,
                        "output_tokens": data.usage.completion_tokens,
                    }
                }
            )
            # Only run tool calls of a response Mistral completed
            if pending_tool_calls:
                yield {"tool_calls": _tool_inputs(pending_tool_calls)}
                pending_tool_calls.clear()
            continue

        if not isinstance(data, (MessageOutputEvent, FunctionCallEvent)):
            continue

        if not started:
            started = True
            yield {"role": "assistant"}

        if isinstance(data, FunctionCallEvent):
            pending = pending_tool_calls.setdefault(
                data.tool_call_id,
                {"id": data.tool_call_id, "name": data.name, "arguments": ""},
            )
            pending["arguments"] += data.arguments
            continue

        # A message delta is plain text or a single chunk. Chunks that only
        # reference a search result carry no text of their own.
        content = data.content
        text, thinking = _split_delta_content(
            content if isinstance(content, str) else [content]
        )
        if thinking:
            yield {"thinking_content": thinking}
        if text:
            yield {"content": text}


class MistralBaseLLMEntity(Entity):
    """Mistral base LLM entity."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(self, entry: MistralConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._web_search_fallback_logged = False
        default_model = LEGACY_CHAT_MODEL
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
        client = self.entry.runtime_data

        # Agents saved before 1.3.0 stored no model and keep the one they used
        model = options.get(CONF_CHAT_MODEL, LEGACY_CHAT_MODEL)

        messages = _convert_content_to_messages(chat_log.content)

        # Mistral only offers web search through its Conversations API, which
        # takes the history as entries instead of chat messages
        web_search = bool(
            options.get(CONF_WEB_SEARCH, RECOMMENDED_WEB_SEARCH) and not structure
        )
        entries = _convert_content_to_entries(chat_log.content) if web_search else []

        # Handle image/file attachments for vision models
        last_content = chat_log.content[-1]
        if (
            last_content.role == "user"
            and hasattr(last_content, "attachments")
            and last_content.attachments
        ):
            if await async_model_supports_vision(self.hass, self.entry, model) is False:
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="model_does_not_support_vision",
                    translation_placeholders={"model": model},
                )
            files = await async_prepare_files_for_prompt(
                self.hass,
                [(a.path, a.mime_type) for a in last_content.attachments],
            )
            for last_message in (messages[-1], *entries[-1:]):
                if last_message.get("role") != "user" or not isinstance(
                    last_message["content"], str
                ):
                    raise HomeAssistantError(
                        "Expected a user message with string content when attachments are present"
                    )
                last_message["content"] = [
                    {"type": "text", "text": last_message["content"]},
                    *files,
                ]

        sampling: dict[str, Any] = {
            "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
        }
        reasoning_effort = options.get(
            CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
        )
        if reasoning_effort != "none":
            sampling["reasoning_effort"] = reasoning_effort

        tools: list[dict[str, Any]] = []
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        model_args: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "safe_prompt": options.get(CONF_SAFE_PROMPT, RECOMMENDED_SAFE_PROMPT),
            **sampling,
        }
        if structure:
            model_args["response_format"] = _format_structured_output(
                structure_name or "response", structure, chat_log.llm_api
            )
        if tools:
            model_args["tools"] = tools
            model_args["tool_choice"] = "auto"

        conversation_args: dict[str, Any] = {
            "model": model,
            "inputs": entries,
            "tools": [*tools, {"type": "web_search"}],
            "completion_args": {**sampling, "tool_choice": "auto"},
            # Nothing is kept on Mistral's servers, the history is sent each time
            "store": False,
        }
        if instructions := "\n".join(
            content.content
            for content in chat_log.content
            if isinstance(content, conversation.SystemContent) and content.content
        ):
            conversation_args["instructions"] = instructions

        retries = 0
        for _iteration in range(max_iterations):
            try:
                stream = None
                if web_search:
                    stream = await self._async_web_search_stream(
                        chat_log, conversation_args
                    )
                    web_search = stream is not None
                if stream is None:
                    stream = _transform_stream(
                        chat_log, await client.chat.stream_async(**model_args)
                    )

                new_content = [
                    content
                    async for content in chat_log.async_add_delta_content_stream(
                        self.entity_id, stream
                    )
                ]
                messages.extend(_convert_content_to_messages(new_content))
                if web_search:
                    entries.extend(_convert_content_to_entries(new_content))
                retries = 0

            except MistralAPIError as err:
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
                    self.entry.async_start_reauth(self.hass)
                    raise HomeAssistantError(
                        "Authentication error with Mistral"
                    ) from err
                if (
                    err.status_code in (400, 422)
                    and model_args.get("response_format", {}).get("type")
                    == "json_schema"
                ):
                    # Not every model or schema works with strict output
                    LOGGER.warning(
                        "Mistral rejected the JSON schema for %s, retrying in "
                        "JSON mode: %s",
                        model,
                        err,
                    )
                    _use_json_mode(model_args)
                    continue
                LOGGER.error("Error talking to Mistral: %s", err)
                raise HomeAssistantError(
                    f"Error talking to Mistral: {err.message}"
                ) from err
            except MISTRAL_CONNECTION_ERRORS as err:
                LOGGER.error("Could not reach Mistral: %r", err)
                raise HomeAssistantError(
                    f"Could not reach Mistral: {str(err) or type(err).__name__}"
                ) from err

            if not chat_log.unresponded_tool_results:
                break

    async def _async_web_search_stream(
        self, chat_log: conversation.ChatLog, conversation_args: dict[str, Any]
    ) -> (
        AsyncGenerator[
            conversation.AssistantContentDeltaDict
            | conversation.ToolResultContentDeltaDict
        ]
        | None
    ):
        """Start an answer that may search the web.

        Returns None when Mistral does not take the request, so the caller can
        answer through the regular chat API without web search.
        """
        client = self.entry.runtime_data
        try:
            result = await client.beta.conversations.start_stream_async(
                **conversation_args
            )
        except (MistralAPIError, ValueError) as err:
            # Rate limits, a rejected API key and outages are for the caller.
            # Anything else means this model or request cannot use the
            # Conversations API, or the SDK refused to build the request.
            if isinstance(err, MistralAPIError) and err.status_code not in (
                400,
                404,
                422,
            ):
                raise
            log = LOGGER.debug if self._web_search_fallback_logged else LOGGER.warning
            self._web_search_fallback_logged = True
            log(
                "Mistral did not accept a web search request for %s, answering "
                "without web search: %s",
                self.entity_id,
                err,
            )
            return None
        return _transform_conversation_stream(chat_log, result)

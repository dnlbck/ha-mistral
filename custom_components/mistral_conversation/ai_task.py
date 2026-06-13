"""AI Task support for Mistral."""

from json import JSONDecodeError
import logging
from mimetypes import guess_file_type
from typing import TYPE_CHECKING, Any

from mistralai.client.errors import SDKError
from mistralai.client.models import MessageOutputEntry, TextChunk, ToolFileChunk

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
)
from homeassistant.util.json import json_loads

from . import MistralConfigEntry
from .const import (
    BACKOFF_MAX_RETRIES,
    CONF_CHAT_MODEL,
    RECOMMENDED_IMAGE_MODEL,
)
from .entity import (
    MistralBaseLLMEntity,
    async_backoff,
    async_prepare_files_for_prompt,
)

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry

_LOGGER = logging.getLogger(__name__)

IMAGE_GENERATION_INSTRUCTIONS = (
    "Use the image generation tool to generate the image requested by the user."
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: MistralConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [MistralTaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class MistralTaskEntity(ai_task.AITaskEntity, MistralBaseLLMEntity):
    """Mistral AI task entity."""

    _attr_supported_features = (
        ai_task.AITaskEntityFeature.GENERATE_DATA
        | ai_task.AITaskEntityFeature.GENERATE_IMAGE
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )

    def __init__(self, entry: MistralConfigEntry, subentry: "ConfigSubentry") -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        await self._async_handle_chat_log(
            chat_log, task.name, task.structure, max_iterations=1000
        )

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

        text = chat_log.content[-1].content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            _LOGGER.error(
                "Failed to parse JSON response: %s. Response: %s",
                err,
                text,
            )
            raise HomeAssistantError("Error with Mistral structured response") from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task."""
        user_message = chat_log.content[-1]
        if not isinstance(user_message, conversation.UserContent):
            raise HomeAssistantError("Last content in chat log is not a UserContent")

        options = self.subentry.data
        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_IMAGE_MODEL)
        client = self.entry.runtime_data

        inputs: str | list[dict[str, Any]] = user_message.content
        if user_message.attachments:
            files = await async_prepare_files_for_prompt(
                self.hass,
                [(a.path, a.mime_type) for a in user_message.attachments],
            )
            inputs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message.content},
                        *files,
                    ],
                }
            ]

        retries = 0
        while True:
            try:
                response = await client.beta.conversations.start_async(
                    model=model,
                    inputs=inputs,
                    instructions=IMAGE_GENERATION_INSTRUCTIONS,
                    tools=[{"type": "image_generation"}],
                    store=False,
                )
                break
            except SDKError as err:
                if err.status_code == 429 and retries < BACKOFF_MAX_RETRIES:
                    await async_backoff(retries, "image generation")
                    retries += 1
                    continue
                if err.status_code == 401:
                    _LOGGER.error("Authentication error with Mistral: %s", err)
                    raise HomeAssistantError(
                        "Authentication error with Mistral"
                    ) from err
                _LOGGER.error("Error generating image with Mistral: %s", err)
                raise HomeAssistantError(
                    f"Error generating image: {err.message}"
                ) from err

        response_text = ""
        image_chunk: ToolFileChunk | None = None
        for output in response.outputs:
            if not isinstance(output, MessageOutputEntry):
                continue
            if isinstance(output.content, str):
                response_text += output.content
                continue
            for chunk in output.content:
                if isinstance(chunk, ToolFileChunk):
                    if image_chunk is None:
                        image_chunk = chunk
                    else:
                        _LOGGER.warning(
                            "Prompt generated multiple images, using the first one"
                        )
                elif isinstance(chunk, TextChunk):
                    response_text += chunk.text

        if image_chunk is None:
            raise HomeAssistantError("Response did not include an image")

        try:
            download = await client.files.download_async(file_id=image_chunk.file_id)
            image_data = await download.aread()
        except SDKError as err:
            _LOGGER.error("Error downloading generated image: %s", err)
            raise HomeAssistantError("Error downloading generated image") from err

        mime_type = None
        if image_chunk.file_type:
            mime_type = guess_file_type(f"image.{image_chunk.file_type}")[0]
        if not mime_type:
            mime_type = "image/png"

        if response.usage is not None:
            chat_log.async_trace(
                {
                    "stats": {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                    }
                }
            )

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=response_text,
            )
        )

        return ai_task.GenImageTaskResult(
            image_data=image_data,
            conversation_id=chat_log.conversation_id,
            mime_type=mime_type,
            model=model,
        )

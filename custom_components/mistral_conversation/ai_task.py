"""AI Task support for Mistral."""

import json
from json import JSONDecodeError

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
)

from . import MistralConfigEntry
from .const import CONF_PROMPT, DOMAIN
from .entity import MAX_TOOL_ITERATIONS, MistralBaseLLMEntity


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

    _attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA

    def __init__(self, entry: MistralConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        options = self.subentry.data

        await chat_log.async_provide_llm_data(
            task.as_llm_context(DOMAIN),
            options.get(CONF_LLM_HASS_API),
            options.get(CONF_PROMPT),
        )

        if task.structure:
            chat_log.async_add_user_content(
                conversation.UserContent(
                    content=task.instructions,
                    role="user",
                )
            )

            schema_desc = (
                f"Respond with a JSON object that matches this schema: {task.structure}"
            )
            chat_log.extra_system_prompt = schema_desc

        await self._async_handle_chat_log(
            chat_log, task.name, task.structure, max_iterations=MAX_TOOL_ITERATIONS
        )

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation=conversation.async_get_result_from_chat_log(
                    None, chat_log
                ),
                data=None,
            )

        text = chat_log.content[-1]
        if not isinstance(text, conversation.AssistantContent):
            raise HomeAssistantError("Expected assistant response from Mistral")

        content = text.content or ""

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:]  # Remove opening ```json or ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
        except (JSONDecodeError, ValueError) as err:
            raise HomeAssistantError(
                f"Failed to parse Mistral response as JSON: {err}"
            ) from err

        return ai_task.GenDataTaskResult(
            conversation=conversation.async_get_result_from_chat_log(None, chat_log),
            data=data,
        )

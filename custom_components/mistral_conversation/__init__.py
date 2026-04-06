"""The Mistral Conversation integration."""

from __future__ import annotations

from types import MappingProxyType

from mistralai.client import Mistral
from mistralai.client.errors import SDKError
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import (
    config_validation as cv,
    device_registry as dr,
    entity_registry as er,
    selector,
)
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_SAFE_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_NAME,
    DEFAULT_STT_NAME,
    DEFAULT_TTS_NAME,
    DOMAIN,
    LOGGER,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_SAFE_PROMPT,
    RECOMMENDED_STT_OPTIONS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_TTS_OPTIONS,
)

SERVICE_GENERATE_CONTENT = "generate_content"

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION, Platform.STT, Platform.TTS)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type MistralConfigEntry = ConfigEntry[Mistral]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Mistral Conversation."""
    await async_migrate_integration(hass)

    async def send_prompt(call: ServiceCall) -> ServiceResponse:
        """Send a prompt to Mistral and return the response."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        # Get first conversation subentry for options
        conversation_subentry = next(
            (
                sub
                for sub in entry.subentries.values()
                if sub.subentry_type == "conversation"
            ),
            None,
        )
        if not conversation_subentry:
            raise ServiceValidationError("No conversation configuration found")

        model: str = conversation_subentry.data.get(
            CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL
        )
        client: Mistral = entry.runtime_data

        try:
            response = await client.chat.complete_async(
                model=model,
                messages=[
                    {"role": "user", "content": call.data[CONF_PROMPT]},
                ],
                max_tokens=conversation_subentry.data.get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                ),
                top_p=conversation_subentry.data.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                temperature=conversation_subentry.data.get(
                    CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                ),
                safe_prompt=conversation_subentry.data.get(
                    CONF_SAFE_PROMPT, RECOMMENDED_SAFE_PROMPT
                ),
            )
        except SDKError as err:
            if err.status_code == 401:
                entry.async_start_reauth(hass)
                raise HomeAssistantError("Authentication error") from err
            raise HomeAssistantError(
                f"Error generating content: {err.message}"
            ) from err

        if not response or not response.choices:
            raise HomeAssistantError("No response returned")

        return {"text": response.choices[0].message.content}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        send_prompt,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {"integration": DOMAIN}
                ),
                vol.Required(CONF_PROMPT): cv.string,
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Set up Mistral Conversation from a config entry."""
    client = Mistral(
        api_key=entry.data[CONF_API_KEY],
        async_client=get_async_client(hass),
    )

    try:
        await client.models.list_async()
    except SDKError as err:
        if err.status_code == 401:
            raise ConfigEntryAuthFailed(err) from err
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Unload Mistral."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(hass: HomeAssistant, entry: MistralConfigEntry) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_integration(hass: HomeAssistant) -> None:
    """Migrate integration entry structure."""
    entries = sorted(
        hass.config_entries.async_entries(DOMAIN),
        key=lambda e: e.disabled_by is not None,
    )
    if not any(entry.version == 1 for entry in entries):
        return

    api_keys_entries: dict[str, tuple[MistralConfigEntry, bool]] = {}
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)

    for entry in entries:
        use_existing = False
        subentry = ConfigSubentry(
            data=entry.options,
            subentry_type="conversation",
            title=entry.title,
            unique_id=None,
        )
        if entry.data[CONF_API_KEY] not in api_keys_entries:
            use_existing = True
            all_disabled = all(
                e.disabled_by is not None
                for e in entries
                if e.data[CONF_API_KEY] == entry.data[CONF_API_KEY]
            )
            api_keys_entries[entry.data[CONF_API_KEY]] = (entry, all_disabled)

        parent_entry, all_disabled = api_keys_entries[entry.data[CONF_API_KEY]]

        hass.config_entries.async_add_subentry(parent_entry, subentry)
        conversation_entity_id = entity_registry.async_get_entity_id(
            "conversation",
            DOMAIN,
            entry.entry_id,
        )
        device = device_registry.async_get_device(
            identifiers={(DOMAIN, entry.entry_id)}
        )

        if conversation_entity_id is not None:
            conversation_entity_entry = entity_registry.entities[conversation_entity_id]
            entity_disabled_by = conversation_entity_entry.disabled_by
            if (
                entity_disabled_by is er.RegistryEntryDisabler.CONFIG_ENTRY
                and not all_disabled
            ):
                entity_disabled_by = (
                    er.RegistryEntryDisabler.DEVICE
                    if device
                    else er.RegistryEntryDisabler.USER
                )
            entity_registry.async_update_entity(
                conversation_entity_id,
                config_entry_id=parent_entry.entry_id,
                config_subentry_id=subentry.subentry_id,
                disabled_by=entity_disabled_by,
                new_unique_id=subentry.subentry_id,
            )

        if device is not None:
            device_disabled_by = device.disabled_by
            if (
                device.disabled_by is dr.DeviceEntryDisabler.CONFIG_ENTRY
                and not all_disabled
            ):
                device_disabled_by = dr.DeviceEntryDisabler.USER
            device_registry.async_update_device(
                device.id,
                disabled_by=device_disabled_by,
                new_identifiers={(DOMAIN, subentry.subentry_id)},
                add_config_subentry_id=subentry.subentry_id,
                add_config_entry_id=parent_entry.entry_id,
            )
            if parent_entry.entry_id != entry.entry_id:
                device_registry.async_update_device(
                    device.id,
                    remove_config_entry_id=entry.entry_id,
                )
            else:
                device_registry.async_update_device(
                    device.id,
                    remove_config_entry_id=entry.entry_id,
                    remove_config_subentry_id=None,
                )

        if not use_existing:
            await hass.config_entries.async_remove(entry.entry_id)
        else:
            _add_ai_task_subentry(hass, entry)
            hass.config_entries.async_update_entry(
                entry,
                title=DEFAULT_NAME,
                options={},
                version=2,
                minor_version=4,
            )


async def async_migrate_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Migrate entry."""
    LOGGER.debug("Migrating from version %s:%s", entry.version, entry.minor_version)

    if entry.version > 2:
        return False

    if entry.version == 2 and entry.minor_version < 2:
        hass.config_entries.async_update_entry(entry, minor_version=2)

    if entry.version == 2 and entry.minor_version == 2:
        _add_ai_task_subentry(hass, entry)
        hass.config_entries.async_update_entry(entry, minor_version=3)

    if entry.version == 2 and entry.minor_version == 3:
        device_registry = dr.async_get(hass)
        entity_registry = er.async_get(hass)
        devices = dr.async_entries_for_config_entry(device_registry, entry.entry_id)
        entity_entries = er.async_entries_for_config_entry(
            entity_registry, entry.entry_id
        )
        if entry.disabled_by is None:
            for device in devices:
                if device.disabled_by is not dr.DeviceEntryDisabler.CONFIG_ENTRY:
                    continue
                device_registry.async_update_device(
                    device.id,
                    disabled_by=dr.DeviceEntryDisabler.USER,
                )
            for entity in entity_entries:
                if entity.disabled_by is not er.RegistryEntryDisabler.CONFIG_ENTRY:
                    continue
                entity_registry.async_update_entity(
                    entity.entity_id,
                    disabled_by=er.RegistryEntryDisabler.DEVICE,
                )
        hass.config_entries.async_update_entry(entry, minor_version=4)

    if entry.version == 2 and entry.minor_version == 4:
        _add_tts_subentry(hass, entry)
        hass.config_entries.async_update_entry(entry, minor_version=5)

    if entry.version == 2 and entry.minor_version == 5:
        _add_stt_subentry(hass, entry)
        hass.config_entries.async_update_entry(entry, minor_version=6)

    LOGGER.debug(
        "Migration to version %s:%s successful",
        entry.version,
        entry.minor_version,
    )

    return True


def _add_ai_task_subentry(hass: HomeAssistant, entry: MistralConfigEntry) -> None:
    """Add AI Task subentry to the config entry."""
    hass.config_entries.async_add_subentry(
        entry,
        ConfigSubentry(
            data=MappingProxyType(RECOMMENDED_AI_TASK_OPTIONS),
            subentry_type="ai_task_data",
            title=DEFAULT_AI_TASK_NAME,
            unique_id=None,
        ),
    )


def _add_stt_subentry(hass: HomeAssistant, entry: MistralConfigEntry) -> None:
    """Add STT subentry to the config entry."""
    hass.config_entries.async_add_subentry(
        entry,
        ConfigSubentry(
            data=MappingProxyType(RECOMMENDED_STT_OPTIONS),
            subentry_type="stt",
            title=DEFAULT_STT_NAME,
            unique_id=None,
        ),
    )


def _add_tts_subentry(hass: HomeAssistant, entry: MistralConfigEntry) -> None:
    """Add TTS subentry to the config entry."""
    hass.config_entries.async_add_subentry(
        entry,
        ConfigSubentry(
            data=MappingProxyType(RECOMMENDED_TTS_OPTIONS),
            subentry_type="tts",
            title=DEFAULT_TTS_NAME,
            unique_id=None,
        ),
    )

"""Text-to-speech support for Mistral (Voxtral TTS)."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

from mistralai.client import Mistral
from mistralai.client.errors import SDKError

from homeassistant.components.tts import (
    ATTR_PREFERRED_FORMAT,
    ATTR_VOICE,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
)

from . import MistralConfigEntry
from .const import (
    BACKOFF_MAX_RETRIES,
    CONF_CHAT_MODEL,
    CONF_TTS_VOICE,
    LOGGER,
    RECOMMENDED_TTS_MODEL,
)
from .entity import MistralBaseLLMEntity, async_backoff


# Languages supported by Voxtral TTS
_TTS_LANGUAGES = [
    "en",
    "fr",
    "es",
    "pt",
    "it",
    "nl",
    "de",
    "hi",
    "ar",
]

# Supported audio output formats
_SUPPORTED_FORMATS = ["mp3", "wav", "flac", "opus", "pcm"]


async def _async_fetch_voices(client: Mistral) -> list[Voice]:
    """Fetch available voices from the Mistral API."""
    voices: list[Voice] = []
    offset = 0
    limit = 50
    while True:
        try:
            page = await client.audio.voices.list_async(limit=limit, offset=offset)
        except SDKError:
            LOGGER.warning("Failed to fetch voices from Mistral API")
            break
        if not page or not page.items:
            break
        for v in page.items:
            voices.append(Voice(v.id, v.name or v.id))
        if offset + len(page.items) >= page.total:
            break
        offset += len(page.items)
    return voices


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: MistralConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up TTS entities."""
    client: Mistral = config_entry.runtime_data
    voices = await _async_fetch_voices(client)

    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "tts":
            continue

        async_add_entities(
            [MistralTTSEntity(config_entry, subentry, voices)],
            config_subentry_id=subentry.subentry_id,
        )


class MistralTTSEntity(TextToSpeechEntity, MistralBaseLLMEntity):
    """Mistral TTS entity using Voxtral TTS."""

    _attr_has_entity_name = False
    _attr_supported_options = [ATTR_VOICE, ATTR_PREFERRED_FORMAT]

    def __init__(
        self,
        entry: MistralConfigEntry,
        subentry: ConfigSubentry,
        voices: list[Voice],
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        self._attr_name = subentry.title
        self._voices = voices
        # Prefer a neutral voice as the default
        self._default_voice_id = self._pick_default_voice(voices)

    @staticmethod
    def _pick_default_voice(voices: list[Voice]) -> str | None:
        """Pick a sensible default voice, preferring 'Neutral' variants."""
        if not voices:
            return None
        for v in voices:
            if "neutral" in (v.name or "").lower():
                return v.voice_id
        return voices[0].voice_id

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return _TTS_LANGUAGES

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return "en"

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Return a list of supported voices for a language."""
        return self._voices

    @property
    def default_options(self) -> Mapping[str, Any]:
        """Return a mapping with the default options."""
        voice = self.subentry.data.get(CONF_TTS_VOICE) or self._default_voice_id or ""
        return {
            ATTR_VOICE: voice,
            ATTR_PREFERRED_FORMAT: "mp3",
        }

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict | None = None,
    ) -> TtsAudioType:
        """Load TTS from Mistral Voxtral TTS."""
        options = {**self.subentry.data, **(options or {})}
        client: Mistral = self.entry.runtime_data
        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_TTS_MODEL)

        # Voice from options, subentry config, or first available
        voice_id = (
            options.get(ATTR_VOICE)
            or options.get(CONF_TTS_VOICE)
            or self._default_voice_id
            or ""
        )
        if not voice_id:
            LOGGER.warning("No voice ID configured and no voices available from API")

        # Audio output format
        response_format = options.get(ATTR_PREFERRED_FORMAT, "mp3")
        if response_format not in _SUPPORTED_FORMATS:
            original = response_format
            if response_format == "ogg":
                response_format = "opus"
            elif response_format == "raw":
                response_format = "pcm"
            else:
                response_format = "mp3"
            LOGGER.debug(
                "Unsupported format '%s', using '%s' instead",
                original,
                response_format,
            )

        for attempt in range(BACKOFF_MAX_RETRIES + 1):
            try:
                response = await client.audio.speech.complete_async(
                    model=model,
                    input=message,
                    voice_id=voice_id,
                    response_format=response_format,
                    stream=False,
                )

                if response is None or not response.audio_data:
                    LOGGER.error("No audio data returned from Mistral TTS")
                    return (None, None)

                audio_bytes = base64.b64decode(response.audio_data)
                return (response_format, audio_bytes)

            except SDKError as err:
                if err.status_code == 429 and attempt < BACKOFF_MAX_RETRIES:
                    await async_backoff(attempt, "TTS")
                    continue

                LOGGER.error("Error generating TTS audio with Mistral: %s", err)
                return (None, None)

        LOGGER.error("Rate limited by Mistral TTS after maximum retries")
        return (None, None)

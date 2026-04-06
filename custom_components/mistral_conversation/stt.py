"""Speech-to-text support for Mistral (Voxtral)."""

from __future__ import annotations

from collections.abc import AsyncIterable
import io
import wave

from mistralai.client import Mistral
from mistralai.client.errors import SDKError

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
)

from . import MistralConfigEntry
from .const import (
    BACKOFF_MAX_RETRIES,
    CONF_CHAT_MODEL,
    LOGGER,
    RECOMMENDED_STT_MODEL,
)
from .entity import MistralBaseLLMEntity, async_backoff


# Languages supported by Voxtral STT
_STT_LANGUAGES = [
    "en",
    "zh",
    "hi",
    "es",
    "ar",
    "fr",
    "pt",
    "ru",
    "de",
    "ja",
    "ko",
    "it",
    "nl",
]

# Audio formats we accept
_SUPPORTED_FORMATS = [
    stt.AudioFormats.WAV,
    stt.AudioFormats.OGG,
]

_SUPPORTED_BIT_RATES = [
    stt.AudioBitRates.BITRATE_8,
    stt.AudioBitRates.BITRATE_16,
]

_SUPPORTED_SAMPLE_RATES = [
    stt.AudioSampleRates.SAMPLERATE_16000,
    stt.AudioSampleRates.SAMPLERATE_44100,
    stt.AudioSampleRates.SAMPLERATE_48000,
]

_SUPPORTED_CHANNELS = [
    stt.AudioChannels.CHANNEL_MONO,
    stt.AudioChannels.CHANNEL_STEREO,
]

_FORMAT_EXTENSION_MAP = {
    stt.AudioFormats.WAV: "wav",
    stt.AudioFormats.OGG: "ogg",
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: MistralConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up STT entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "stt":
            continue

        async_add_entities(
            [MistralSTTEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class MistralSTTEntity(stt.SpeechToTextEntity, MistralBaseLLMEntity):
    """Mistral STT entity using Voxtral."""

    _attr_has_entity_name = False

    def __init__(self, entry: MistralConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        self._attr_name = subentry.title

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return _STT_LANGUAGES

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return a list of supported formats."""
        return _SUPPORTED_FORMATS

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return a list of supported codecs."""
        return [stt.AudioCodecs.PCM, stt.AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return a list of supported bit rates."""
        return _SUPPORTED_BIT_RATES

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return a list of supported sample rates."""
        return _SUPPORTED_SAMPLE_RATES

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return a list of supported channels."""
        return _SUPPORTED_CHANNELS

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process an audio stream to STT service."""
        # Collect all audio data using bytearray for O(n) performance
        audio_bytes = bytearray()
        async for chunk in stream:
            audio_bytes.extend(chunk)

        if not audio_bytes:
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        audio_data = bytes(audio_bytes)

        # Add WAV header for raw PCM data when format is WAV
        if metadata.format == stt.AudioFormats.WAV:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(metadata.channel.value)
                wf.setsampwidth(metadata.bit_rate.value // 8)
                wf.setframerate(metadata.sample_rate.value)
                wf.writeframes(audio_data)
            audio_data = wav_buffer.getvalue()

        client: Mistral = self.entry.runtime_data
        model = self.subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_STT_MODEL)

        ext = _FORMAT_EXTENSION_MAP.get(metadata.format, "wav")
        file_name = f"audio.{ext}"

        # Strip BCP-47 to ISO 639-1 (e.g. "en-US" -> "en")
        language = None
        if metadata.language:
            language = metadata.language.split("-")[0]

        for attempt in range(BACKOFF_MAX_RETRIES + 1):
            try:
                response = await client.audio.transcriptions.complete_async(
                    model=model,
                    file={
                        "content": audio_data,
                        "file_name": file_name,
                    },
                    language=language,
                    diarize=False,
                )

                text = response.text if response and response.text else ""
                return stt.SpeechResult(
                    text.strip(),
                    stt.SpeechResultState.SUCCESS
                    if text
                    else stt.SpeechResultState.ERROR,
                )
            except SDKError as err:
                if err.status_code == 429 and attempt < BACKOFF_MAX_RETRIES:
                    await async_backoff(attempt, "STT")
                    continue

                LOGGER.error("Error transcribing audio with Mistral: %s", err)
                return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        LOGGER.error("Rate limited by Mistral STT after maximum retries")
        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

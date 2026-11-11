"""Tests for the Mistral AI speech-to-text and text-to-speech entities."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from homeassistant.components import stt, tts
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .conftest import api_error

METADATA = stt.SpeechMetadata(
    language="en-US",
    format=stt.AudioFormats.WAV,
    codec=stt.AudioCodecs.PCM,
    bit_rate=stt.AudioBitRates.BITRATE_16,
    sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
    channel=stt.AudioChannels.CHANNEL_MONO,
)


async def _audio() -> AsyncGenerator[bytes]:
    yield b"\x00\x01" * 160


async def test_stt_transcribes_audio(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """Raw PCM audio is wrapped as WAV and sent for transcription."""
    mock_client.audio.transcriptions.complete_async = AsyncMock(
        return_value=SimpleNamespace(text=" Turn on the lights ")
    )
    entity = stt.async_get_speech_to_text_entity(hass, "stt.mistral_stt")

    result = await entity.async_process_audio_stream(METADATA, _audio())

    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "Turn on the lights"
    kwargs = mock_client.audio.transcriptions.complete_async.call_args.kwargs
    assert kwargs["model"] == "voxtral-mini-latest"
    assert kwargs["language"] == "en"
    assert kwargs["file"]["file_name"] == "audio.wav"
    assert kwargs["file"]["content"].startswith(b"RIFF")


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("No route to host"), api_error(500, "Internal error")],
    ids=["network down", "server error"],
)
async def test_stt_reports_failures(
    hass: HomeAssistant,
    mock_client: MagicMock,
    setup_integration: MockConfigEntry,
    error: Exception,
) -> None:
    """A failed transcription is an error result, not an exception."""
    mock_client.audio.transcriptions.complete_async = AsyncMock(side_effect=error)
    entity = stt.async_get_speech_to_text_entity(hass, "stt.mistral_stt")

    result = await entity.async_process_audio_stream(METADATA, _audio())

    assert result.result is stt.SpeechResultState.ERROR


async def test_tts_generates_audio(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """Speech is requested with the account's voice and decoded."""
    mock_client.audio.speech.complete_async = AsyncMock(
        return_value=SimpleNamespace(audio_data=base64.b64encode(b"MP3").decode())
    )
    entity = hass.data[tts.DATA_COMPONENT].get_entity("tts.mistral_tts")

    assert [voice.voice_id for voice in entity.async_get_supported_voices("en")] == [
        "voice-1"
    ]
    audio = await entity.async_get_tts_audio("Hello", "en", entity.default_options)

    assert audio == ("mp3", b"MP3")
    kwargs = mock_client.audio.speech.complete_async.call_args.kwargs
    assert kwargs["model"] == "voxtral-mini-tts-2603"
    assert kwargs["voice_id"] == "voice-1"
    assert kwargs["input"] == "Hello"


async def test_tts_reports_failures(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """A network failure yields no audio instead of an exception."""
    mock_client.audio.speech.complete_async = AsyncMock(
        side_effect=httpx.ConnectError("No route to host")
    )
    entity = hass.data[tts.DATA_COMPONENT].get_entity("tts.mistral_tts")

    assert await entity.async_get_tts_audio("Hello", "en", {}) == (None, None)


async def test_tts_loads_voices_later_when_setup_could_not(
    hass: HomeAssistant, mock_client: MagicMock, mock_config_entry: MockConfigEntry
) -> None:
    """A voice list that failed to load during setup is fetched on first use."""
    voices = mock_client.audio.voices.list_async.return_value
    mock_client.audio.voices.list_async.side_effect = api_error(429, "Slow down")
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    entity = hass.data[tts.DATA_COMPONENT].get_entity("tts.mistral_tts")
    assert entity.async_get_supported_voices("en") == []

    mock_client.audio.voices.list_async.side_effect = None
    mock_client.audio.voices.list_async.return_value = voices
    mock_client.audio.speech.complete_async = AsyncMock(
        return_value=SimpleNamespace(audio_data=base64.b64encode(b"MP3").decode())
    )
    audio = await entity.async_get_tts_audio("Hello", "en", {})

    assert audio == ("mp3", b"MP3")
    assert mock_client.audio.speech.complete_async.call_args.kwargs["voice_id"] == (
        "voice-1"
    )
    assert [voice.voice_id for voice in entity.async_get_supported_voices("en")] == [
        "voice-1"
    ]

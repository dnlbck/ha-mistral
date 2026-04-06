"""Constants for the Mistral Conversation integration."""

import logging
from typing import Any

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

DOMAIN = "mistral_conversation"
LOGGER: logging.Logger = logging.getLogger(__package__)

DEFAULT_CONVERSATION_NAME = "Mistral Conversation"
DEFAULT_AI_TASK_NAME = "Mistral AI Task"
DEFAULT_STT_NAME = "Mistral STT"
DEFAULT_TTS_NAME = "Mistral TTS"
DEFAULT_NAME = "Mistral Conversation"

CONF_CHAT_MODEL = "chat_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_PROMPT = "prompt"
CONF_RECOMMENDED = "recommended"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_SAFE_PROMPT = "safe_prompt"
CONF_TTS_VOICE = "tts_voice"

RECOMMENDED_CHAT_MODEL = "mistral-small-latest"
RECOMMENDED_MAX_TOKENS = 3000
RECOMMENDED_TEMPERATURE = 0.7
RECOMMENDED_TOP_P = 1.0
RECOMMENDED_SAFE_PROMPT = False
RECOMMENDED_STT_MODEL = "voxtral-mini-latest"
RECOMMENDED_TTS_MODEL = "voxtral-mini-tts-2603"

RECOMMENDED_CONVERSATION_OPTIONS: dict[str, Any] = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}
RECOMMENDED_AI_TASK_OPTIONS: dict[str, Any] = {
    CONF_RECOMMENDED: True,
}
RECOMMENDED_STT_OPTIONS: dict[str, Any] = {}
RECOMMENDED_TTS_OPTIONS: dict[str, Any] = {
    CONF_CHAT_MODEL: RECOMMENDED_TTS_MODEL,
}

# Backoff settings for 429 rate limit retries
BACKOFF_INITIAL_DELAY = 1.0  # seconds
BACKOFF_MAX_DELAY = 60.0  # seconds
BACKOFF_MULTIPLIER = 2.0
BACKOFF_MAX_RETRIES = 5

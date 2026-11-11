"""Constants for the Mistral AI integration."""

import logging
from typing import Any

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

DOMAIN = "mistral_conversation"
LOGGER: logging.Logger = logging.getLogger(__package__)

DEFAULT_CONVERSATION_NAME = "Mistral AI"
DEFAULT_AI_TASK_NAME = "Mistral AI Task"
DEFAULT_STT_NAME = "Mistral STT"
DEFAULT_TTS_NAME = "Mistral TTS"
DEFAULT_NAME = "Mistral AI"

# Appended to model labels in config flow dropdowns for vision-capable models
VISION_MODEL_MARKER = "👁️"

CONF_CHAT_MODEL = "chat_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_PROMPT = "prompt"
CONF_RECOMMENDED = "recommended"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_SAFE_PROMPT = "safe_prompt"
CONF_TTS_VOICE = "tts_voice"
CONF_WEB_SEARCH = "web_search"

# Preselected for new installs and newly added agents
RECOMMENDED_CHAT_MODEL = "mistral-large-4"
# Agents saved before 1.3.0 only stored a model when one was picked by hand.
# The others keep the defaults of that time instead of moving to the new one.
LEGACY_CHAT_MODEL = "mistral-small-latest"
RECOMMENDED_IMAGE_MODEL = "mistral-medium-latest"
RECOMMENDED_MAX_TOKENS = 3000
RECOMMENDED_TEMPERATURE = 0.7
RECOMMENDED_TOP_P = 1.0
RECOMMENDED_REASONING_EFFORT = "none"

# reasoning_effort values accepted by the Mistral chat API (mistralai SDK 3.x)
REASONING_EFFORTS: list[str] = ["none", "minimal", "low", "medium", "high", "xhigh"]
RECOMMENDED_SAFE_PROMPT = False
RECOMMENDED_STT_MODEL = "voxtral-mini-latest"
RECOMMENDED_WEB_SEARCH = False
RECOMMENDED_TTS_MODEL = "voxtral-mini-tts-2603"

RECOMMENDED_CONVERSATION_OPTIONS: dict[str, Any] = {
    CONF_RECOMMENDED: True,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}
RECOMMENDED_AI_TASK_OPTIONS: dict[str, Any] = {
    CONF_RECOMMENDED: True,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
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

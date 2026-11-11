"""Tests for the Mistral AI config and subentry flows."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from homeassistant.config_entries import SOURCE_USER, ConfigEntryState
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_conversation.const import DOMAIN

from .conftest import api_error


def _subentry_id(entry: MockConfigEntry, subentry_type: str) -> str:
    return next(
        subentry.subentry_id
        for subentry in entry.subentries.values()
        if subentry.subentry_type == subentry_type
    )


def _schema_keys(result: dict[str, Any]) -> list[str]:
    return [str(key) for key in result["data_schema"].schema]


def _model_options(result: dict[str, Any]) -> dict[str, str]:
    """Return the model dropdown of a form as {model id: label}."""
    for key, field in result["data_schema"].schema.items():
        if str(key) == "chat_model":
            return {
                option["value"]: option["label"] for option in field.config["options"]
            }
    raise AssertionError("The form has no model dropdown")


def _preselected_model(result: dict[str, Any]) -> str:
    """Return the model a form shows before the user touches it."""
    for key in result["data_schema"].schema:
        if str(key) == "chat_model":
            return key.description["suggested_value"]
    raise AssertionError("The form has no model dropdown")


@pytest.mark.usefixtures("core_integration")
async def test_user_flow_creates_all_services(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    """A valid API key creates the entry with one subentry per service."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM

    with patch(
        "custom_components.mistral_conversation.async_setup_entry", return_value=True
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "test-key"}
        )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Mistral AI"
    assert result["data"] == {CONF_API_KEY: "test-key"}
    subentries = {
        subentry.subentry_type: subentry.data
        for subentry in result["result"].subentries.values()
    }
    assert sorted(subentries) == ["ai_task_data", "conversation", "stt", "tts"]
    # New installs start on the recommended model
    assert subentries["conversation"]["chat_model"] == "mistral-large-4"
    assert subentries["ai_task_data"]["chat_model"] == "mistral-large-4"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (api_error(401, "Unauthorized"), "invalid_auth"),
        (httpx.ConnectError("No route to host"), "cannot_connect"),
        (api_error(500, "Internal error"), "unknown"),
        (RuntimeError("boom"), "unknown"),
    ],
    ids=["invalid key", "network down", "server error", "bug"],
)
@pytest.mark.usefixtures("core_integration")
async def test_user_flow_reports_errors(
    hass: HomeAssistant, mock_client: MagicMock, error: Exception, reason: str
) -> None:
    """Validation problems are shown on the form and can be corrected."""
    mock_client.models.list_async.side_effect = error
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "test-key"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": reason}


async def test_reauth_accepts_the_same_key(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Re-authenticating with the entry's own key is not a duplicate."""
    result = await setup_integration.start_reauth_flow(hass)
    assert result["step_id"] == "reauth_confirm"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "test-key"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert setup_integration.state is ConfigEntryState.LOADED
    assert "update listener" not in caplog.text


@pytest.mark.parametrize("loaded", [True, False], ids=["loaded", "setup failed"])
async def test_reauth_stores_the_new_key_and_reloads(
    hass: HomeAssistant,
    mock_client: MagicMock,
    mock_config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
    loaded: bool,
) -> None:
    """A new key is saved and the entry comes back up with it."""
    if not loaded:
        mock_client.models.list_async.side_effect = api_error(401, "Unauthorized")
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert (mock_config_entry.state is ConfigEntryState.LOADED) is loaded
    mock_client.models.list_async.side_effect = None

    result = await mock_config_entry.start_reauth_flow(hass)
    with patch(
        "custom_components.mistral_conversation.Mistral", return_value=mock_client
    ) as mock_mistral:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "new-key"}
        )
        await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert mock_config_entry.data == {CONF_API_KEY: "new-key"}
    assert mock_config_entry.state is ConfigEntryState.LOADED
    assert mock_mistral.call_args.kwargs["api_key"] == "new-key"
    assert "update listener" not in caplog.text


async def test_conversation_reconfigure_recommended(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    """The model is picked on the first page, from the account's chat models."""
    result = await setup_integration.start_subentry_reconfigure_flow(
        hass, _subentry_id(setup_integration, "conversation")
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert _model_options(result) == {
        "mistral-large-4": "mistral-large-4 👁️",
        "mistral-large-latest": "mistral-large-latest 👁️",
        "zai-glm-5-3": "zai-glm-5-3",
    }

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            "chat_model": "zai-glm-5-3",
            "prompt": "Speak like a pirate.",
            CONF_LLM_HASS_API: ["assist"],
            "recommended": True,
        },
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    subentry_id = _subentry_id(setup_integration, "conversation")
    assert dict(setup_integration.subentries[subentry_id].data) == {
        "chat_model": "zai-glm-5-3",
        "prompt": "Speak like a pirate.",
        CONF_LLM_HASS_API: ["assist"],
        "recommended": True,
    }


async def test_conversation_reconfigure_advanced(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    """Advanced settings, web search included, are saved."""
    result = await setup_integration.start_subentry_reconfigure_flow(
        hass, _subentry_id(setup_integration, "conversation")
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {"chat_model": "mistral-large-4", "recommended": False},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "advanced"

    advanced = {
        "max_tokens": 500,
        "top_p": 0.9,
        "temperature": 0.2,
        "safe_prompt": True,
        "reasoning_effort": "high",
        "web_search": True,
    }
    assert set(_schema_keys(result)) == set(advanced)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], advanced
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    subentry_id = _subentry_id(setup_integration, "conversation")
    data = dict(setup_integration.subentries[subentry_id].data)
    assert data["recommended"] is False
    assert data["chat_model"] == "mistral-large-4"
    assert {key: data[key] for key in advanced} == advanced
    # Unchecking "Control Home Assistant" removes the stored API
    assert CONF_LLM_HASS_API not in data


async def test_ai_task_has_no_web_search_setting(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    """Web search is a conversation feature only."""
    result = await setup_integration.start_subentry_reconfigure_flow(
        hass, _subentry_id(setup_integration, "ai_task_data")
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {"chat_model": "mistral-large-4", "recommended": False},
    )

    assert result["step_id"] == "advanced"
    assert "web_search" not in _schema_keys(result)


@pytest.mark.parametrize(
    "conversation_options",
    [{"recommended": True, "prompt": "Be brief.", CONF_LLM_HASS_API: ["assist"]}],
)
async def test_reconfigure_keeps_the_model_of_older_agents(
    hass: HomeAssistant, setup_integration: MockConfigEntry
) -> None:
    """An agent saved before 1.3.0 stored no model; the form shows the one it uses."""
    result = await setup_integration.start_subentry_reconfigure_flow(
        hass, _subentry_id(setup_integration, "conversation")
    )

    assert _preselected_model(result) == "mistral-small-latest"


@pytest.mark.parametrize("subentry_type", ["conversation", "ai_task_data"])
async def test_new_agents_start_on_the_recommended_model(
    hass: HomeAssistant, setup_integration: MockConfigEntry, subentry_type: str
) -> None:
    """A newly added agent has the recommended model preselected."""
    result = await hass.config_entries.subentries.async_init(
        (setup_integration.entry_id, subentry_type), context={"source": SOURCE_USER}
    )

    assert _preselected_model(result) == "mistral-large-4"


@pytest.mark.parametrize(
    ("subentry_type", "user_input", "expected_models"),
    [
        (
            "ai_task_data",
            {
                "name": "Vision tasks",
                "chat_model": "mistral-large-4",
                "recommended": True,
            },
            {"mistral-large-4", "mistral-large-latest", "zai-glm-5-3"},
        ),
        (
            "stt",
            {"name": "Transcriber", "chat_model": "voxtral-mini-latest"},
            {"voxtral-mini-latest"},
        ),
        (
            "tts",
            {"name": "Speaker", "chat_model": "voxtral-mini-tts-2603"},
            {"voxtral-mini-tts-2603"},
        ),
    ],
)
async def test_add_subentry(
    hass: HomeAssistant,
    setup_integration: MockConfigEntry,
    subentry_type: str,
    user_input: dict[str, Any],
    expected_models: set[str],
) -> None:
    """Each service type can be added with a model fit for it."""
    result = await hass.config_entries.subentries.async_init(
        (setup_integration.entry_id, subentry_type), context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert set(_model_options(result)) == expected_models

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], user_input
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == user_input["name"]
    assert result["data"] == {
        key: value for key, value in user_input.items() if key != "name"
    }


async def test_model_list_falls_back_when_mistral_is_unreachable(
    hass: HomeAssistant, mock_client: MagicMock, setup_integration: MockConfigEntry
) -> None:
    """The flow still offers models when the account's list cannot be fetched."""
    mock_client.models.list_async.side_effect = httpx.ConnectError("offline")
    with patch(
        "custom_components.mistral_conversation.models.MODEL_LIST_TTL",
        setup_integration.created_at - setup_integration.created_at,
    ):
        result = await setup_integration.start_subentry_reconfigure_flow(
            hass, _subentry_id(setup_integration, "stt")
        )

    assert result["type"] is FlowResultType.FORM
    assert "voxtral-mini-latest" in _model_options(result)

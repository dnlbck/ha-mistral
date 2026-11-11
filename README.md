# Mistral AI

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=dnlbck&repository=ha-mistral&category=integration)
[![GitHub Release](https://img.shields.io/github/v/release/dnlbck/ha-mistral?sort=semver)](https://github.com/dnlbck/ha-mistral/releases)
[![Validate](https://github.com/dnlbck/ha-mistral/actions/workflows/validate.yml/badge.svg)](https://github.com/dnlbck/ha-mistral/actions/workflows/validate.yml)
[![Lint](https://github.com/dnlbck/ha-mistral/actions/workflows/lint.yml/badge.svg)](https://github.com/dnlbck/ha-mistral/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Home Assistant custom integration for [Mistral AI](https://mistral.ai/) — conversation, AI tasks, speech-to-text, and text-to-speech.

## Features

- **Conversation** — Chat with Mistral models with full tool/function calling and streaming, plus optional web search
- **AI Task** — Structured data generation and image generation
- **Speech-to-Text** — Transcribe audio via Voxtral (13 languages)
- **Text-to-Speech** — Synthesize speech via Voxtral TTS (9 languages)

## Requirements

- Home Assistant **2025.9.0** or newer
- A [Mistral AI API key](https://console.mistral.ai/api-keys/)

## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant instance
2. Click the three dots in the top right corner and select **Custom repositories**
3. Add this repository URL and select **Integration** as the category
4. Click **Add**
5. Search for "Mistral AI" in HACS and install it
6. Restart Home Assistant

### Manual

1. Download the latest release from GitHub
2. Copy the `custom_components/mistral_conversation` directory to your Home Assistant `config/custom_components/` directory
3. Restart Home Assistant

## Configuration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for **Mistral AI**
3. Enter your Mistral AI API key
4. The integration creates four sub-entries by default — you can reconfigure or add more under the integration's options:
   - **Conversation** — Pick the model first (the list comes straight from your Mistral account; `mistral-large-4` is preselected), then set instructions, which Home Assistant APIs the model can call, and optional sampling parameters
   - **AI Task** — Structured data generation (the structure you ask for is sent to the model as a JSON schema) and image generation via Mistral's image generation tool (uses the agent's configured model)
   - **STT** — Audio transcription via Voxtral
   - **TTS** — Speech synthesis via Voxtral TTS (pick the model here; the voice is chosen when the engine speaks, e.g. in an Assist pipeline)

To reconfigure a sub-entry, click the three dots next to it and choose **Reconfigure**. Each conversation agent keeps its own model, so you can run different models in different Assist pipelines.

`mistral-large-4` is only the default for new installs and newly added agents. Updating the integration never switches the model of an agent you already have: agents set up before 1.3.0 keep running on the model they used until you pick another one.

## Model capabilities

The model dropdown in each agent's settings is fetched live from your Mistral account, with vision-capable models marked with **👁️**. What each capability means for this integration:

| Capability | Where it matters |
|------------|------------------|
| **Vision (👁️)** | Photos and camera snapshots sent to Assist (e.g. asking what's in an image). Text-only models — including the hosted Z.ai GLM models (`zai-glm-5-3`, `zai-glm-5-2`) — reject images with a clear error. |
| **Function calling** | Controlling Home Assistant through Assist ("turn off the kitchen lights"). Supported by most chat models, including the GLM models. |
| **Image generation** | AI Task image generation. Works with any chat-capable model; Mistral's built-in image tool (FLUX) does the drawing. |
| **Reasoning** | The "Reasoning effort" advanced option: `none`, `minimal`, `low`, `medium`, `high` or `xhigh`. Higher levels think longer and use more tokens. |
| **Web search** | The "Web search" advanced option of a conversation agent. Its answers then come from Mistral's Conversations API, which can look things up on the web and still control Home Assistant. Nothing is stored on Mistral's side. The "Safe prompt" option does not apply to these requests. |
| **Audio transcription / speech** | STT and TTS sub-entries (Voxtral models). |

As of October 2026, the vision-capable chat models include Mistral Large 4 (natively multimodal, the integration's default), Mistral Medium 3.5, Mistral Small 4, and the Ministral 3 family (the old Pixtral models are retired). GLM 5.3 is a strong pick for text chat and Home Assistant control at a lower price — just don't use it where images are involved.

## Services

### `mistral_conversation.generate_content`

Send a prompt to Mistral and get a response.

| Field | Required | Description |
|-------|----------|-------------|
| `config_entry` | Yes | The Mistral AI config entry to use |
| `prompt` | Yes | The prompt to send to Mistral |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| **Invalid API key** | Double-check the key at [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys/). Re-authenticate via **Reconfigure** on the main integration entry. |
| **Rate limited (429)** | The integration retries automatically with exponential backoff. If it persists, check your Mistral plan limits. |
| **No audio from TTS** | Voices are fetched from your Mistral account and picked when the engine speaks (e.g. in an Assist pipeline). If no voices load, check your API key has access to Voxtral TTS. |
| **"… does not support images"** | The agent's model is text-only (e.g. a GLM model). Reconfigure the agent and pick a model marked 👁️, or send the question without the image. |
| **"Could not reach Mistral"** | Home Assistant had no connection to the Mistral API. If this happens while Home Assistant starts, the integration retries on its own once the network is back. |
| **Web search has no effect** | Look for "answering without web search" in the log. Mistral's Conversations API refused the request, usually because the selected model cannot be used there, and the agent answered through the regular chat API instead. Pick a Mistral model such as Mistral Large or Medium. |

## Development

Lint with `ruff check .` and `ruff format --check .` (the ruff version is pinned in `.github/lint-requirements.txt`).

The tests run the integration inside a real Home Assistant core with a mocked Mistral client. They need Python 3.14 on Linux, macOS or WSL, because Home Assistant itself does not run on Windows:

```bash
pip install -r .github/test-requirements.txt
python tests/requirements.py > /tmp/requirements.txt
pip install -r /tmp/requirements.txt
pytest
```

The first file pins the Home Assistant release the tests run against; the second step installs what Home Assistant would otherwise install at runtime.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

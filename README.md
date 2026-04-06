# Mistral Conversation

[![HACS Validation](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Home Assistant custom integration for [Mistral AI](https://mistral.ai/) — conversation, AI tasks, speech-to-text, and text-to-speech.

## Features

- **Conversation** — Chat with Mistral models with full tool/function calling and streaming
- **AI Task** — Structured data generation
- **Speech-to-Text** — Transcribe audio via Voxtral (13 languages)
- **Text-to-Speech** — Synthesize speech via Voxtral TTS (9 languages)

## Requirements

- Home Assistant **2025.7.0** or newer
- A [Mistral AI API key](https://console.mistral.ai/api-keys/)

## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant instance
2. Click the three dots in the top right corner and select **Custom repositories**
3. Add this repository URL and select **Integration** as the category
4. Click **Add**
5. Search for "Mistral Conversation" in HACS and install it
6. Restart Home Assistant

### Manual

1. Download the latest release from GitHub
2. Copy the `custom_components/mistral_conversation` directory to your Home Assistant `config/custom_components/` directory
3. Restart Home Assistant

## Configuration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for **Mistral Conversation**
3. Enter your Mistral AI API key
4. The integration creates four sub-entries by default — you can reconfigure or add more under the integration's options:
   - **Conversation** — Model, temperature, system prompt, and which Home Assistant APIs the model can call
   - **AI Task** — Structured data generation (JSON output from a prompt)
   - **STT** — Audio transcription via Voxtral
   - **TTS** — Speech synthesis via Voxtral TTS (pick a voice or leave blank for the default)

To reconfigure a sub-entry, click the three dots next to it and choose **Reconfigure**.

## Services

### `mistral_conversation.generate_content`

Send a prompt to Mistral and get a response.

| Field | Required | Description |
|-------|----------|-------------|
| `config_entry` | Yes | The Mistral Conversation config entry to use |
| `prompt` | Yes | The prompt to send to Mistral |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| **Invalid API key** | Double-check the key at [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys/). Re-authenticate via **Reconfigure** on the main integration entry. |
| **Rate limited (429)** | The integration retries automatically with exponential backoff. If it persists, check your Mistral plan limits. |
| **No audio from TTS** | Make sure a voice is configured in the TTS sub-entry. If no voices load, check your API key has access to Voxtral TTS. |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

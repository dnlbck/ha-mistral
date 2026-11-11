"""Print the requirements the tests need on top of the test harness.

Home Assistant installs these at runtime, the test harness does not. They are
read from the manifests, so they always match the integration and the Home
Assistant release the harness pins:

    python tests/requirements.py > /tmp/requirements.txt
    pip install -r /tmp/requirements.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import homeassistant

# Home Assistant components the integration is built on
COMPONENTS = ("ai_task", "camera", "conversation", "ffmpeg", "stt", "tts")


def main() -> None:
    """Print one requirement per line."""
    repository = Path(__file__).parent.parent
    components = Path(homeassistant.__file__).parent / "components"
    manifests = [
        repository / "custom_components" / "mistral_conversation" / "manifest.json",
        *(components / name / "manifest.json" for name in COMPONENTS),
    ]
    requirements: set[str] = set()
    for manifest in manifests:
        requirements.update(
            json.loads(manifest.read_text(encoding="utf-8")).get("requirements", [])
        )
    print("\n".join(sorted(requirements)))


if __name__ == "__main__":
    main()

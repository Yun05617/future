"""Confirm the running showcase has the updated portrait and voice-turn hooks."""

import json
from urllib.request import urlopen


def verify_showcase_config(config: dict) -> None:
    headers = [
        item for item in config.get("components", [])
        if item.get("props", {}).get("elem_id") == "companion-header"
    ]
    voice_turns = [
        item for item in config.get("dependencies", [])
        if item.get("api_name") == "_run_turn"
    ]
    if not headers or "companion-turn-start" not in headers[0]["props"].get("js_on_load", ""):
        raise ValueError("An older showcase is still running: updated portrait is missing.")
    if not voice_turns or "companion-turn-start" not in (voice_turns[0].get("js") or ""):
        raise ValueError("An older showcase is still running: portrait fade hook is missing.")


if __name__ == "__main__":
    with urlopen("http://127.0.0.1:7862/config", timeout=5) as response:
        verify_showcase_config(json.load(response))
    print("Verified: the updated showcase is loaded.")

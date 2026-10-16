import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("check_showcase_ready", Path("scripts/check_showcase_ready.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_startup_rejects_old_showcase():
    with pytest.raises(ValueError, match="older showcase"):
        module.verify_showcase_config({"components": [], "dependencies": []})


def test_startup_accepts_updated_showcase():
    from speech_emotion.showcase_ui import build_showcase_demo

    module.verify_showcase_config(build_showcase_demo().get_config_file())

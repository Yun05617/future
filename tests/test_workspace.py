import numpy as np
import pytest

from speech_emotion.workspace import ProjectWorkspace, WorkspaceError


def test_archive_input_keeps_original_and_assigns_unique_paths(tmp_path, wav_factory):
    source = wav_factory("voice.wav", np.ones(16_000, dtype=np.float32))
    workspace = ProjectWorkspace(tmp_path / "專題研究")

    first = workspace.archive_input(source, "upload")
    second = workspace.archive_input(source, "upload")

    assert first.original_path.exists()
    assert second.original_path.exists()
    assert first.original_path != second.original_path
    assert first.original_path.parent.name == "自動留存"


def test_write_json_does_not_allow_category_to_escape_workspace(tmp_path):
    workspace = ProjectWorkspace(tmp_path / "專題研究")

    with pytest.raises(WorkspaceError):
        workspace.write_json("../outside", {"record_id": "x"})

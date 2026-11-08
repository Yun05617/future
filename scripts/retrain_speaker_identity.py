"""Rebuild local profiles and learn an identity head without starting the UI."""

import argparse
import json
from pathlib import Path
import sys
import torch

import runtime_env

sys.path.insert(0, str(runtime_env.PROJECT_ROOT / 'src'))

from speech_emotion.speaker_identity import (
    SpeechBrainEmbedder, rebuild_speaker_profiles, speaker_training_summary,
)
from speech_emotion.workspace import ProjectWorkspace


class ProgressEmbedder(SpeechBrainEmbedder):
    def __init__(self):
        super().__init__()
        self.processed = 0

    def embed(self, path):
        result = super().embed(path)
        self.processed += 1
        if self.processed == 1 or self.processed % 10 == 0:
            print(f'已擷取 {self.processed} 段聲紋特徵', flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    workspace = ProjectWorkspace(args.workspace)
    rebuild_speaker_profiles(workspace, ProgressEmbedder())
    summary = speaker_training_summary(workspace)
    summary['quality_warning_count'] = len(summary.pop('quality_warnings', []))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

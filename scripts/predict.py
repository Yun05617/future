"""Run one saved speech-emotion model on one WAV file."""

import argparse
import json
from pathlib import Path

from speech_emotion.prediction import predict_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/speech_emotion.joblib"))
    parser.add_argument("--utterance-id", default="demo-001")
    args = parser.parse_args()
    print(json.dumps(predict_file(args.audio_path, args.model, args.utterance_id), ensure_ascii=False))


if __name__ == "__main__":
    main()

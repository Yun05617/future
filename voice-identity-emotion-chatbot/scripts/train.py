"""Train the frozen XLS-R plus SVC classifier from a validated CSV manifest."""

import argparse
from pathlib import Path

from speech_emotion.manifest import load_manifest
from speech_emotion.training import features_for_path, save_model, train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/speech_emotion.joblib"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    pipeline, metrics = train_pipeline(manifest, args.audio_root)
    first_path = args.audio_root / manifest.iloc[0]["path"]
    feature_names = list(features_for_path(first_path).index)
    save_model(pipeline, feature_names, args.output)
    print(metrics)
    print(f"saved model: {args.output}")


if __name__ == "__main__":
    main()

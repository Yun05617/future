"""Export local pitch-shifted WAV copies and a provenance manifest."""

import argparse
from pathlib import Path

import runtime_env  # noqa: F401

from speech_emotion.pitch_augmentation import PitchAugmentationError, export_pitch_batch


def main():
    parser = argparse.ArgumentParser(description='批次產生音高擴增副本，不修改原始音檔或模型。')
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--semitones', type=int, default=4)
    parser.add_argument('--emotion', choices=['positive', 'negative', 'calm'])
    args = parser.parse_args()
    try:
        result = export_pitch_batch(args.inputs, args.output_root, args.semitones, args.emotion)
    except (PitchAugmentationError, OSError) as error:
        parser.error(str(error))
    print(f'Created {len(result.audio_paths)} experimental copies; shift {result.semitones:+d} semitones.')
    print(result.archive_path)
    print('Originals and speaker profiles were not changed. Identity preservation is not verified.')


if __name__ == '__main__':
    main()

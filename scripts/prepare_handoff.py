"""Rebase copied training JSON paths after moving the team handoff folder."""
import json
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def relocate(value, root):
    if isinstance(value, list):
        return [relocate(item, root) for item in value]
    if isinstance(value, dict):
        return {key: relocate(item, root) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    normalized = value.replace('\\', '/')
    groups = [('/data/raw/', 'data/raw/'), ('/data/processed/', 'data/processed/'),
              ('/models/', 'models/'), ('/01_原始音檔/', 'workspace/01_原始音檔/'),
              ('/03_語音身分辨識/', 'workspace/03_語音身分辨識/'),
              ('/09_資料擴增/', 'workspace/09_資料擴增/')]
    for marker, relative in groups:
        if marker in normalized:
            candidate = (root / relative / normalized.split(marker, 1)[1]).resolve()
            if not candidate.is_relative_to(root.resolve()):
                raise ValueError('A copied training path escapes the project folder.')
            return candidate.as_posix()
    return value


def prepare(root=ROOT):
    root = root.resolve()
    updated = 0
    for relative in ('data/raw', 'data/processed', 'workspace/03_語音身分辨識', 'workspace/09_資料擴增'):
        for path in (root / relative).rglob('*.json'):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('Unexpected training metadata link.')
            try:
                data = json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError, UnicodeError):
                continue
            rebased = relocate(data, root)
            if rebased != data:
                temporary = path.with_name('.handoff-' + uuid4().hex + '.tmp')
                temporary.write_text(json.dumps(rebased, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                temporary.replace(path)
                updated += 1
    return updated


if __name__ == '__main__':
    print(f'Prepared training metadata: {prepare()} files updated. Learned weights and audio unchanged.')

"""Import only the bundled Gemma model; never copy credentials or overwrite a model."""
import argparse
import hashlib
import json
from pathlib import Path
import re

MANIFEST = Path('manifests/registry.ollama.ai/library/gemma4/12b')


def import_models(source, destination):
    payload = (source / MANIFEST).read_bytes()
    manifest = json.loads(payload)
    for item in [manifest['config'], *manifest['layers']]:
        digest = item['digest']
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
            raise ValueError('Invalid model digest.')
        name = digest.replace(':', '-')
        original = source / 'blobs' / name
        target = destination / 'blobs' / name
        if original.stat().st_size != item['size']:
            raise ValueError('Model source size mismatch.')
        if target.exists():
            with target.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest[7:]:
                    raise ValueError('An existing model blob has different content; nothing overwritten.')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            digest_check = hashlib.sha256()
            with original.open('rb') as src, target.open('xb') as dst:
                created = True
                while block := src.read(16 * 1024 * 1024):
                    digest_check.update(block)
                    dst.write(block)
            if digest_check.hexdigest() != digest[7:]:
                raise ValueError('Model checksum mismatch.')
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
    manifest_target = destination / MANIFEST
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    if manifest_target.exists():
        if manifest_target.read_bytes() != payload:
            raise ValueError('An existing Gemma manifest differs; nothing overwritten.')
    else:
        with manifest_target.open('xb') as stream:
            stream.write(payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'offline-models/ollama')
    parser.add_argument('--destination', type=Path, default=Path.home() / '.ollama/models')
    args = parser.parse_args()
    import_models(args.source, args.destination)
    print('Imported gemma4:12b. Restart Ollama, then check: ollama list')

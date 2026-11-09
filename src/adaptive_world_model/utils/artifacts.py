"""Saved configurations/source, explicit invalid values and append-only metrics."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil


def sanitized(value):
    if isinstance(value, dict): return {str(key): sanitized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [sanitized(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    if hasattr(value, 'item'): return sanitized(value.item())
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitized(value), indent=2, allow_nan=False), encoding='utf-8')


def append_jsonl(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    invalid = []
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value): invalid.append(key)
    if invalid: row = {**row, 'invalid_numeric_fields': invalid}
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(sanitized(row), allow_nan=False)+'\n')


def snapshot_sources(root, destination):
    root, destination = Path(root), Path(destination)
    hashes = {}
    files = [root/'pyproject.toml', root/'requirements.txt', root/'README.md',
             *sorted((root/'src').rglob('*.py')), *sorted((root/'scripts').rglob('*.py')),
             *sorted((root/'configs/cartpole').glob('*.json')),
             *sorted((root/'docs').glob('*.md'))]
    for path in files:
        if not path.is_file(): continue
        relative = path.relative_to(root); target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(destination.parent/'source_hashes.json', hashes)
    (destination.parent/'source_snapshot_or_commit.txt').write_text(
        'Source snapshot: source_snapshot/\nSHA256 manifest: source_hashes.json\n'
        'Root workspace is not a Git repository; external/reference_repos is separately versioned.\n', encoding='utf-8')


def new_run_directory(root, config, name='cartpole_v1'):
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = Path(root)/'results/cartpole'/f'{timestamp}_{name}'
    path.mkdir(parents=True, exist_ok=False)
    for directory in ['checkpoints', 'figures', 'logs']: (path/directory).mkdir()
    write_json(path/'config.json', config)
    # JSON is a YAML 1.2 subset; this view needs no extra parser dependency.
    write_json(path/'config.yaml', config)
    snapshot_sources(root, path/'source_snapshot')
    return path

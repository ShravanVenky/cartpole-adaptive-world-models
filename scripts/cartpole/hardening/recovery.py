"""Explicit provenance for the IO-only repair; original run snapshots stay immutable."""
import datetime as dt
import json
from pathlib import Path

from diagnostics.context import file_hash
from .storage import atomic_write, save_json


REPAIR_RECORD = 'io_repair_revision.json'
REPAIR_SNAPSHOT = 'io_repair_source_snapshot'
LEGACY_IO_FILES = {str(Path(name)): sha for name, sha in {
    'scripts/cartpole/hardening/common.py': 'a77c2cc1454807d4096aa28c7a56e7bce6d7ffd8b4e6d322060da73b27fe18d0',
    'scripts/cartpole/hardening/runner.py': '2ea77e686c9e4fca15d4b889cfeb144c8da57524f1109c5c67bf1ff5a97bffa0',
}.items()}
REPAIR_PATHS = {*LEGACY_IO_FILES, str(Path('scripts/cartpole/hardening/storage.py')),
                str(Path('scripts/cartpole/hardening/recovery.py'))}


def changed_paths(original, current):
    return {name for name in set(original) | set(current) if original.get(name) != current.get(name)}


def expected_sources(run):
    run = Path(run)
    original = json.loads((run / 'source_hashes.json').read_text(encoding='utf-8'))
    revision_path = run / REPAIR_RECORD
    if not revision_path.exists():
        return original
    revision = json.loads(revision_path.read_text(encoding='utf-8'))
    current = revision['sources']
    changed = changed_paths(original, current)
    if (revision.get('kind') != 'hardening_io_retry_v1'
            or revision.get('base_manifest_sha256') != file_hash(run / 'source_hashes.json')
            or revision.get('config_sha256') != file_hash(run / 'config_locked.json')
            or not set(original).issubset(current)
            or not changed or not changed.issubset(REPAIR_PATHS)
            or revision.get('changed_sources') != sorted(changed)
            or any(original.get(name) != sha for name, sha in LEGACY_IO_FILES.items())):
        raise RuntimeError('Invalid IO repair provenance; original source/config locks remain required')
    return current


def verify_sources(run, current):
    run = Path(run)
    expected = expected_sources(run)
    if current != expected:
        raise RuntimeError('Source version mismatch; an explicitly recorded IO repair is required for an older run')
    original = json.loads((run / 'source_hashes.json').read_text(encoding='utf-8'))
    for name, sha in original.items():
        snapshot = run / 'source_snapshot' / name
        if not snapshot.is_file() or file_hash(snapshot) != sha:
            raise RuntimeError(f'Original source snapshot missing/changed: {snapshot}')
    for name in changed_paths(original, expected):
        snapshot = run / REPAIR_SNAPSHOT / name
        if not snapshot.is_file() or file_hash(snapshot) != expected[name]:
            raise RuntimeError(f'IO repair snapshot missing/changed: {snapshot}')
    return expected


def register_io_repair(run, root, current):
    """Explicit one-time migration after reviewing the infrastructure-only diff.

    Not called automatically by prepare or resume. Only the exact legacy IO
    version is eligible; scientific sources/configuration cannot be changed.
    """
    run, root = Path(run), Path(root)
    if (run / REPAIR_RECORD).exists():
        verify_sources(run, current)
        return
    original = json.loads((run / 'source_hashes.json').read_text(encoding='utf-8'))
    changed = changed_paths(original, current)
    if (not changed or not changed.issubset(REPAIR_PATHS)
            or not set(original).issubset(current)
            or any(original.get(name) != sha for name, sha in LEGACY_IO_FILES.items())):
        raise RuntimeError('This source change is not the supported IO-only repair')
    # Verify the complete original archive before authorizing any continuation.
    verify_sources(run, original)
    for name in sorted(changed):
        payload = (root / name).read_bytes()
        if file_hash(root / name) != current[name]:
            raise RuntimeError(f'Source changed during IO repair registration: {name}')
        target = run / REPAIR_SNAPSHOT / name
        atomic_write(target, lambda stream, payload=payload: stream.write(payload))
    save_json(run / REPAIR_RECORD, dict(
        kind='hardening_io_retry_v1', registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        reason='Bounded sharing-lock retries, advisory progress, failure details and check-only resume verification. '
               'No scientific configuration, optimizer, simulator, controller, seed or checkpoint changes.',
        base_manifest_sha256=file_hash(run / 'source_hashes.json'),
        config_sha256=file_hash(run / 'config_locked.json'), changed_sources=sorted(changed), sources=current))
    verify_sources(run, current)

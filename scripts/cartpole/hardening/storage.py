"""Atomic hardening outputs with bounded retries for Windows sharing locks."""
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from adaptive_world_model.utils.artifacts import sanitized


RETRY_DELAYS = (.05, .1, .2, .4, .8, 1.6)


def _retry_locked(operation, path):
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return operation()
        except OSError as error:
            locked = isinstance(error, PermissionError) or getattr(error, 'winerror', None) in (5, 32, 33)
            if not locked or attempt == len(RETRY_DELAYS):
                error.add_note(f'Atomic output {path}; attempt {attempt + 1}; sharing-lock retries exhausted={locked}')
                raise
            time.sleep(RETRY_DELAYS[attempt])


def atomic_write(path, writer):
    """Publish only a closed, complete file; never truncate the previous output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = _retry_locked(
        lambda: tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent), path)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
        _retry_locked(lambda: os.replace(temporary, path), path)
    finally:
        # A rejected publish must leave the previous output intact. Cleanup is
        # advisory too: a scanner may briefly hold the unique temporary file.
        try:
            temporary.unlink(missing_ok=True)
        except OSError as error:
            warnings.warn(f'Could not remove temporary output {temporary}: {error}', RuntimeWarning)


def save_json(path, data):
    payload = json.dumps(sanitized(data), indent=2, allow_nan=False).encode('utf-8')
    atomic_write(path, lambda stream: stream.write(payload))


def save_npz(path, **arrays):
    atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))


def save_advisory_json(path, data):
    """Progress/telemetry is advisory; committed job pairs remain authoritative."""
    try:
        save_json(path, data)
    except OSError as error:
        warnings.warn(f'Advisory output was not updated: {path}: {error!r}. '
                      'Saved scientific jobs remain authoritative.', RuntimeWarning)
        return False
    return True

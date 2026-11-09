"""Copy the original archived paths into a new directory. Never executes them."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='New empty directory for restored legacy files')
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists(): parser.error('Destination must not already exist; archived/original files will not be overwritten')
    manifest = json.loads((ROOT/'archive/pendulum/archive_manifest.json').read_text())
    # Inspect the recorded schema rather than guessing old locations.
    entries = manifest['files']
    destination.mkdir(parents=True)
    for item in entries:
        source = ROOT/item['archived']
        target = destination/item['original']
        if destination not in target.resolve().parents:
            raise ValueError('Archive manifest path escapes destination')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f'Restored {len(entries)} files to {destination}. No legacy script was executed.')


if __name__ == '__main__': main()

"""Add the explicitly requested V1 freeze note and a separate immutable manifest."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT/'results/cartpole/20261005_232356_821640_cartpole_v1'
INDEX = ROOT/'archive/cartpole_v1/v1_hashes.json'


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): value.update(chunk)
    return value.hexdigest()


def verify():
    saved = json.loads(INDEX.read_text(encoding='utf-8'))
    failures = [name for name, digest in saved['files'].items()
                if not (BASELINE/name).is_file() or sha256(BASELINE/name) != digest]
    if failures: raise RuntimeError(f'Frozen V1 result changed: {failures[:10]}')
    return saved


def main():
    if INDEX.exists():
        saved = verify(); print(f"Frozen V1 unchanged: {len(saved['files'])} files"); return
    if not json.loads((BASELINE/'status.json').read_text())['complete']:
        raise RuntimeError('Only a completed result may become the immutable baseline')
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    original = {str(p.relative_to(BASELINE)): sha256(p) for p in sorted(BASELINE.rglob('*')) if p.is_file()}
    configuration_hash = original['config.json']; source_hash = original['source_hashes.json']
    note = BASELINE/'FROZEN_RESULT.md'
    if note.exists(): raise RuntimeError('A freeze note already exists without its manifest; inspect before proceeding')
    note.write_text(f'''# Frozen CartPole V1 result

Frozen on {datetime.now(ZoneInfo('America/Chicago')).isoformat()}.

This completed five-seed result is immutable. Existing configuration, source snapshot, reports,
checkpoints, plots, raw metrics and logs will not be overwritten or retuned.
New experiments use separate run directories and reuse these exact pretrained bases.

Main sequential OOD median paired improvements over Frozen:

| Measure | Meta residual | Generic Ridge |
|---|---:|---:|
| One-step error removed | 36.31% | 24.71% |
| 100-step error removed | 15.34% | 27.89% |

Meta has 14.77% median paired one-step benefit over Ridge, but reaches the predeclared
10% benefit in only 53.75% of cases (75% target). At 100 steps it has 10.49% more
error than Ridge. It favors pole-mass/length shifts; Ridge favors cart-mass and
strong gain shifts. All five seeds completed, nominal competence passed, and no
rejected updates/nonfinite predictions were recorded. Control remains untested.

Interpretation: useful nonlinear local prediction repair, without consistent superiority
in recursive dynamics. This is the baseline for testing objective versus plasticity location.

Exact original configuration: [config.json](config.json), SHA256 `{configuration_hash}`.
Exact source version: [source_snapshot](source_snapshot/), with [source_hashes.json](source_hashes.json)
SHA256 `{source_hash}`. The workspace has no root Git commit; the saved source is authoritative.
Original records: [report](report.md), [summary](summary.json), [trial metrics](per_trial_metrics.csv),
[rollout metrics](rollout_metrics.csv), [checkpoints](checkpoints/), and [logs](logs/).

The external manifest `archive/cartpole_v1/v1_hashes.json` records every original artifact
and this added note. Its paths are relative to this run. No original file was modified.
''', encoding='utf-8')
    original['FROZEN_RESULT.md'] = sha256(note)
    INDEX.write_text(json.dumps({'baseline': str(BASELINE.relative_to(ROOT)),
                     'frozen_date': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
                     'files': original}, indent=2), encoding='utf-8')
    verify(); print(f'FROZEN_RESULT={note}; verified {len(original)} files')


if __name__ == '__main__': main()

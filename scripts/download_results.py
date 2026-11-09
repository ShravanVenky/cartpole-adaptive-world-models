"""Download checksummed public result archives; never run a model or experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def replace(source,target):
    for attempt in range(7):
        try:os.replace(source,target);return
        except PermissionError:
            if attempt==6:raise
            time.sleep(min(.05*2**attempt,1.6))


def download(artifact,cache):
    name=artifact['asset']
    if Path(name).name!=name:raise ValueError('Invalid archive name')
    target=cache/name;partial=cache/(name+'.part')
    if target.exists():
        if digest(target)==artifact['sha256']:return target
        raise RuntimeError(f'Existing cached archive has the wrong checksum: {target}')
    if partial.exists() and partial.stat().st_size>=artifact['archive_bytes']:
        if digest(partial)!=artifact['sha256']:raise RuntimeError(f'Completed partial archive has the wrong checksum: {partial}')
        replace(partial,target);return target
    for attempt in range(5):
        offset=partial.stat().st_size if partial.exists() else 0
        request=urllib.request.Request(artifact['download_url'],headers={'User-Agent':'cartpole-result-downloader'})
        if offset:request.add_header('Range',f'bytes={offset}-')
        try:
            with urllib.request.urlopen(request,timeout=120) as response:
                append=offset>0 and response.status==206
                with partial.open('ab' if append else 'wb') as output:
                    shutil.copyfileobj(response,output,length=2**20)
            if digest(partial)!=artifact['sha256']:raise RuntimeError(f'Archive checksum mismatch: {partial}')
            replace(partial,target);return target
        except (OSError,TimeoutError):
            if attempt==4:raise
            time.sleep(2**attempt)
    raise RuntimeError('Download did not complete')


def extract(artifact,archive,destination):
    expected_prefix=f"results/cartpole/{artifact['run']}/"
    destination=destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        records=[json.loads(line) for line in bundle.read('ARTIFACT_MANIFEST.jsonl').splitlines()]
        if len(records)!=artifact['files']:raise RuntimeError('Unexpected manifest size')
        for row in records:
            name=row['path']
            if not name.startswith(expected_prefix) or '..' in Path(name).parts or Path(name).is_absolute():
                raise RuntimeError(f'Invalid archive path: {name}')
            target=(destination/name).resolve()
            if not target.is_relative_to(destination):raise RuntimeError('Archive path leaves destination')
            if target.exists():
                if digest(target)!=row['sha256']:raise RuntimeError(f'Refusing to overwrite a different existing file: {target}')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name(target.name+'.download.part')
            h=hashlib.sha256()
            with bundle.open(name) as source,partial.open('wb') as output:
                for block in iter(lambda:source.read(2**20),b''):h.update(block);output.write(block)
            if h.hexdigest()!=row['sha256']:raise RuntimeError(f'Extracted file checksum mismatch: {name}')
            replace(partial,target)


def main():
    index=json.loads((ROOT/'artifacts/index.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--all',action='store_true')
    choice.add_argument('--run',choices=[a['run'] for a in index['artifacts']])
    choice.add_argument('--list',action='store_true')
    parser.add_argument('--destination',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.list:
        for artifact in index['artifacts']:
            print(f"{artifact['run']}  {artifact['archive_bytes']/2**20:.1f} MiB  {artifact['files']} files")
        return
    cache=ROOT/'.download-cache';cache.mkdir(exist_ok=True)
    for artifact in index['artifacts']:
        if not args.all and artifact['run']!=args.run:continue
        print(f"Downloading {artifact['asset']}",flush=True)
        archive=download(artifact,cache);extract(artifact,archive,args.destination)
        print(f"Verified and extracted {artifact['run']}",flush=True)


if __name__=='__main__':main()

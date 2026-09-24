#!/usr/bin/env python3
"""Verify explicitly named release ZIPs and extract each into its own directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def member_path(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or any(x in ('', '.', '..') for x in name.split('/')) or ':' in name:
        raise ValueError(f'Unsafe archive member: {name!r}')
    return path


def parse_manifest(text):
    records = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        member_path(name)
        if name in records or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid or duplicate manifest entry')
        records[name] = digest
    return records


def verify_directory(root):
    root = Path(root).resolve()
    records = parse_manifest((root / 'MANIFEST.sha256').read_text())
    for name, digest in records.items():
        path = root.joinpath(*member_path(name).parts)
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path) != digest:
            raise ValueError(f'Manifest checksum/path mismatch: {name}')
    return records


def extract_verified(archive, expected, target):
    archive, target = Path(archive), Path(target)
    if sha(archive) != expected:
        raise ValueError(f'Archive SHA-256 mismatch: {archive.name}')
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite extracted directory: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.extract-', dir=target.parent))
    try:
        with zipfile.ZipFile(archive) as source:
            files = {}
            for info in source.infolist():
                if info.is_dir():
                    member_path(info.filename.rstrip('/'))
                    continue
                member_path(info.filename)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)) or info.filename in files:
                    raise ValueError('Links, special files, or duplicate ZIP members are prohibited')
                files[info.filename] = info
            records = parse_manifest(source.read('MANIFEST.sha256').decode())
            if set(files) != set(records) | {'MANIFEST.sha256'}:
                raise ValueError('ZIP manifest must cover every non-manifest file exactly once')
            for name, info in files.items():
                destination = temporary.joinpath(*member_path(name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as incoming, destination.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
        verify_directory(temporary)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--bundle-sha256', required=True, help='Expected SHA-256 from the authenticated release checksum file')
    parser.add_argument('--attachment-dir', required=True, type=Path, help='Directory containing explicitly indexed ZIP filenames')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    extract_verified(args.bundle, args.bundle_sha256, args.output_dir / 'bundle')
    parts = json.loads((args.output_dir / 'bundle/model_artifacts.json').read_text())
    names = set()
    manifest_hashes = {}
    for part in parts:
        name = part['filename']
        if member_path(name).name != name or name in names:
            raise ValueError('Attachment filenames must be unique basenames')
        names.add(name)
        archive = args.attachment_dir / name
        if archive.stat().st_size != part['bytes']:
            raise ValueError(f'Attachment byte count mismatch: {name}')
        extract_verified(archive, part['sha256'], args.output_dir / 'attachments' / name)
        actual = verify_directory(args.output_dir / 'attachments' / name)
        if set(actual) != set(part['members']):
            raise ValueError(f'Attachment member index mismatch: {name}')
        manifest_hashes[name] = sha(args.output_dir / 'attachments' / name / 'MANIFEST.sha256')
    (args.output_dir / 'bootstrap.json').write_text(json.dumps({
        'bundle_filename': args.bundle.name, 'bundle_sha256': args.bundle_sha256,
        'bundle_manifest_sha256': sha(args.output_dir / 'bundle/MANIFEST.sha256'),
        'attachment_manifest_sha256': manifest_hashes,
        'authentication': 'Caller supplied expected hash from authenticated release; checksums alone do not establish publisher identity',
        'attachments': parts,
    }, indent=2) + '\n')
    print(f'Verified bundle: {args.output_dir / "bundle"}')


if __name__ == '__main__':
    main()

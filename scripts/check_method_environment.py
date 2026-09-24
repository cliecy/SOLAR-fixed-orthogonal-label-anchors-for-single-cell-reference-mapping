#!/usr/bin/env python3
"""Reject a method interpreter whose numerical packages differ from its lock."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('solar', 'analysis'), required=True)
    args = parser.parse_args()
    root = ROOT / 'environments' / args.method
    if Path(sys.prefix).resolve() != (root / '.venv').resolve():
        raise RuntimeError(f'Use {root}/.venv/bin/python, not {sys.executable}')
    lock = tomllib.loads((root / 'uv.lock').read_text())
    names = {'anndata', 'numpy', 'scipy', 'scikit-learn', 'torch'}
    if args.method == 'analysis':
        names |= {'pandas', 'matplotlib'}
    locked = {name: {p['version'] for p in lock['package'] if p['name'] == name} for name in names}
    if sys.version_info[:2] != ((3, 12) if args.method == 'solar' else (3, 11)):
        raise RuntimeError('Use Python 3.12 for SOLAR or Python 3.11 for analysis')
    for name in names:
        observed = version(name).split('+', 1)[0]
        if observed not in locked[name]:
            raise RuntimeError(f'{name}={observed} is not a locked version: {sorted(locked[name])}')
    if args.method == 'solar' and sys.version_info[:2] != (3, 12):
        raise RuntimeError('The released SOLAR training environment used Python 3.12')
    print(f'{args.method}: interpreter and numerical package versions match the lock')


if __name__ == '__main__':
    main()

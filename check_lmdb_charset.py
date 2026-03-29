#!/usr/bin/env python3
"""Analyze base vs extended char ratio in LMDB datasets.

Usage:
    python check_lmdb_charset.py <root_path> [--unicode_mapping ...]

Finds all LMDBs under root_path (directories containing data.mdb) and prints
the extended char percentage for each, plus an overall total.
"""
import argparse
import json
from pathlib import Path

import lmdb


def find_lmdbs(root: Path) -> list[Path]:
    """Find all LMDB directories (containing data.mdb) under root."""
    if (root / 'data.mdb').exists():
        return [root]
    return sorted(p.parent for p in root.rglob('data.mdb'))


def analyze_lmdb(lmdb_path: Path, ext_chars: set) -> tuple[int, int, int]:
    """Returns (num_samples, total_chars, ext_chars_count)."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with env.begin() as txn:
        num_samples = int(txn.get(b'num-samples').decode())
        total = 0
        ext = 0
        for idx in range(1, num_samples + 1):
            label = txn.get(f'label-{idx:09d}'.encode()).decode()
            total += len(label)
            ext += sum(1 for c in label if c in ext_chars)
    env.close()
    return num_samples, total, ext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', help='Root path (scans all LMDBs underneath)')
    parser.add_argument(
        '--unicode_mapping',
        default='confusion_pl_output/unicode_mapping.json',
        help='Path to unicode_mapping.json',
    )
    args = parser.parse_args()
    root = Path(args.root_path).expanduser().resolve()
    mapping_path = str(Path(args.unicode_mapping).expanduser().resolve())

    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    ext_chars = {v['unicode'] for v in mapping.values()}

    lmdb_dirs = find_lmdbs(root)
    if not lmdb_dirs:
        print(f'No LMDBs found under {root}')
        return

    grand_samples = 0
    grand_total = 0
    grand_ext = 0

    name_w = max(len(str(p.relative_to(root))) for p in lmdb_dirs)
    name_w = max(name_w, len('TOTAL'))

    print(f'{"Dataset":<{name_w}}  {"#samples":>8}  {"#chars":>8}  {"ext%":>6}')
    print(f'{"-" * name_w}  {"-" * 8}  {"-" * 8}  {"-" * 6}')

    for lmdb_path in lmdb_dirs:
        name = str(lmdb_path.relative_to(root))
        n_samples, n_chars, n_ext = analyze_lmdb(lmdb_path, ext_chars)
        pct = 100 * n_ext / n_chars if n_chars > 0 else 0
        print(f'{name:<{name_w}}  {n_samples:>8}  {n_chars:>8}  {pct:>5.2f}%')
        grand_samples += n_samples
        grand_total += n_chars
        grand_ext += n_ext

    grand_pct = 100 * grand_ext / grand_total if grand_total > 0 else 0
    print(f'{"-" * name_w}  {"-" * 8}  {"-" * 8}  {"-" * 6}')
    print(f'{"TOTAL":<{name_w}}  {grand_samples:>8}  {grand_total:>8}  {grand_pct:>5.2f}%')


if __name__ == '__main__':
    main()

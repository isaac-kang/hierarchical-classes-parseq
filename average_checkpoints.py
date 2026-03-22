#!/usr/bin/env python3
"""Average weights of checkpoints containing 'epoch' in their filename.

Usage:
    python average_checkpoints.py <checkpoint_folder>
"""
import sys
from pathlib import Path

import torch


def average_checkpoints(ckpt_folder: str) -> None:
    folder = Path(ckpt_folder)
    ckpt_paths = sorted(p for p in folder.glob('*.ckpt') if 'epoch' in p.name)

    if not ckpt_paths:
        print(f'No checkpoints with "epoch" in name found in {folder}')
        sys.exit(1)

    print(f'Averaging {len(ckpt_paths)} checkpoints:')
    for p in ckpt_paths:
        print(f'  {p.name}')

    # Load all state dicts
    checkpoints = [torch.load(p, map_location='cpu', weights_only=False) for p in ckpt_paths]
    n = len(checkpoints)

    # Average state_dict weights
    avg_state_dict = {}
    for key in checkpoints[0]['state_dict']:
        tensors = [ckpt['state_dict'][key].float() for ckpt in checkpoints]
        avg_state_dict[key] = torch.stack(tensors).mean(dim=0)

    # Use the last checkpoint as base (for metadata), replace state_dict
    result = checkpoints[-1].copy()
    result['state_dict'] = avg_state_dict

    out_path = folder / 'top3_wa.ckpt'
    torch.save(result, out_path)
    print(f'\nSaved averaged checkpoint -> {out_path}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <checkpoint_folder>')
        sys.exit(1)
    average_checkpoints(sys.argv[1])

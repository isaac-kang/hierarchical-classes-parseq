#!/usr/bin/env python3
"""Compute cosine similarity between base char and extended char representations.

Two representation types:
  1. char_emb: Token embedding from model.text_embed.embedding
  2. head_wt:  Classification head weight from model.head.weight

Usage:
  python tools/char_similarity.py --checkpoint <ckpt> --unicode_mapping confusion_pl_output/unicode_mapping.json
"""

import argparse
import json
import string
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strhub.models.utils import load_from_checkpoint


def get_representations(model):
    """Extract char_emb and head_wt matrices from model."""
    char_emb = model.model.text_embed.embedding.weight.detach()  # (num_tokens, embed_dim)
    head_wt = model.model.head.weight.detach()  # (num_tokens - 2, embed_dim)
    return char_emb, head_wt


def cosine_sim(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def load_pairs_from_unicode_mapping(mapping_path):
    """Load (base_char, confused_with, ext_name) pairs from unicode_mapping.json."""
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    pairs = []
    for ext_name, entry in mapping.items():
        pairs.append({
            'base_char': entry['base_char'],
            'confused_with': entry['confused_with'],
            'ext_name': ext_name,
            'unicode': entry['unicode'],
        })
    return pairs


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--unicode_mapping', default='confusion_pl_output/unicode_mapping.json',
                        help='Path to unicode_mapping.json')
    parser.add_argument('--output', default=None, help='Output file path (default: <mapping_dir>/char_similarity.json)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    charset = string.digits + string.ascii_lowercase + string.ascii_uppercase
    chars = sorted(charset)

    # Load pairs from unicode_mapping.json
    mapping_path = str(Path(args.unicode_mapping).expanduser().resolve())
    print(f'Loading unicode mapping from {mapping_path}')
    pairs = load_pairs_from_unicode_mapping(mapping_path)
    print(f'Found {len(pairs)} extended class pairs')

    # Load model
    print(f'Loading model from {args.checkpoint}')
    model = load_from_checkpoint(args.checkpoint, charset_test=charset).eval().to(args.device)

    tokenizer = model.tokenizer
    stoi = tokenizer._stoi

    char_emb, head_wt = get_representations(model)

    print(f'\nchar_emb shape: {char_emb.shape}')
    print(f'head_wt shape:  {head_wt.shape}')
    print(f'charset_train:  {model.hparams.charset_train[:20]}... ({len(model.hparams.charset_train)} chars)')
    print()

    # Group pairs by base_char for display
    grouped = defaultdict(list)
    for p in pairs:
        grouped[p['base_char']].append(p)

    results = []

    # Check if model has extended chars in tokenizer
    sample_ext = pairs[0]['unicode'] if pairs else None
    has_ext = sample_ext is not None and sample_ext in stoi
    if has_ext:
        print('Model has extended unicode chars in tokenizer — Ext Sim columns enabled')
    else:
        print('Model does NOT have extended unicode chars in tokenizer — Ext Sim columns will show N/A')
    print()

    # Define char groups
    digits = set(string.digits)
    lowercase = set(string.ascii_lowercase)
    uppercase = set(string.ascii_uppercase)

    def get_group(c):
        if c in digits:
            return digits
        if c in lowercase:
            return lowercase
        if c in uppercase:
            return uppercase
        return set()

    # Precompute per-base: avg sim with same-group chars (excluding self and confused chars)
    def compute_group_avg(base_ch, base_vec, weight_mat, max_idx):
        """Avg cosine sim of base_ch vs same-group chars, excluding self and confused chars."""
        group = get_group(base_ch)
        confused_set = {p['confused_with'] for p in grouped.get(base_ch, [])}
        exclude = {base_ch} | confused_set
        sims = []
        for other in group:
            if other in exclude:
                continue
            other_idx = stoi.get(other)
            if other_idx is None or other_idx >= max_idx:
                continue
            sims.append(cosine_sim(base_vec, weight_mat[other_idx]))
        return np.mean(sims) if sims else None

    sep = '─' * 160
    print(sep)
    print(f'│ {"Base":^5} │ {"Conf":^5} │ {"ExtClass":^10} │'
          f' {"Emb Sim":>8} │ {"ExtEmb":>8} │ {"SimExt":>8} │ {"GrpEmb":>8} │'
          f' {"‖Base‖":>7} │ {"‖Conf‖":>7} │ {"‖Ext‖":>7} │'
          f' {"Head Sim":>8} │ {"ExtHead":>8} │ {"SimExt":>8} │ {"GrpHead":>8} │'
          f' {"‖Base‖":>7} │ {"‖Conf‖":>7} │ {"‖Ext‖":>7} │')
    print(sep)

    for ch in sorted(grouped.keys()):
        base_idx = stoi.get(ch)
        if base_idx is None:
            continue

        base_emb_vec = char_emb[base_idx]
        base_head_vec = head_wt[base_idx] if base_idx < head_wt.shape[0] else None

        # Compute group avg once per base char
        grp_emb_avg = compute_group_avg(ch, base_emb_vec, char_emb, char_emb.shape[0])
        grp_head_avg = compute_group_avg(ch, base_head_vec, head_wt, head_wt.shape[0]) if base_head_vec is not None else None

        for i, p in enumerate(grouped[ch]):
            confused_char = p['confused_with']
            ext_unicode = p['unicode']
            confused_idx = stoi.get(confused_char)
            ext_idx = stoi.get(ext_unicode)
            if confused_idx is None:
                continue

            emb_sim = cosine_sim(base_emb_vec, char_emb[confused_idx])
            head_sim = None
            if base_head_vec is not None and confused_idx < head_wt.shape[0]:
                head_sim = cosine_sim(base_head_vec, head_wt[confused_idx])

            # L2 norms
            base_emb_norm = base_emb_vec.norm().item()
            conf_emb_norm = char_emb[confused_idx].norm().item()
            ext_emb_norm = None
            base_head_norm = base_head_vec.norm().item() if base_head_vec is not None else None
            conf_head_norm = head_wt[confused_idx].norm().item() if confused_idx < head_wt.shape[0] else None
            ext_head_norm = None

            ext_emb_sim = None
            ext_head_sim = None
            sim_ext_emb = None  # ext ↔ confused
            sim_ext_head = None
            if ext_idx is not None:
                ext_emb_norm = char_emb[ext_idx].norm().item()
                ext_emb_sim = cosine_sim(base_emb_vec, char_emb[ext_idx])
                sim_ext_emb = cosine_sim(char_emb[ext_idx], char_emb[confused_idx])
                if base_head_vec is not None and ext_idx < head_wt.shape[0]:
                    ext_head_norm = head_wt[ext_idx].norm().item()
                    ext_head_sim = cosine_sim(base_head_vec, head_wt[ext_idx])
                    if confused_idx < head_wt.shape[0]:
                        sim_ext_head = cosine_sim(head_wt[ext_idx], head_wt[confused_idx])

            result = {
                'base_char': ch,
                'confused_with': confused_char,
                'ext_name': p['ext_name'],
                'emb_cosine_sim': round(emb_sim, 4),
                'head_cosine_sim': round(head_sim, 4) if head_sim is not None else None,
                'ext_emb_cosine_sim': round(ext_emb_sim, 4) if ext_emb_sim is not None else None,
                'ext_head_cosine_sim': round(ext_head_sim, 4) if ext_head_sim is not None else None,
                'sim_ext_emb': round(sim_ext_emb, 4) if sim_ext_emb is not None else None,
                'sim_ext_head': round(sim_ext_head, 4) if sim_ext_head is not None else None,
                'grp_emb_avg': round(grp_emb_avg, 4) if grp_emb_avg is not None else None,
                'grp_head_avg': round(grp_head_avg, 4) if grp_head_avg is not None else None,
                'base_emb_norm': round(base_emb_norm, 4),
                'conf_emb_norm': round(conf_emb_norm, 4),
                'ext_emb_norm': round(ext_emb_norm, 4) if ext_emb_norm is not None else None,
                'base_head_norm': round(base_head_norm, 4) if base_head_norm is not None else None,
                'conf_head_norm': round(conf_head_norm, 4) if conf_head_norm is not None else None,
                'ext_head_norm': round(ext_head_norm, 4) if ext_head_norm is not None else None,
            }
            results.append(result)

            head_str = f'{head_sim:>8.4f}' if head_sim is not None else f'{"N/A":>8}'
            ext_emb_str = f'{ext_emb_sim:>8.4f}' if ext_emb_sim is not None else f'{"N/A":>8}'
            ext_head_str = f'{ext_head_sim:>8.4f}' if ext_head_sim is not None else f'{"N/A":>8}'
            sim_ext_emb_str = f'{sim_ext_emb:>8.4f}' if sim_ext_emb is not None else f'{"N/A":>8}'
            sim_ext_head_str = f'{sim_ext_head:>8.4f}' if sim_ext_head is not None else f'{"N/A":>8}'
            ext_emb_norm_str = f'{ext_emb_norm:>7.3f}' if ext_emb_norm is not None else f'{"N/A":>7}'
            ext_head_norm_str = f'{ext_head_norm:>7.3f}' if ext_head_norm is not None else f'{"N/A":>7}'
            conf_head_norm_str = f'{conf_head_norm:>7.3f}' if conf_head_norm is not None else f'{"N/A":>7}'
            # Show group avg and base norm only on first row of each base char
            if i == 0:
                grp_emb_str = f'{grp_emb_avg:>8.4f}' if grp_emb_avg is not None else f'{"N/A":>8}'
                grp_head_str = f'{grp_head_avg:>8.4f}' if grp_head_avg is not None else f'{"N/A":>8}'
                base_emb_norm_str = f'{base_emb_norm:>7.3f}'
                base_head_norm_str = f'{base_head_norm:>7.3f}' if base_head_norm is not None else f'{"N/A":>7}'
            else:
                grp_emb_str = f'{"":>8}'
                grp_head_str = f'{"":>8}'
                base_emb_norm_str = f'{"":>7}'
                base_head_norm_str = f'{"":>7}'
            prefix = f'  {ch:^3}  ' if i == 0 else f' {"":5} '
            print(f'│{prefix}│ {confused_char:^5} │ {p["ext_name"]:<10} │'
                  f' {emb_sim:>8.4f} │ {ext_emb_str} │ {sim_ext_emb_str} │ {grp_emb_str} │'
                  f' {base_emb_norm_str} │ {conf_emb_norm:>7.3f} │ {ext_emb_norm_str} │'
                  f' {head_str} │ {ext_head_str} │ {sim_ext_head_str} │ {grp_head_str} │'
                  f' {base_head_norm_str} │ {conf_head_norm_str} │ {ext_head_norm_str} │')

        print(sep)

    # Summary statistics
    emb_sims = [r['emb_cosine_sim'] for r in results]
    head_sims = [r['head_cosine_sim'] for r in results if r['head_cosine_sim'] is not None]
    ext_emb_sims = [r['ext_emb_cosine_sim'] for r in results if r['ext_emb_cosine_sim'] is not None]
    ext_head_sims = [r['ext_head_cosine_sim'] for r in results if r['ext_head_cosine_sim'] is not None]
    sim_ext_emb_sims = [r['sim_ext_emb'] for r in results if r['sim_ext_emb'] is not None]
    sim_ext_head_sims = [r['sim_ext_head'] for r in results if r['sim_ext_head'] is not None]

    # Norm statistics
    base_emb_norms = [r['base_emb_norm'] for r in results]
    conf_emb_norms = [r['conf_emb_norm'] for r in results]
    ext_emb_norms = [r['ext_emb_norm'] for r in results if r['ext_emb_norm'] is not None]
    base_head_norms = [r['base_head_norm'] for r in results if r['base_head_norm'] is not None]
    conf_head_norms = [r['conf_head_norm'] for r in results if r['conf_head_norm'] is not None]
    ext_head_norms = [r['ext_head_norm'] for r in results if r['ext_head_norm'] is not None]

    print(f'\nStatistics (n={len(results)}):')
    print(f'  {"":20} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8} {"Median":>8}')
    for label, vals in [('Emb Sim', emb_sims), ('Head Sim', head_sims),
                         ('Ext Emb Sim', ext_emb_sims), ('Ext Head Sim', ext_head_sims),
                         ('SimExt Emb', sim_ext_emb_sims), ('SimExt Head', sim_ext_head_sims),
                         ('‖Base Emb‖', base_emb_norms), ('‖Conf Emb‖', conf_emb_norms),
                         ('‖Ext Emb‖', ext_emb_norms),
                         ('‖Base Head‖', base_head_norms), ('‖Conf Head‖', conf_head_norms),
                         ('‖Ext Head‖', ext_head_norms)]:
        if vals:
            arr = np.array(vals)
            print(f'  {label:20} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f} {np.median(arr):>8.4f}')
        else:
            print(f'  {label:20} {"N/A (no ext chars in model)":>44}')

    # Save JSON
    out_dir = Path(mapping_path).parent
    output_path = args.output or str(out_dir / 'char_similarity.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {output_path}')

    # ==================== Full char-to-char similarity heatmaps ====================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # Gather indices for charset chars (sorted)
    char_indices = []
    valid_chars = []
    for ch in chars:
        idx = stoi.get(ch)
        if idx is not None:
            char_indices.append(idx)
            valid_chars.append(ch)
    char_indices = torch.tensor(char_indices, device=args.device)
    nc = len(valid_chars)

    # Build confused pair set from unicode_mapping for overlay
    confused_pairs = set()
    for p in pairs:
        confused_pairs.add((p['base_char'], p['confused_with']))

    for repr_name, weight_matrix, max_idx in [
        ('char_emb', char_emb, char_emb.shape[0]),
        ('head_weight', head_wt, head_wt.shape[0]),
    ]:
        valid_mask = char_indices < max_idx
        if not valid_mask.all():
            skip = [valid_chars[i] for i in range(nc) if not valid_mask[i]]
            print(f'WARNING: {repr_name} skipping chars not in weight matrix: {skip}')

        vecs = weight_matrix[char_indices[valid_mask]]
        vecs_norm = F.normalize(vecs, dim=1)
        sim_matrix = (vecs_norm @ vecs_norm.T).cpu().numpy()

        vc = [valid_chars[i] for i in range(nc) if valid_mask[i]]
        nc_v = len(vc)

        # --- Heatmap 1: pure cosine similarity ---
        fig, ax = plt.subplots(figsize=(18, 16))
        vmin, vmax = sim_matrix.min(), sim_matrix.max()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 else None
        cmap = 'RdBu_r' if vmin < 0 else 'YlOrRd'
        im = ax.imshow(sim_matrix, cmap=cmap, norm=norm, aspect='equal')
        ax.set_xticks(range(nc_v))
        ax.set_yticks(range(nc_v))
        ax.set_xticklabels(vc, fontsize=7)
        ax.set_yticklabels(vc, fontsize=7)
        ax.set_xlabel('Char j')
        ax.set_ylabel('Char i')
        ax.set_title(f'Cosine Similarity ({repr_name})')
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        path = out_dir / f'sim_heatmap_{repr_name}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved {path}')

        # --- Heatmap 2: with confusion overlay ---
        fig, ax = plt.subplots(figsize=(18, 16))
        im = ax.imshow(sim_matrix, cmap=cmap, norm=norm, aspect='equal')

        for i_idx, ch_i in enumerate(vc):
            for j_idx, ch_j in enumerate(vc):
                if (ch_i, ch_j) in confused_pairs:
                    rect = plt.Rectangle((j_idx - 0.5, i_idx - 0.5), 1, 1,
                                         linewidth=1.5, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

        ax.set_xticks(range(nc_v))
        ax.set_yticks(range(nc_v))
        ax.set_xticklabels(vc, fontsize=7)
        ax.set_yticklabels(vc, fontsize=7)
        ax.set_xlabel('Char j')
        ax.set_ylabel('Char i')
        ax.set_title(f'Cosine Similarity ({repr_name}) — red = confused pair')
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        path = out_dir / f'sim_heatmap_{repr_name}_confused.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved {path}')

        np.save(out_dir / f'sim_matrix_{repr_name}.npy', sim_matrix)

    print(f'\nAll heatmaps saved to {out_dir}/')


if __name__ == '__main__':
    main()

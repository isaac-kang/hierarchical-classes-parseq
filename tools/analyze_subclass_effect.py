#!/usr/bin/env python3
"""Analyze whether subclass split (CCD) actually helps reduce misrecognition.

Three analyses:
  1. Hard sample logit analysis: For samples where base model predicts B->D,
     check if CCD model's f(x)·W_B1 > f(x)·W_D (i.e., B_1 absorbs hard samples)
  2. Normal sample leakage: Check if easy B samples get pulled into B_1
  3. Merge-after accuracy: Compare per-char accuracy before/after merging subclasses

Usage:
  python tools/analyze_subclass_effect.py \
    --baseline_ckpt <baseline_checkpoint> \
    --ccd_ckpt <ccd_checkpoint> \
    --unicode_mapping confusion_pl_output/unicode_mapping.json \
    --data_root ~/data/STR/parseq/english/lmdb
"""

import argparse
import json
import string
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strhub.data.module import SceneTextDataModule
from strhub.data.utils import PLCharsetAdapter
from strhub.models.utils import load_from_checkpoint


def get_head_weight_and_tokenizer(model):
    """Return head weight matrix and tokenizer stoi mapping."""
    head_wt = model.model.head.weight.detach()  # (num_classes, embed_dim)
    return head_wt, model.tokenizer._stoi


def extract_features_and_logits(model, images):
    """Run forward pass and return per-position features (before head) and logits.

    Returns:
        features: (B, L, embed_dim) - decoder output before classification head
        logits: (B, L, num_classes) - classification logits
    """
    tokenizer = model.tokenizer
    inner = model.model  # PARSeq inner model

    max_length = inner.max_label_length
    bs = images.shape[0]
    num_steps = max_length + 1
    memory = inner.encode(images)
    pos_queries = inner.pos_queries[:, :num_steps].expand(bs, -1, -1)
    tgt_mask = query_mask = torch.triu(
        torch.ones((num_steps, num_steps), dtype=torch.bool, device=images.device), 1
    )

    if inner.decode_ar:
        tgt_in = torch.full((bs, num_steps), tokenizer.pad_id, dtype=torch.long, device=images.device)
        tgt_in[:, 0] = tokenizer.bos_id
        all_features = []
        all_logits = []
        for i in range(num_steps):
            j = i + 1
            tgt_out = inner.decode(
                tgt_in[:, :j], memory,
                tgt_mask[:j, :j],
                tgt_query=pos_queries[:, i:j],
                tgt_query_mask=query_mask[i:j, :j],
            )
            p_i = inner.head(tgt_out)
            all_features.append(tgt_out)
            all_logits.append(p_i)
            if j < num_steps:
                tgt_in[:, j] = p_i.squeeze().argmax(-1)
                if (tgt_in == tokenizer.eos_id).any(dim=-1).all():
                    break
        features = torch.cat(all_features, dim=1)
        logits = torch.cat(all_logits, dim=1)
    else:
        tgt_in = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=images.device)
        features = inner.decode(tgt_in, memory, tgt_query=pos_queries)
        logits = inner.head(features)

    return features, logits


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_ckpt', required=True, help='Baseline model checkpoint')
    parser.add_argument('--ccd_ckpt', required=True, help='CCD model checkpoint')
    parser.add_argument('--unicode_mapping', default='confusion_pl_output/unicode_mapping.json')
    parser.add_argument('--data_root', default='~/data/STR/parseq/english/lmdb')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=None, help='Output JSON path')
    args = parser.parse_args()
    args.data_root = str(Path(args.data_root).expanduser().resolve())

    # --- Load unicode mapping ---
    mapping_path = str(Path(args.unicode_mapping).expanduser().resolve())
    with open(mapping_path, 'r', encoding='utf-8') as f:
        unicode_mapping = json.load(f)
    ext_to_base = {v['unicode']: v['base_char'] for v in unicode_mapping.values()}

    # Build reverse map: base_char -> [(ext_name, confused_with, unicode)]
    base_to_ext = defaultdict(list)
    for ext_name, entry in unicode_mapping.items():
        base_to_ext[entry['base_char']].append({
            'ext_name': ext_name,
            'confused_with': entry['confused_with'],
            'unicode': entry['unicode'],
        })

    # --- Charset ---
    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase

    # --- Load models ---
    print('Loading baseline model...')
    baseline = load_from_checkpoint(args.baseline_ckpt, charset_test=charset_test).eval().to(args.device)

    print('Loading CCD model...')
    ccd = load_from_checkpoint(args.ccd_ckpt, charset_test=charset_test).eval().to(args.device)
    ccd.charset_adapter = PLCharsetAdapter(charset_test, ext_to_base)

    baseline_stoi = baseline.tokenizer._stoi
    ccd_stoi = ccd.tokenizer._stoi

    # --- Data ---
    hp = ccd.hparams
    datamodule = SceneTextDataModule(
        args.data_root, '_unused_', hp.img_size, hp.max_label_length,
        hp.charset_train, charset_test, args.batch_size, args.num_workers, False,
    )

    test_set = sorted(set(SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK))

    # ========================================================================
    # Analysis structures
    # ========================================================================
    # Per confusion pair (base_char, confused_with):
    #   - hard samples: baseline predicts confused_with instead of base_char
    #   - easy samples: baseline correctly predicts base_char
    # For each, record CCD model's logits for base, ext, confused classes

    # Analysis 1 & 2: per-char logit comparison
    pair_stats = defaultdict(lambda: {
        'hard': [],   # samples where baseline: gt=base, pred=confused
        'easy': [],   # samples where baseline: gt=base, pred=base (correct)
    })

    # Analysis 3: merge-after accuracy
    baseline_char_correct = defaultdict(int)
    baseline_char_total = defaultdict(int)
    ccd_char_correct = defaultdict(int)  # after merging ext -> base
    ccd_char_total = defaultdict(int)

    print('\nRunning analysis...')
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        for imgs, labels in tqdm(iter(dataloader), desc=name):
            imgs = imgs.to(args.device)

            # --- Baseline: get predictions ---
            bl_logits = baseline.forward(imgs)
            bl_probs = bl_logits.softmax(-1)
            bl_preds, _ = baseline.tokenizer.decode(bl_probs)

            # --- CCD: get features and logits ---
            ccd_features, ccd_logits = extract_features_and_logits(ccd, imgs)
            ccd_probs = ccd_logits.softmax(-1)
            ccd_preds_raw, _ = ccd.tokenizer.decode(ccd_probs)

            ccd_head_wt = ccd.model.head.weight  # (num_classes, embed_dim)

            for sample_idx, (bl_pred, ccd_pred_raw, gt) in enumerate(zip(bl_preds, ccd_preds_raw, labels)):
                # Merge CCD prediction
                ccd_pred_merged = ccd.charset_adapter(ccd_pred_raw)

                # Per-character analysis
                max_len = max(len(gt), len(bl_pred), len(ccd_pred_raw))
                feat = ccd_features[sample_idx]  # (L, embed_dim)

                for pos in range(min(max_len, feat.shape[0])):
                    gt_ch = gt[pos] if pos < len(gt) else None
                    bl_ch = bl_pred[pos] if pos < len(bl_pred) else None
                    ccd_ch_merged = ccd_pred_merged[pos] if pos < len(ccd_pred_merged) else None

                    if gt_ch is None:
                        continue

                    # Analysis 3: per-char accuracy
                    baseline_char_total[gt_ch] += 1
                    ccd_char_total[gt_ch] += 1
                    if bl_ch == gt_ch:
                        baseline_char_correct[gt_ch] += 1
                    if ccd_ch_merged == gt_ch:
                        ccd_char_correct[gt_ch] += 1

                    # Analysis 1 & 2: logit comparison for confusion pairs
                    if gt_ch not in base_to_ext:
                        continue  # not a base char with confusion pairs

                    f_x = feat[pos]  # (embed_dim,)

                    for ext_info in base_to_ext[gt_ch]:
                        confused_ch = ext_info['confused_with']
                        ext_unicode = ext_info['unicode']

                        # Get CCD model logits for: base, ext_unicode, confused
                        base_idx = ccd_stoi.get(gt_ch)
                        ext_idx = ccd_stoi.get(ext_unicode)
                        confused_idx = ccd_stoi.get(confused_ch)

                        if base_idx is None or confused_idx is None:
                            continue

                        # Compute dot products (logits) manually: f(x) · W
                        logit_base = (f_x * ccd_head_wt[base_idx]).sum().item() if base_idx < ccd_head_wt.shape[0] else None
                        logit_ext = (f_x * ccd_head_wt[ext_idx]).sum().item() if ext_idx is not None and ext_idx < ccd_head_wt.shape[0] else None
                        logit_confused = (f_x * ccd_head_wt[confused_idx]).sum().item() if confused_idx < ccd_head_wt.shape[0] else None

                        if logit_base is None or logit_confused is None:
                            continue

                        record = {
                            'logit_base': logit_base,
                            'logit_ext': logit_ext,
                            'logit_confused': logit_confused,
                        }

                        pair_key = (gt_ch, confused_ch)

                        # Is this a hard sample? (baseline predicted confused_ch instead of gt_ch)
                        if bl_ch == confused_ch:
                            pair_stats[pair_key]['hard'].append(record)
                        elif bl_ch == gt_ch:
                            pair_stats[pair_key]['easy'].append(record)

    # ========================================================================
    # Report
    # ========================================================================

    print('\n' + '=' * 100)
    print('ANALYSIS 1: Hard sample absorption (baseline: gt=B, pred=D → does CCD model give B_1 > D?)')
    print('=' * 100)
    print(f'{"Base":>5} {"Conf":>5} {"#Hard":>6} │'
          f' {"mean(B)":>8} {"mean(B_1)":>9} {"mean(D)":>8} │'
          f' {"B1>D %":>7} {"B>D %":>6} {"(B∨B1)>D %":>11}')
    print('─' * 90)

    analysis1_results = []
    for (base_ch, confused_ch), stats in sorted(pair_stats.items()):
        hard = stats['hard']
        if not hard:
            continue
        n = len(hard)
        mean_base = sum(r['logit_base'] for r in hard) / n
        mean_ext = sum(r['logit_ext'] for r in hard if r['logit_ext'] is not None) / max(1, sum(1 for r in hard if r['logit_ext'] is not None))
        mean_conf = sum(r['logit_confused'] for r in hard) / n

        # B_1 > D rate
        ext_wins = sum(1 for r in hard if r['logit_ext'] is not None and r['logit_ext'] > r['logit_confused'])
        ext_valid = sum(1 for r in hard if r['logit_ext'] is not None)
        ext_win_rate = 100 * ext_wins / ext_valid if ext_valid > 0 else 0

        # B > D rate
        base_wins = sum(1 for r in hard if r['logit_base'] > r['logit_confused'])
        base_win_rate = 100 * base_wins / n

        # max(B, B_1) > D rate
        either_wins = sum(1 for r in hard if (
            r['logit_base'] > r['logit_confused'] or
            (r['logit_ext'] is not None and r['logit_ext'] > r['logit_confused'])
        ))
        either_win_rate = 100 * either_wins / n

        print(f'{base_ch:>5} {confused_ch:>5} {n:>6} │'
              f' {mean_base:>8.2f} {mean_ext:>9.2f} {mean_conf:>8.2f} │'
              f' {ext_win_rate:>7.1f} {base_win_rate:>6.1f} {either_win_rate:>11.1f}')

        analysis1_results.append({
            'base': base_ch, 'confused': confused_ch, 'n_hard': n,
            'mean_logit_base': round(mean_base, 4),
            'mean_logit_ext': round(mean_ext, 4),
            'mean_logit_confused': round(mean_conf, 4),
            'ext_wins_D_pct': round(ext_win_rate, 2),
            'base_wins_D_pct': round(base_win_rate, 2),
            'either_wins_D_pct': round(either_win_rate, 2),
        })

    print('\n' + '=' * 100)
    print('ANALYSIS 2: Normal sample leakage (baseline: gt=B, pred=B → does CCD B_1 steal from B?)')
    print('=' * 100)
    print(f'{"Base":>5} {"Conf":>5} {"#Easy":>6} │'
          f' {"mean(B)":>8} {"mean(B_1)":>9} {"mean(D)":>8} │'
          f' {"B1>B %":>7} (leakage rate)')
    print('─' * 90)

    analysis2_results = []
    for (base_ch, confused_ch), stats in sorted(pair_stats.items()):
        easy = stats['easy']
        if not easy:
            continue
        n = len(easy)
        mean_base = sum(r['logit_base'] for r in easy) / n
        mean_ext = sum(r['logit_ext'] for r in easy if r['logit_ext'] is not None) / max(1, sum(1 for r in easy if r['logit_ext'] is not None))
        mean_conf = sum(r['logit_confused'] for r in easy) / n

        # Leakage: B_1 > B rate (ext steals from base)
        leakage = sum(1 for r in easy if r['logit_ext'] is not None and r['logit_ext'] > r['logit_base'])
        ext_valid = sum(1 for r in easy if r['logit_ext'] is not None)
        leakage_rate = 100 * leakage / ext_valid if ext_valid > 0 else 0

        print(f'{base_ch:>5} {confused_ch:>5} {n:>6} │'
              f' {mean_base:>8.2f} {mean_ext:>9.2f} {mean_conf:>8.2f} │'
              f' {leakage_rate:>7.1f}')

        analysis2_results.append({
            'base': base_ch, 'confused': confused_ch, 'n_easy': n,
            'mean_logit_base': round(mean_base, 4),
            'mean_logit_ext': round(mean_ext, 4),
            'mean_logit_confused': round(mean_conf, 4),
            'leakage_pct': round(leakage_rate, 2),
        })

    print('\n' + '=' * 100)
    print('ANALYSIS 3: Per-character accuracy — Baseline vs CCD (after merge)')
    print('=' * 100)

    # Only show chars that have confusion pairs
    chars_with_pairs = set()
    for base_ch, _ in pair_stats.keys():
        chars_with_pairs.add(base_ch)
    # Also add confused_with chars
    for _, confused_ch in pair_stats.keys():
        chars_with_pairs.add(confused_ch)

    print(f'{"Char":>5} {"#Total":>7} │ {"BL Acc%":>8} {"CCD Acc%":>9} {"Δ":>7}')
    print('─' * 50)

    analysis3_results = []
    for ch in sorted(chars_with_pairs):
        total = baseline_char_total.get(ch, 0)
        if total == 0:
            continue
        bl_acc = 100 * baseline_char_correct.get(ch, 0) / total
        ccd_acc = 100 * ccd_char_correct.get(ch, 0) / total
        delta = ccd_acc - bl_acc
        marker = '✓' if delta > 0 else ('✗' if delta < 0 else ' ')

        print(f'{ch:>5} {total:>7} │ {bl_acc:>8.2f} {ccd_acc:>9.2f} {delta:>+7.2f} {marker}')

        analysis3_results.append({
            'char': ch, 'total': total,
            'baseline_acc': round(bl_acc, 2),
            'ccd_acc': round(ccd_acc, 2),
            'delta': round(delta, 2),
        })

    # Overall
    total_bl = sum(baseline_char_correct.values())
    total_ccd = sum(ccd_char_correct.values())
    total_all = sum(baseline_char_total.values())
    print('─' * 50)
    print(f'{"ALL":>5} {total_all:>7} │ {100*total_bl/total_all:>8.2f} {100*total_ccd/total_all:>9.2f} {100*(total_ccd-total_bl)/total_all:>+7.2f}')

    # Save JSON
    out_path = args.output or str(Path(mapping_path).parent / 'subclass_analysis.json')
    results = {
        'analysis1_hard_absorption': analysis1_results,
        'analysis2_leakage': analysis2_results,
        'analysis3_per_char_accuracy': analysis3_results,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()

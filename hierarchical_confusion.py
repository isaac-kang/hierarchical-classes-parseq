#!/usr/bin/env python3
"""Hierarchical Classification - Step 1: Confusion Matrix & Pseudo-Label Generation

1. Generate char-level confusion matrix from pretrained model on 6 benchmark test sets.
   For each character, find top-3 most confused characters → create sub-classes (e.g., B, B1, B2, B3).
   Save confusion matrix and mapping.

2. For SVT train set (val/SVT), generate pseudo-labels (PL) based on the confusion mapping.
   If GT='B' and pred matches one of B's top-3 confused chars → PL = that confused class name.
   Save pred, gt, PL comparison as text file.
"""

import argparse
import json
import os
import string
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from strhub.data.dataset import LmdbDataset
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


def align_sequences(s1, s2):
    """Align two sequences using edit distance DP. Returns aligned_s1, aligned_s2 with '-' for gaps."""
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    aligned_s1, aligned_s2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (s1[i - 1] == s2[j - 1] or dp[i][j] == dp[i - 1][j - 1] + 1):
            aligned_s1.append(s1[i - 1])
            aligned_s2.append(s2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned_s1.append(s1[i - 1])
            aligned_s2.append('-')
            i -= 1
        else:
            aligned_s1.append('-')
            aligned_s2.append(s2[j - 1])
            j -= 1
    return list(reversed(aligned_s1)), list(reversed(aligned_s2))


@torch.inference_mode()
def step1_confusion_matrix(model, datamodule, output_dir, charset):
    """Step 1: Build char-level confusion matrix from 6 benchmark test sets."""
    print("\n" + "=" * 60)
    print("STEP 1: Building Character-level Confusion Matrix")
    print("=" * 60)

    # confusion_counts[gt_char][pred_char] = count
    confusion_counts = defaultdict(lambda: defaultdict(int))
    total_chars = 0
    correct_chars = 0

    test_set = SceneTextDataModule.TEST_BENCHMARK_SUB
    max_width = max(map(len, test_set))

    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            imgs = imgs.to(model.device)
            logits = model(imgs)
            probs = logits.softmax(-1)
            preds, _ = model.tokenizer.decode(probs)

            for pred, gt in zip(preds, labels):
                pred = model.charset_adapter(pred)
                # Align pred and gt at character level
                aligned_pred, aligned_gt = align_sequences(pred, gt)

                for ap, ag in zip(aligned_pred, aligned_gt):
                    if ag == '-':  # insertion in pred (no GT char)
                        continue
                    if ap == '-':  # deletion (GT char not predicted)
                        confusion_counts[ag]['<DEL>'] += 1
                        total_chars += 1
                        continue
                    confusion_counts[ag][ap] += 1
                    total_chars += 1
                    if ap == ag:
                        correct_chars += 1

    print(f"\nTotal aligned chars: {total_chars}, Correct: {correct_chars} ({100*correct_chars/total_chars:.2f}%)")

    # Build confusion matrix as numpy array for visualization
    chars = sorted(confusion_counts.keys())
    all_pred_chars = set()
    for gt_char in confusion_counts:
        all_pred_chars.update(confusion_counts[gt_char].keys())
    pred_chars = sorted(all_pred_chars)

    # Save raw confusion counts as JSON
    confusion_json = {}
    for gt_char in sorted(confusion_counts.keys()):
        confusion_json[gt_char] = dict(sorted(
            confusion_counts[gt_char].items(), key=lambda x: x[1], reverse=True
        ))

    confusion_path = os.path.join(output_dir, 'confusion_counts.json')
    with open(confusion_path, 'w') as f:
        json.dump(confusion_json, f, indent=2, ensure_ascii=False)
    print(f"Confusion counts saved to: {confusion_path}")

    # Build top-3 confusion mapping for each character
    # For each GT char, find top-3 most confused (misrecognized) pred chars (excluding correct)
    confusion_mapping = {}
    for gt_char in sorted(confusion_counts.keys()):
        misrecognitions = {}
        for pred_char, count in confusion_counts[gt_char].items():
            if pred_char != gt_char:  # exclude correct predictions
                misrecognitions[pred_char] = count

        # Sort by count descending
        sorted_misrec = sorted(misrecognitions.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_misrec[:3]

        correct_count = confusion_counts[gt_char].get(gt_char, 0)
        total_for_char = sum(confusion_counts[gt_char].values())
        error_count = total_for_char - correct_count

        confusion_mapping[gt_char] = {
            'total': total_for_char,
            'correct': correct_count,
            'error': error_count,
            'accuracy': correct_count / total_for_char if total_for_char > 0 else 0,
            'top3_confused': [
                {'pred_char': pc, 'count': cnt, 'sub_class': f'{gt_char}{i+1}'}
                for i, (pc, cnt) in enumerate(top3)
            ]
        }

    mapping_path = os.path.join(output_dir, 'confusion_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(confusion_mapping, f, indent=2, ensure_ascii=False)
    print(f"Confusion mapping saved to: {mapping_path}")

    # Print summary: characters with highest error rates
    print("\n--- Top-20 Characters by Error Count ---")
    sorted_by_error = sorted(
        confusion_mapping.items(),
        key=lambda x: x[1]['error'],
        reverse=True
    )
    print(f"{'GT':>4} | {'Total':>6} | {'Correct':>7} | {'Error':>5} | {'Acc%':>6} | Top-3 Confused (pred_char: count)")
    print("-" * 90)
    for gt_char, info in sorted_by_error[:20]:
        top3_str = ', '.join(
            f"'{t['pred_char']}': {t['count']} → {t['sub_class']}"
            for t in info['top3_confused']
        )
        print(f"'{gt_char}':  {info['total']:>5} | {info['correct']:>7} | {info['error']:>5} | {info['accuracy']*100:>5.1f}% | {top3_str}")

    return confusion_mapping


@torch.inference_mode()
def step2_pseudo_labels(model, datamodule, confusion_mapping, output_dir, svt_path):
    """Step 2: Generate pseudo-labels for SVT train set based on confusion mapping.

    Rule: if GT='B' and pred is one of B's top-3 confused chars,
          then PL = the corresponding sub-class (e.g., 'B1').
          Otherwise PL = GT.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Generating Pseudo-Labels for SVT Train Set")
    print("=" * 60)

    # Build reverse lookup: for each GT char, map {confused_pred_char -> sub_class_name}
    # e.g., for GT='B': {'D': 'B1', '8': 'B2', 'R': 'B3'}
    gt_to_confused = {}
    for gt_char, info in confusion_mapping.items():
        reverse_map = {}
        for entry in info['top3_confused']:
            if entry['count'] > 0:  # only include if there were actual confusions
                reverse_map[entry['pred_char']] = entry['sub_class']
        if reverse_map:
            gt_to_confused[gt_char] = reverse_map

    # Load SVT train set (using val/SVT)
    hp = model.hparams
    transform = SceneTextDataModule.get_transform(hp.img_size)
    charset_test = string.digits + string.ascii_lowercase
    svt_dataset = LmdbDataset(
        svt_path,
        charset_test,
        hp.max_label_length,
        remove_whitespace=True,
        normalize_unicode=True,
        transform=transform,
    )
    svt_loader = torch.utils.data.DataLoader(
        svt_dataset, batch_size=512, num_workers=4, pin_memory=True
    )

    print(f"SVT train set: {len(svt_dataset)} samples from {svt_path}")

    results = []
    total_samples = 0
    pl_changed_samples = 0

    for imgs, labels in tqdm(svt_loader, desc='SVT PL'):
        imgs = imgs.to(model.device)
        logits = model(imgs)
        probs = logits.softmax(-1)
        preds, probs_list = model.tokenizer.decode(probs)

        for pred, prob, gt in zip(preds, probs_list, labels):
            pred = model.charset_adapter(pred)
            confidence = prob.log().mean().exp().item() if len(prob) > 0 else 0.0

            # Generate PL character by character
            aligned_pred, aligned_gt = align_sequences(pred, gt)

            pl_chars = []
            char_details = []
            sample_has_change = False

            for ap, ag in zip(aligned_pred, aligned_gt):
                if ag == '-':
                    # Insertion in pred, no GT char → skip for PL
                    continue
                if ap == '-':
                    # Deletion: GT char not predicted → keep GT
                    pl_chars.append(ag)
                    char_details.append(f'{ag}→<DEL>→{ag}')
                    continue

                # Check if this GT char has a confusion mapping and pred matches
                if ag in gt_to_confused and ap in gt_to_confused[ag]:
                    sub_class = gt_to_confused[ag][ap]
                    pl_chars.append(sub_class)
                    char_details.append(f'{ag}→{ap}→{sub_class}')
                    sample_has_change = True
                else:
                    pl_chars.append(ag)
                    if ap != ag:
                        char_details.append(f'{ag}→{ap}→{ag}(no mapping)')
                    # else: correct prediction, PL = GT

            pl_label = ''.join(pl_chars)
            total_samples += 1
            if sample_has_change:
                pl_changed_samples += 1

            results.append({
                'gt': gt,
                'pred': pred,
                'pl': pl_label,
                'confidence': confidence,
                'changed': sample_has_change,
                'details': ' | '.join(char_details) if char_details else '',
            })

    # Save results
    pl_file = os.path.join(output_dir, 'svt_train_pseudo_labels.txt')
    with open(pl_file, 'w') as f:
        f.write(f"{'GT':<20} {'PRED':<20} {'PL':<30} {'CONF':>8} {'CHANGED':>8} DETAILS\n")
        f.write("-" * 120 + "\n")
        for r in results:
            f.write(
                f"{r['gt']:<20} {r['pred']:<20} {r['pl']:<30} {r['confidence']:>8.4f} "
                f"{'*' if r['changed'] else '':>8} {r['details']}\n"
            )

    # Also save only changed samples for easier review
    pl_changed_file = os.path.join(output_dir, 'svt_train_pseudo_labels_changed.txt')
    with open(pl_changed_file, 'w') as f:
        f.write(f"{'GT':<20} {'PRED':<20} {'PL':<30} {'CONF':>8} DETAILS\n")
        f.write("-" * 120 + "\n")
        for r in results:
            if r['changed']:
                f.write(
                    f"{r['gt']:<20} {r['pred']:<20} {r['pl']:<30} {r['confidence']:>8.4f} "
                    f"{r['details']}\n"
                )

    print(f"\nTotal samples: {total_samples}")
    print(f"PL changed samples: {pl_changed_samples} ({100*pl_changed_samples/total_samples:.2f}%)")
    print(f"PL results saved to: {pl_file}")
    print(f"PL changed only saved to: {pl_changed_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Hierarchical Confusion: confusion matrix + pseudo-label generation'
    )
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='/data/isaackang/data/STR/parseq/english/lmdb')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='hierarchical_confusion_output')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top confused chars per class')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    # Use lowercase + digits charset for evaluation (standard STR benchmark)
    charset_test = string.digits + string.ascii_lowercase
    kwargs.update({'charset_test': charset_test})

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.checkpoint}")
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams

    # Setup data module
    datamodule = SceneTextDataModule(
        args.data_root, '_unused_', hp.img_size, hp.max_label_length,
        hp.charset_train, hp.charset_test,
        args.batch_size, args.num_workers, False,
    )

    # Step 1: Confusion matrix from 6 benchmark test sets
    confusion_mapping = step1_confusion_matrix(
        model, datamodule, args.output_dir, charset_test
    )

    # Step 2: Pseudo-labels for SVT train set
    svt_train_path = os.path.join(args.data_root, 'val', 'SVT')
    if not os.path.exists(svt_train_path):
        print(f"WARNING: SVT train path not found: {svt_train_path}")
        print("Trying test/SVT instead...")
        svt_train_path = os.path.join(args.data_root, 'test', 'SVT')

    step2_pseudo_labels(model, datamodule, confusion_mapping, args.output_dir, svt_train_path)

    print("\n" + "=" * 60)
    print("DONE. Output files:")
    print(f"  {args.output_dir}/confusion_counts.json       - Raw char confusion counts")
    print(f"  {args.output_dir}/confusion_mapping.json      - Top-3 confusion mapping per char")
    print(f"  {args.output_dir}/svt_train_pseudo_labels.txt - All SVT PL results")
    print(f"  {args.output_dir}/svt_train_pseudo_labels_changed.txt - Changed PL only")
    print("=" * 60)


if __name__ == '__main__':
    main()

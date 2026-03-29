#!/usr/bin/env python3
"""Test script for PARSeq models trained with Pseudo-Labeling (PL) extended charset.

Identical to test.py, but replaces model.charset_adapter with PLCharsetAdapter so that
extended Unicode variant characters predicted by the PL model are mapped back to their
base characters before comparison with ground-truth labels.
"""

import argparse
import json
import string
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import torch

from strhub.data.module import SceneTextDataModule
from strhub.data.utils import PLCharsetAdapter
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    err_sample_count: int = 0     # samples with at least one wrong char (word-level error)
    err_sample_ratio: float = 0.0  # % of samples with error
    err_char_count: int = 0       # total wrong chars (char-level)
    err_char_ratio: float = 0.0   # % of chars that are wrong
    total_chars: int = 0          # total chars (for combined calculation)
    ext_sample_count: int = 0     # samples containing at least one ext char
    ext_sample_ratio: float = 0.0  # % of samples containing at least one ext char
    ext_count: int = 0           # total extended chars predicted
    ext_ratio: float = 0.0       # % of predicted chars that are extended
    ext_correct: float = 0.0     # % of ext chars whose base char matches gt at that position


def print_results_table(results: list[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | #ErrSmp | ErrSmp% | #ErrChr | ErrChr% | #ExtSmp | ExtSmp% | #Ext |  Ext% | ExtAcc% |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|--------:|--------:|--------:|--------:|--------:|-----:|------:|--------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0)
    total_ext_count = 0
    total_ext_correct = 0
    total_pred_chars = 0
    total_ext_samples = 0
    total_err_samples = 0
    total_err_chars = 0
    total_all_chars = 0
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        total_ext_count += res.ext_count
        total_ext_samples += res.ext_sample_count
        total_err_samples += res.err_sample_count
        total_err_chars += res.err_char_count
        total_all_chars += res.total_chars
        # Recover absolute pred chars count for combined ext_ratio
        n_pred_chars = round(res.ext_count / (res.ext_ratio / 100)) if res.ext_ratio > 0 else res.total_chars
        total_pred_chars += n_pred_chars
        total_ext_correct += round(res.ext_correct / 100 * res.ext_count) if res.ext_count > 0 else 0
        print(
            f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} '
            f'| {res.err_sample_count:>7} | {res.err_sample_ratio:>7.2f} '
            f'| {res.err_char_count:>7} | {res.err_char_ratio:>7.2f} '
            f'| {res.ext_sample_count:>7} | {res.ext_sample_ratio:>7.1f} '
            f'| {res.ext_count:>4} | {res.ext_ratio:>5.2f} | {res.ext_correct:>7.1f} |',
            file=file,
        )
    c.accuracy /= c.num_samples
    c.err_sample_count = total_err_samples
    c.err_sample_ratio = 100 * total_err_samples / c.num_samples if c.num_samples > 0 else 0
    c.err_char_count = total_err_chars
    c.total_chars = total_all_chars
    c.err_char_ratio = 100 * total_err_chars / total_all_chars if total_all_chars > 0 else 0
    c.ext_sample_count = total_ext_samples
    c.ext_count = total_ext_count
    c.ext_ratio = 100 * total_ext_count / total_pred_chars if total_pred_chars > 0 else 0
    c.ext_correct = 100 * total_ext_correct / total_ext_count if total_ext_count > 0 else 0
    c.ext_sample_ratio = 100 * total_ext_samples / c.num_samples if c.num_samples > 0 else 0
    print('|-{:-<{w}}-|-----------|----------|---------|---------|---------|---------|---------|---------|------|-------|---------|'.format('----', w=w), file=file)
    print(
        f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} '
        f'| {c.err_sample_count:>7} | {c.err_sample_ratio:>7.2f} '
        f'| {c.err_char_count:>7} | {c.err_char_ratio:>7.2f} '
        f'| {c.ext_sample_count:>7} | {c.ext_sample_ratio:>7.1f} '
        f'| {c.ext_count:>4} | {c.ext_ratio:>5.2f} | {c.ext_correct:>7.1f} |',
        file=file,
    )


def align_to_gt(pred, gt):
    """Align pred to gt using edit distance. Returns a dict mapping gt index -> pred char (or None for insertions)."""
    n, m = len(gt), len(pred)
    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    # Backtrace: build gt_index -> pred_char mapping
    result = {}
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (gt[i - 1] == pred[j - 1] or dp[i][j] == dp[i - 1][j - 1] + 1):
            result[i - 1] = pred[j - 1]  # match or substitution
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            result[i - 1] = None  # deletion in pred (gt char has no match)
            i -= 1
        else:
            j -= 1  # insertion in pred (extra pred char, skip)
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='~/data/STR/parseq/english/lmdb')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reference', default=None, help="Reference model checkpoint (e.g. 'pretrained=parseq') for comparison")
    parser.add_argument('--align', action='store_true', default=False, help='Use edit-distance alignment for ref vs gt comparison')
    parser.add_argument('--print_ext_table', action='store_true', default=False, help='Print per-prediction ext logit detail table')
    parser.add_argument(
        '--unicode_mapping',
        default='confusion_pl_output/unicode_mapping.json',
        help='Path to unicode_mapping.json produced by confusion_and_pl.py',
    )
    args, unknown = parser.parse_known_args()
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    args.unicode_mapping = str(Path(args.unicode_mapping).expanduser().resolve())
    kwargs = parse_model_args(unknown)

    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    # Load unicode_mapping.json and build ext_to_base mapping
    with open(args.unicode_mapping, 'r', encoding='utf-8') as f:
        unicode_mapping = json.load(f)
    ext_to_base = {v['unicode']: v['base_char'] for v in unicode_mapping.values()}
    print(f'Loaded unicode mapping: {len(ext_to_base)} extended chars from {args.unicode_mapping}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    # Replace charset_adapter with PLCharsetAdapter to map extended chars back to base chars
    model.charset_adapter = PLCharsetAdapter(charset_test, ext_to_base)

    # Load reference model for comparison
    ref_model = None
    if args.reference:
        ref_model = load_from_checkpoint(args.reference, **kwargs).eval().to(args.device)
        print(f'Loaded reference model: {args.reference}')

    hp = model.hparams
    datamodule = SceneTextDataModule(
        args.data_root,
        '_unused_',
        hp.img_size,
        hp.max_label_length,
        hp.charset_train,
        charset_test,
        args.batch_size,
        args.num_workers,
        False,
        rotation=args.rotation,
    )

    test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
    if args.new:
        test_set += SceneTextDataModule.TEST_NEW
    test_set = sorted(set(test_set))

    ext_chars = set(ext_to_base.keys())

    results = {}
    ext_logit_rows = []  # collect ext logit info for summary table
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        total_pred_chars = 0
        total_ext_chars = 0
        ext_converted_correct = 0
        ext_converted_total = 0
        samples_with_ext = 0
        err_samples = 0
        err_chars = 0
        total_gt_chars = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct

            # Extended char analysis: run model again to get raw predictions
            logits = model.forward(imgs.to(model.device))
            probs = logits.softmax(-1)
            preds_raw, _ = model.tokenizer.decode(probs)
            itos = model.tokenizer._itos

            # Reference model predictions
            if ref_model is not None:
                ref_logits = ref_model.forward(imgs.to(ref_model.device))
                ref_probs = ref_logits.softmax(-1)
                ref_preds, _ = ref_model.tokenizer.decode(ref_probs)
            else:
                ref_preds = [None] * len(preds_raw)
            for sample_idx, (pred_raw, gt, ref_pred) in enumerate(zip(preds_raw, labels, ref_preds)):
                total_pred_chars += len(pred_raw)
                # Char-level error: compare converted prediction with GT
                pred_adapted = model.charset_adapter(pred_raw)
                max_len = max(len(pred_adapted), len(gt))
                total_gt_chars += max_len
                sample_has_err = (pred_adapted != gt)
                if sample_has_err:
                    err_samples += 1
                for i in range(max_len):
                    p = pred_adapted[i] if i < len(pred_adapted) else ''
                    g = gt[i] if i < len(gt) else ''
                    if p != g:
                        err_chars += 1
                has_ext = False
                # Build ref alignment if needed
                ref_aligned = None
                if ref_pred is not None and args.align:
                    ref_aligned = align_to_gt(ref_pred, gt)
                for i, c in enumerate(pred_raw):
                    if c in ext_chars:
                        total_ext_chars += 1
                        has_ext = True
                        base_c = ext_to_base[c]
                        ext_converted_total += 1
                        if i < len(gt) and base_c == gt[i]:
                            ext_converted_correct += 1
                        # Collect top-3 logits at this position
                        # logits shape: (B, num_steps, C), position i in pred corresponds to position i in logits
                        pos_logits = logits[sample_idx, i]
                        top3_vals, top3_ids = pos_logits.topk(3)
                        top3_chars = [itos[idx] for idx in top3_ids.tolist()]
                        gt_char = gt[i] if i < len(gt) else '?'
                        if ref_pred is None:
                            ref_char = '?'
                        elif ref_aligned is not None:
                            ref_char = ref_aligned.get(i) or '?'
                        else:
                            ref_char = ref_pred[i] if i < len(ref_pred) else '?'
                        ext_logit_rows.append({
                            'dataset': name,
                            'gt': gt,
                            'pred_raw': pred_raw,
                            'pos': i,
                            'gt_char': gt_char,
                            'pred_char': c,
                            'base_char': base_c,
                            'ref_char': ref_char,
                            'top1': top3_chars[0], 'top1_v': top3_vals[0].item(),
                            'top2': top3_chars[1], 'top2_v': top3_vals[1].item(),
                            'top3': top3_chars[2], 'top3_v': top3_vals[2].item(),
                            'case_diff': ref_char != '?' and gt_char.lower() == ref_char.lower() and gt_char != ref_char,
                            'alpha_diff': ref_char != '?' and gt_char.lower() != ref_char.lower(),
                        })
                if has_ext:
                    samples_with_ext += 1

        accuracy = 100 * correct / total
        err_sample_ratio = 100 * err_samples / total if total > 0 else 0
        err_char_ratio = 100 * err_chars / total_gt_chars if total_gt_chars > 0 else 0
        ext_ratio = 100 * total_ext_chars / total_pred_chars if total_pred_chars > 0 else 0
        ext_correct_ratio = 100 * ext_converted_correct / ext_converted_total if ext_converted_total > 0 else 0
        ext_sample_ratio = 100 * samples_with_ext / total if total > 0 else 0
        results[name] = Result(name, total, accuracy,
                               err_samples, err_sample_ratio,
                               err_chars, err_char_ratio, total_gt_chars,
                               samples_with_ext, ext_sample_ratio,
                               total_ext_chars, ext_ratio, ext_correct_ratio)

    result_groups = {
        'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB,
        'Benchmark': SceneTextDataModule.TEST_BENCHMARK,
    }
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)

    # Print ext logit summary table
    if ext_logit_rows:
        has_ref = ref_model is not None

        if args.print_ext_table:
            ref_cols = ' | Ref  | RefOK |' if has_ref else ''
            header = f'| {"Dataset":<16} | {"GT":<12} | {"Pred_raw":<12} | Pos | GT_ch | Pred | Base | {"Top1":>5}(logit) | {"Top2":>5}(logit) | {"Top3":>5}(logit) | Type  | Match |{ref_cols}'
            sep = '-' * len(header)
            print('\nExt prediction logit details:')
            print(header)
            print(sep)
            for r in ext_logit_rows:
                if r['case_diff']:
                    err_type = 'case'
                elif r['alpha_diff']:
                    err_type = 'alpha'
                else:
                    err_type = ''
                ref_str = ''
                if has_ref:
                    ref_str = (f' | {r["ref_char"]:>4} | {"O" if r["ref_char"] == r["gt_char"] else "X":>5} |')
                match = 'O' if r['gt_char'] == r['base_char'] else 'X'
                print(f'| {r["dataset"]:<16} | {r["gt"]:<12} | {r["pred_raw"]:<12} | {r["pos"]:>3} '
                      f'| {r["gt_char"]:>5} | {r["pred_char"]:>4} | {r["base_char"]:>4} '
                      f'| {r["top1"]:>5}({r["top1_v"]:>5.2f}) '
                      f'| {r["top2"]:>5}({r["top2_v"]:>5.2f}) '
                      f'| {r["top3"]:>5}({r["top3_v"]:>5.2f}) | {err_type:>5} | {match:>5} |{ref_str}')

        # Summary stats
        case_rows = [r for r in ext_logit_rows if r['case_diff']]
        alpha_rows = [r for r in ext_logit_rows if r['alpha_diff']]
        match_rows = [r for r in ext_logit_rows if not r['case_diff'] and not r['alpha_diff']]
        print(f'\nTotal ext predictions: {len(ext_logit_rows)}')
        print(f'  Match (gt == base):       {len(match_rows)}')
        print(f'  Case diff (gt.lower()==base.lower()): {len(case_rows)}')
        print(f'  Alpha diff (different letter):        {len(alpha_rows)}')

        if has_ref:
            def _case_alpha_by_ref(rows):
                """gt vs ref_char: ref가 어떻게 틀렸는지"""
                case = sum(1 for r in rows if r['gt_char'].lower() == r['ref_char'].lower())
                return case, len(rows) - case

            def _case_alpha_by_ext(rows):
                """gt vs base_char: ext가 어떻게 틀렸는지"""
                case = sum(1 for r in rows if r['gt_char'].lower() == r['base_char'].lower())
                return case, len(rows) - case

            n = len(ext_logit_rows)
            ext_correct = sum(1 for r in ext_logit_rows if r['base_char'] == r['gt_char'])
            ref_correct = sum(1 for r in ext_logit_rows if r['ref_char'] == r['gt_char'])
            rescue_rows = [r for r in ext_logit_rows if r['ref_char'] != r['gt_char'] and r['base_char'] == r['gt_char']]
            regress_rows = [r for r in ext_logit_rows if r['ref_char'] == r['gt_char'] and r['base_char'] != r['gt_char']]
            both_wrong_rows = [r for r in ext_logit_rows if r['ref_char'] != r['gt_char'] and r['base_char'] != r['gt_char']]
            both_correct_rows = [r for r in ext_logit_rows if r['ref_char'] == r['gt_char'] and r['base_char'] == r['gt_char']]

            print(f'\n  Ref vs Ext comparison ({n} ext predictions):')
            print(f'    Ext correct: {ext_correct} ({100*ext_correct/n:.1f}%)')
            print(f'    Ref correct: {ref_correct} ({100*ref_correct/n:.1f}%)')

            rc, ra = _case_alpha_by_ref(rescue_rows)
            print(f'    Rescue    (ref X -> ext O): {len(rescue_rows)}')
            print(f'      - ref was case wrong:  {rc}')
            print(f'      - ref was alpha wrong: {ra}')

            rc, ra = _case_alpha_by_ext(regress_rows)
            print(f'    Regress   (ref O -> ext X): {len(regress_rows)}')
            print(f'      - ext is case wrong:  {rc}')
            print(f'      - ext is alpha wrong: {ra}')

            rc, ra = _case_alpha_by_ext(both_wrong_rows)
            print(f'    Both wrong  (ref X, ext X): {len(both_wrong_rows)}')
            print(f'      - ext is case wrong:  {rc}')
            print(f'      - ext is alpha wrong: {ra}')

            print(f'    Both correct(ref O, ext O): {len(both_correct_rows)}')


if __name__ == '__main__':
    main()

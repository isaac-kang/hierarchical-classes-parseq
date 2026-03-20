#!/usr/bin/env python3
"""
Step 1: Generate character-level confusion matrix from pretrained model on 6 benchmark datasets.
        Extract top-3 confused classes per character and create extended class mapping.
Step 2: Perform Pseudo-Labeling (PL) on SVT train set using the confusion mapping.
"""

import argparse
import json
import string
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.data.utils import CharsetAdapter
from strhub.models.utils import load_from_checkpoint


# ==================== Unicode Variant Pool ====================
# Priority order per category:
#   Digits:    Enclosed Alphanumerics > Superscripts > Subscripts
#   Lowercase: Latin-1 Supplement > Latin Extended-A > Latin Extended Additional > Latin Extended-B > IPA
#   Uppercase: Latin-1 Supplement > Latin Extended-A > Latin Extended Additional > Latin Extended-B

UNICODE_VARIANTS = {
    # Digits — Superscripts > Subscripts > Enclosed Alphanumerics > Math
    '0': '⁰₀⓪𝟎𝟘𝟢𝟬𝟶',
    '1': '¹₁①𝟏𝟙𝟣𝟭𝟷',
    '2': '²₂②𝟐𝟚𝟤𝟮𝟸',
    '3': '³₃③𝟑𝟛𝟥𝟯𝟹',
    '4': '⁴₄④𝟒𝟜𝟦𝟰𝟺',
    '5': '⁵₅⑤𝟓𝟝𝟧𝟱𝟻',
    '6': '⁶₆⑥𝟔𝟞𝟨𝟲𝟼',
    '7': '⁷₇⑦𝟕𝟟𝟩𝟳𝟽',
    '8': '⁸₈⑧𝟖𝟠𝟪𝟴𝟾',
    '9': '⁹₉⑨𝟗𝟡𝟫𝟵𝟿',
    # Lowercase — Latin-1 Supp > Ext-A > Ext-Additional > Ext-B > IPA
    'a': 'àáâãäåāăąǎǟǡȁȃạảấầẩẫậắằẳẵặ',
    'b': 'ḃḅḇƀƃɓᵬᶀ',
    'c': 'çćĉċčḉｃ𝐜',
    'd': 'ďḋḍḏḑḓđ',
    'e': 'èéêëēĕėęěȅȇẹẻẽếềểễệ',
    'f': 'ḟƒᵮᶂ',
    'g': 'ĝğġģǧǵḡ',
    'h': 'ĥȟḣḥḧḩḫẖħ',
    'i': 'ìíîïĩīĭįǐȉȋịỉĩḭ',
    'j': 'ĵǰɉʝ',
    'k': 'ķǩḱḳḵƙ',
    'l': 'ĺļľḷḹḻḽŀł',
    'm': 'ḿṁṃ',
    'n': 'ñńņňǹṅṇṉṋ',
    'o': 'òóôõöōŏőơǒǫǭȍȏọỏốồổỗộớờởỡợ',
    'p': 'ṕṗƥᵽᶈ',
    'q': 'ɋʠ',
    'r': 'ŕŗřȑȓṙṛṝṟ',
    's': 'śŝşšșṡṣṥṧṩ',
    't': 'ţťțṫṭṯṱẗ',
    'u': 'ùúûüũūŭůűųưǔǖǘǚǜȕȗụủứừửữự',
    'v': 'ṽṿʋᶌｖ𝐯',
    'w': 'ŵẁẃẅẇẉẘ',
    'x': 'ẋẍᶍ',
    'y': 'ýÿŷȳẏẙỳỵỷỹ',
    'z': 'źżžẑẓẕ',
    # Uppercase — Latin-1 Supp > Ext-A > Ext-Additional > Ext-B
    'A': 'ÀÁÂÃÄÅĀĂĄǍǞǠȀȂẠẢẤẦẨẪẬẮẰẲẴẶ',
    'B': 'ḂḄḆƁƂɃ',
    'C': 'ÇĆĈĊČḈ',
    'D': 'ĎḊḌḎḐḒĐ',
    'E': 'ÈÉÊËĒĔĖĘĚȄȆẸẺẼẾỀỂỄỆ',
    'F': 'ḞƑＦ𝐅',
    'G': 'ĜĞĠĢǦǴḠ',
    'H': 'ĤȞḢḤḦḨḪ',
    'I': 'ÌÍÎÏĨĪĬĮİǏȈȊỊỈĨḬ',
    'J': 'ĴɈ',
    'K': 'ĶǨḰḲḴƘ',
    'L': 'ĹĻĽḶḸḺḼĿŁ',
    'M': 'ḾṀṂＭ𝐌𝑀',
    'N': 'ÑŃŅŇǸṄṆṈṊ',
    'O': 'ÒÓÔÕÖŌŎŐƠǑǪǬȌȎỌỎỐỒỔỖỘỚỜỞỠỢ',
    'P': 'ṔṖƤＰ',
    'Q': 'Ɋ',
    'R': 'ŔŖŘȐȒṘṚṜṞ',
    'S': 'ŚŜŞŠȘṠṢṤṦṨ',
    'T': 'ŢŤȚṪṬṮṰ',
    'U': 'ÙÚÛÜŨŪŬŮŰŲƯǓǕǗǙǛȔȖỤỦỨỪỬỮỰ',
    'V': 'ṼṾƲＶ𝐕',
    'W': 'ŴẀẂẄẆẈ',
    'X': 'ẊẌ',
    'Y': 'ÝŶŸȲẎỲỴỶỸ',
    'Z': 'ŹŻŽẐẒẔ',
}


def build_unicode_mapping(confusion_detail):
    """
    Assign a unique Unicode character to each extended class.
    Returns:
        ext_to_unicode: dict[str, str] - e.g., {'e_1': 'è', 'a_1': 'à'}
        unicode_to_ext: dict[str, str] - reverse mapping
    """
    ext_to_unicode = {}
    unicode_to_ext = {}
    used = set()

    for ch in sorted(confusion_detail.keys()):
        d = confusion_detail[ch]
        pool = UNICODE_VARIANTS.get(ch, '')
        idx = 0
        for i, t in enumerate(d['confused']):
            ext_name = d['extended_classes'][i + 1]
            # Find next unused Unicode variant
            while idx < len(pool) and pool[idx] in used:
                idx += 1
            if idx < len(pool):
                uni_ch = pool[idx]
                ext_to_unicode[ext_name] = uni_ch
                unicode_to_ext[uni_ch] = ext_name
                used.add(uni_ch)
                idx += 1
            else:
                # Fallback: should not happen if pool is large enough
                ext_to_unicode[ext_name] = f'[{ext_name}]'
                print(f'WARNING: No Unicode variant left for {ext_name} (base={ch})')

    return ext_to_unicode, unicode_to_ext


def needleman_wunsch_align(s1, s2, match_score=1, mismatch_score=-1, gap_score=-1):
    """
    Needleman-Wunsch alignment for two strings.
    Returns list of (char_from_s1_or_None, char_from_s2_or_None) pairs.
    None indicates a gap (insertion/deletion).
    """
    n, m = len(s1), len(s2)
    # Score matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_score
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_score
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match_score if s1[i - 1] == s2[j - 1] else mismatch_score
            dp[i][j] = max(
                dp[i - 1][j - 1] + score,  # match/substitution
                dp[i - 1][j] + gap_score,    # deletion (gap in s2)
                dp[i][j - 1] + gap_score,    # insertion (gap in s1)
            )
    # Traceback
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = match_score if s1[i - 1] == s2[j - 1] else mismatch_score
            if dp[i][j] == dp[i - 1][j - 1] + score:
                alignment.append((s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + gap_score:
            alignment.append((s1[i - 1], None))  # deletion
            i -= 1
        else:
            alignment.append((None, s2[j - 1]))  # insertion
            j -= 1
    alignment.reverse()
    return alignment


def build_confusion_matrix(model, datamodule, test_sets, charset, device):
    """
    Run inference on benchmark datasets and build a character-level confusion matrix.
    Uses Needleman-Wunsch alignment to handle length mismatches.
    confusion[gt_char][pred_char] = count (only for substitutions, not gaps)
    """
    confusion = defaultdict(lambda: defaultdict(int))
    charset_adapter = CharsetAdapter(charset)

    for name, dataloader in datamodule.test_dataloaders(test_sets).items():
        print(f'Processing {name}...')
        for imgs, labels in tqdm(iter(dataloader), desc=name):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = logits.softmax(-1)
            preds, _ = model.tokenizer.decode(probs)

            for pred, gt in zip(preds, labels):
                pred = charset_adapter(pred)
                aligned = needleman_wunsch_align(gt, pred)
                for gt_ch, pred_ch in aligned:
                    if gt_ch is not None and pred_ch is not None:
                        # Substitution or match — record in confusion matrix
                        confusion[gt_ch][pred_ch] += 1
                    # Gaps (insertion/deletion) are skipped for confusion matrix

    return confusion


def extract_confusions(confusion, charset, min_rate=0.001):
    """
    For each character in charset, find ALL confused characters
    whose confusion rate >= min_rate (default 0.1%).
    Returns:
        mapping: dict[str, list[str]] - e.g., {'B': ['8', 'D']}
        extended_classes: dict[str, list[str]] - e.g., {'B': ['B', 'B_1', 'B_2']}
        confusion_detail: dict for saving
    """
    mapping = {}
    extended_classes = {}
    confusion_detail = {}

    for ch in sorted(charset):
        if ch not in confusion:
            continue
        correct_count = confusion[ch].get(ch, 0)
        total = sum(confusion[ch].values())
        if total == 0:
            continue

        # Get all confused characters (excluding correct prediction)
        confused = {k: v for k, v in confusion[ch].items() if k != ch}
        if not confused:
            continue

        # Sort by count descending, include ALL that pass threshold
        sorted_confused = sorted(confused.items(), key=lambda x: (-x[1], x[0]))
        filtered = [(c, cnt) for c, cnt in sorted_confused if cnt / total >= min_rate]

        if not filtered:
            continue

        mapping[ch] = [c for c, _ in filtered]
        extended_classes[ch] = [ch] + [f'{ch}_{i+1}' for i in range(len(filtered))]

        confusion_detail[ch] = {
            'correct': correct_count,
            'total': total,
            'accuracy': correct_count / total,
            'confused': [{'char': c, 'count': cnt, 'rate': cnt / total} for c, cnt in filtered],
            'extended_classes': extended_classes[ch],
            'extended_class_mapping': {
                extended_classes[ch][0]: ch,
                **{extended_classes[ch][i+1]: filtered[i][0] for i in range(len(filtered))}
            }
        }

    return mapping, extended_classes, confusion_detail


def _build_confusion_map(confusion_detail, ext_to_unicode):
    """Build reverse lookup: for each gt_char, {confused_char -> unicode_char}"""
    confusion_map = {}
    for ch, detail in confusion_detail.items():
        ext_mapping = detail['extended_class_mapping']
        reverse = {}
        for ext_name, actual_char in ext_mapping.items():
            if ext_name != ch:
                uni_ch = ext_to_unicode.get(ext_name, ext_name)
                reverse[actual_char] = uni_ch
        confusion_map[ch] = reverse
    return confusion_map


def _apply_pl(gt, pred, confusion_map):
    """Apply PL rule to a single (gt, pred) pair. Returns PL string."""
    aligned = needleman_wunsch_align(gt, pred)
    pl_chars = []
    for gt_ch, pred_ch in aligned:
        if gt_ch is None:
            continue
        if pred_ch is None:
            pl_chars.append(gt_ch)
        elif gt_ch == pred_ch:
            pl_chars.append(gt_ch)
        elif gt_ch in confusion_map and pred_ch in confusion_map[gt_ch]:
            pl_chars.append(confusion_map[gt_ch][pred_ch])
        else:
            pl_chars.append(gt_ch)
    return ''.join(pl_chars)


def perform_pl(model, dataset_path, dataset_name, charset, confusion_detail, ext_to_unicode,
               device, output_dir, save_lmdb=True, lmdb_output_path=None, save_text=False):
    """
    Perform pseudo-labeling on a given dataset.
    PL rule: if gt=X and pred is one of X's confused chars (above threshold),
             then PL = the corresponding extended class (Unicode char).
             Otherwise PL = gt char.

    save_lmdb: saves a new LMDB at lmdb_output_path with PL labels and original images (default).
    save_text: optionally saves a text file with GT/PRED/PL columns.
    """
    import io
    import lmdb as lmdb_lib

    charset_adapter = CharsetAdapter(charset)
    confusion_map = _build_confusion_map(confusion_detail, ext_to_unicode)

    from strhub.data.dataset import LmdbDataset

    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    dataset = LmdbDataset(
        dataset_path,
        charset,
        model.hparams.max_label_length,
        remove_whitespace=True,
        normalize_unicode=True,
        transform=transform,
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=128, num_workers=4, pin_memory=True)

    # Open source LMDB for raw image access (when saving LMDB)
    src_env = None
    if save_lmdb:
        src_env = lmdb_lib.open(dataset_path, readonly=True, lock=False)

    results = []
    for imgs, labels in tqdm(dataloader, desc=f'{dataset_name} PL'):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = logits.softmax(-1)
        preds, _ = model.tokenizer.decode(probs)

        for pred, gt in zip(preds, labels):
            pred = charset_adapter(pred)
            pl_unicode = _apply_pl(gt, pred, confusion_map)
            results.append({
                'gt': gt,
                'pred': pred,
                'pl': pl_unicode,
            })

    # Optionally save text results
    if save_text:
        output_file = output_dir / f'{dataset_name}_pl_results.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f'{"GT":<30} {"PRED":<30} {"PL":<30}\n')
            f.write('=' * 90 + '\n')
            for r in results:
                f.write(f'{r["gt"]:<30} {r["pred"]:<30} {r["pl"]:<30}\n')
        print(f'\nPL text results saved to {output_file}')
    print(f'Total samples: {len(results)}')

    # Stats
    n_changed = sum(1 for r in results if r['gt'] != r['pred'])
    n_pl_applied = sum(1 for r in results if r['gt'] != r['pl'])
    print(f'Samples with wrong prediction: {n_changed}')
    print(f'Samples with PL applied (at least 1 char mapped to extended class): {n_pl_applied}')

    # Save LMDB with PL labels
    if save_lmdb and lmdb_output_path and src_env:
        lmdb_output_path = Path(lmdb_output_path)
        lmdb_output_path.mkdir(parents=True, exist_ok=True)

        # Use source data.mdb file size with generous margin
        src_mdb = Path(dataset_path) / 'data.mdb'
        map_size = max(src_mdb.stat().st_size * 10, 1024 * 1024 * 100)  # 10x or at least 100MB

        dst_env = lmdb_lib.open(str(lmdb_output_path), map_size=map_size)
        with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
            # Write num-samples
            dst_txn.put('num-samples'.encode(), str(len(results)).encode())

            # Copy images from source, write PL labels
            for out_idx, (result, src_lmdb_idx) in enumerate(
                zip(results, dataset.filtered_index_list), start=1
            ):
                # Copy raw image bytes from source
                img_key = f'image-{src_lmdb_idx:09d}'.encode()
                img_data = src_txn.get(img_key)

                dst_img_key = f'image-{out_idx:09d}'.encode()
                dst_label_key = f'label-{out_idx:09d}'.encode()
                dst_txn.put(dst_img_key, img_data)
                dst_txn.put(dst_label_key, result['pl'].encode())

        dst_env.close()
        print(f'PL LMDB saved to {lmdb_output_path}')

    if src_env:
        src_env.close()

    return results


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='pretrained=parseq',
                        help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='/data/isaackang/data/STR/parseq/english/lmdb')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='confusion_pl_output', help='Output directory')
    parser.add_argument('--min_rate', type=float, default=0.001,
                        help='Minimum confusion rate to assign extended class (default: 0.001 = 0.1%%)')
    parser.add_argument('--pl_datasets', nargs='+', default=['val/SVT'],
                        help='Datasets to perform PL on, relative to data_root (e.g., val/SVT val/IC15)')
    parser.add_argument('--save_text', action='store_true',
                        help='Save PL results as text file (optional)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Case-sensitive charset (62 chars: digits + lowercase + uppercase)
    charset = string.digits + string.ascii_lowercase + string.ascii_uppercase
    print(f'Charset: {charset}')

    # Load model
    print('Loading model...')
    model = load_from_checkpoint(args.checkpoint, charset_test=charset).eval().to(args.device)
    hp = model.hparams

    datamodule = SceneTextDataModule(
        args.data_root, '_unused_', hp.img_size, hp.max_label_length,
        hp.charset_train, charset,
        args.batch_size, args.num_workers, False,
    )

    # ==================== Step 1 ====================
    print('\n' + '=' * 60)
    print('Step 1: Building confusion matrix from 6 benchmark datasets')
    print('=' * 60)

    # Use the 6 benchmark (full) datasets
    test_sets = list(SceneTextDataModule.TEST_BENCHMARK)
    confusion = build_confusion_matrix(model, datamodule, test_sets, charset, args.device)

    # Save raw confusion matrix as numpy array
    chars = sorted(charset)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    n = len(chars)
    cm = np.zeros((n, n), dtype=np.int64)
    for gt_ch, preds in confusion.items():
        if gt_ch not in char_to_idx:
            continue
        for pred_ch, count in preds.items():
            if pred_ch not in char_to_idx:
                continue
            cm[char_to_idx[gt_ch], char_to_idx[pred_ch]] = count

    np.save(output_dir / 'confusion_matrix.npy', cm)

    # Save confusion matrix as readable CSV
    csv_path = output_dir / 'confusion_matrix.csv'
    with open(csv_path, 'w') as f:
        f.write('gt\\pred,' + ','.join(chars) + '\n')
        for i, ch in enumerate(chars):
            f.write(ch + ',' + ','.join(str(cm[i, j]) for j in range(n)) + '\n')
    print(f'Confusion matrix saved to {csv_path}')

    # Extract top-3 confusions and create extended class mapping (with threshold)
    print(f'\nConfusion rate threshold: {args.min_rate*100:.1f}%')
    mapping, extended_classes, confusion_detail = extract_confusions(confusion, charset, min_rate=args.min_rate)

    # Save the mapping
    mapping_path = output_dir / 'confusion_mapping.json'
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(confusion_detail, f, indent=2, ensure_ascii=False)
    print(f'Confusion mapping saved to {mapping_path}')

    # Build Unicode mapping
    ext_to_unicode, unicode_to_ext = build_unicode_mapping(confusion_detail)

    # Save Unicode mapping as JSON
    unicode_mapping_path = output_dir / 'unicode_mapping.json'
    mapping_data = {}
    for ch in sorted(confusion_detail.keys()):
        d = confusion_detail[ch]
        for i, t in enumerate(d['confused']):
            ext_name = d['extended_classes'][i + 1]
            uni_ch = ext_to_unicode.get(ext_name, '?')
            mapping_data[ext_name] = {
                'unicode': uni_ch,
                'codepoint': f'U+{ord(uni_ch):04X}' if len(uni_ch) == 1 else '?',
                'unicode_name': unicodedata.name(uni_ch, '?') if len(uni_ch) == 1 else '?',
                'base_char': ch,
                'confused_with': t['char'],
            }
    with open(unicode_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    print(f'Unicode mapping saved to {unicode_mapping_path}')

    # Build and save extended class summary text file
    summary_path = output_dir / 'extended_classes.txt'
    total_extended = 0
    summary_lines = []
    summary_lines.append(f'Extended Class Summary (threshold >= {args.min_rate*100:.1f}%)')
    summary_lines.append('=' * 100)
    summary_lines.append('')
    summary_lines.append(f'{"GT":<5} {"Acc%":<7} {"ExtClass":<10} {"Unicode":<4} {"Codepoint":<10} {"->Confused":<12} {"Count":<7} {"Rate%":<7}')
    summary_lines.append('-' * 100)
    for ch in sorted(confusion_detail.keys()):
        d = confusion_detail[ch]
        acc = d['accuracy'] * 100
        for i, t in enumerate(d['confused']):
            ext_name = d['extended_classes'][i + 1]
            uni_ch = ext_to_unicode.get(ext_name, '?')
            codepoint = f'U+{ord(uni_ch):04X}' if len(uni_ch) == 1 else '?'
            total_extended += 1
            prefix = f'{ch:<5} {acc:<7.1f}' if i == 0 else f'{"":5} {"":7}'
            summary_lines.append(f'{prefix} {ext_name:<10} {uni_ch:<4} {codepoint:<10} -> {t["char"]:<10} {t["count"]:<7} {t["rate"]*100:<7.2f}')
    summary_lines.append('-' * 100)
    summary_lines.append('')
    summary_lines.append(f'Original charset size: {len(charset)}')
    summary_lines.append(f'Characters with extended classes: {len(confusion_detail)}')
    summary_lines.append(f'Total extended classes added: {total_extended}')
    summary_lines.append(f'New total class count: {len(charset) + total_extended}')
    summary_lines.append('')

    # List all extended classes compactly
    summary_lines.append('All extended classes (name -> unicode):')
    all_ext = []
    for ch in sorted(confusion_detail.keys()):
        d = confusion_detail[ch]
        for i, t in enumerate(d['confused']):
            ext_name = d['extended_classes'][i + 1]
            uni_ch = ext_to_unicode.get(ext_name, '?')
            all_ext.append(f'{ext_name}={uni_ch}')
    summary_lines.append(', '.join(all_ext))

    summary_text = '\n'.join(summary_lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text + '\n')
    print(f'Extended class summary saved to {summary_path}')
    print(f'\n{summary_text}')

    # ==================== Step 2 ====================
    print('\n' + '=' * 60)
    print('Step 2: Pseudo-Labeling')
    print('=' * 60)

    for pl_ds in args.pl_datasets:
        ds_path = str(Path(args.data_root) / pl_ds)
        ds_name = pl_ds.replace('/', '_')
        lmdb_out = str(Path(args.data_root) / 'PL' / pl_ds)
        print(f'\nDataset: {pl_ds} ({ds_path})')
        print(f'  LMDB output: {lmdb_out}')
        perform_pl(model, ds_path, ds_name, charset, confusion_detail, ext_to_unicode,
                   args.device, output_dir, save_lmdb=True, lmdb_output_path=lmdb_out,
                   save_text=args.save_text)

    print(f'\nAll outputs saved to {output_dir}/')


if __name__ == '__main__':
    main()

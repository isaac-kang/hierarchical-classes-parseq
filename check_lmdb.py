#!/usr/bin/env python3
"""Check LMDB contents: save 5 random images and their labels."""
import argparse
import io
import random
from pathlib import Path

import lmdb
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lmdb_path', help='Path to LMDB directory')
    parser.add_argument('--output_dir', default='lmdb_check_output')
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        num_samples = int(txn.get(b'num-samples').decode())
        print(f'Total samples: {num_samples}')

        indices = random.sample(range(1, num_samples + 1), min(args.num_samples, num_samples))
        indices.sort()

        lines = []
        for idx in indices:
            label = txn.get(f'label-{idx:09d}'.encode()).decode()
            img_data = txn.get(f'image-{idx:09d}'.encode())
            img = Image.open(io.BytesIO(img_data))

            img_path = output_dir / f'{idx:09d}.png'
            img.save(img_path)

            lines.append(f'{idx:09d}\t{label}\t{img_path.name}')
            print(f'  [{idx}] label={label!r}  size={img.size}')

        label_path = output_dir / 'labels.txt'
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('index\tlabel\timage_file\n')
            for line in lines:
                f.write(line + '\n')
        print(f'\nSaved to {output_dir}/')

    env.close()


if __name__ == '__main__':
    main()

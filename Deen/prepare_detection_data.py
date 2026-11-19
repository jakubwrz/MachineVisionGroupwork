"""Convert MinneApple instance masks to YOLO labels; split by capture sequence."""
from pathlib import Path
import argparse
import csv
import json
import random
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_preview(im, mask, boxes, split, name, destination):
    """Count mask instances independently from the rectangles actually drawn."""
    annotated = int(np.count_nonzero(np.unique(mask)))
    header = 76
    preview = Image.new('RGB', (im.width, im.height + header), 'black')
    preview.paste(im.convert('RGB'), (0, header))
    draw = ImageDraw.Draw(preview)
    drawn = 0
    for instance, x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1 + header, x2-1, y2-1 + header), outline='red', width=2)
        drawn += 1
        draw.text((x1, y1 + header), str(instance), fill='yellow', stroke_width=1, stroke_fill='black')
    font = ImageFont.load_default(size=22)
    status = 'MATCH' if annotated == drawn else 'MISMATCH'
    draw.text((10, 8), f'Mask apples: {annotated} | Boxes drawn: {drawn} | {status}', font=font,
              fill='white' if annotated == drawn else 'orange')
    draw.text((10, 44), f'{split} | {name}', font=ImageFont.load_default(size=16), fill='white')
    preview.save(destination)


def refresh_previews(source, output):
    with (output / 'manifest.csv').open(encoding='utf-8', newline='') as f:
        rows = {row['image']: row for row in csv.DictReader(f)}
    paths = sorted((output / 'previews').glob('*.png'))
    for path in paths:
        row = rows[path.name]
        with Image.open(source / 'images' / path.name) as im, Image.open(source / 'masks' / path.name) as mask:
            if im.size != mask.size:
                raise ValueError(f'Image/mask size mismatch: {path.name}')
            boxes = []
            label_path = output / 'labels' / row['split'] / path.with_suffix('.txt').name
            for index, line in enumerate(label_path.read_text().splitlines(), 1):
                category, cx, cy, w, h = map(float, line.split())
                if category != 0 or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f'Invalid label in {label_path}: {line}')
                boxes.append((index, round((cx-w/2)*im.width), round((cy-h/2)*im.height),
                              round((cx+w/2)*im.width), round((cy+h/2)*im.height)))
            save_preview(im, np.asarray(mask), boxes, row['split'], path.name, path)
    print(f'Refreshed {len(paths)} previews using saved YOLO labels: {output / "previews"}')


def boxes_from_mask(mask):
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('Expected a single-channel integer instance mask')
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    ids, inverse = np.unique(mask[ys, xs], return_inverse=True)
    xmin = np.full(len(ids), mask.shape[1], dtype=int)
    ymin = np.full(len(ids), mask.shape[0], dtype=int)
    xmax = np.zeros(len(ids), dtype=int)
    ymax = np.zeros(len(ids), dtype=int)
    np.minimum.at(xmin, inverse, xs)
    np.minimum.at(ymin, inverse, ys)
    np.maximum.at(xmax, inverse, xs)
    np.maximum.at(ymax, inverse, ys)
    # Exclusive right/bottom edges preserve single-pixel instances.
    return list(zip(ids.tolist(), xmin.tolist(), ymin.tolist(),
                    (xmax + 1).tolist(), (ymax + 1).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent / 'data/detection/train')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'data/apple_yolo')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--preview-count', type=int, default=10)
    parser.add_argument('--refresh-previews', action='store_true', help='Redraw existing previews from saved labels without changing the dataset')
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1 or args.preview_count < 0:
        parser.error('val-fraction must be between 0 and 1; preview-count must be nonnegative')
    source, output = args.source.resolve(), args.output.resolve()
    if args.refresh_previews:
        refresh_previews(source, output)
        return
    if output.exists():
        parser.error(f'Output already exists; choose a new --output directory: {output}')
    images = {p.name: p for p in (source / 'images').glob('*.png')}
    masks = {p.name: p for p in (source / 'masks').glob('*.png')}
    if not images or images.keys() != masks.keys():
        raise ValueError(f'Image/mask pairing failed: images-only={sorted(images.keys()-masks.keys())[:10]}, masks-only={sorted(masks.keys()-images.keys())[:10]}')
    groups = {}
    for name in sorted(images):
        match = re.fullmatch(r'(.+)_image\d+\.png', name)
        if not match:
            raise ValueError(f'Cannot infer capture sequence from {name}')
        groups.setdefault(match[1], []).append(name)
    if len(groups) < 2:
        raise ValueError('At least two capture sequences are needed for a grouped split')
    keys = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    # Closest achievable image count using complete capture groups.
    reachable = {0: ()}
    for key in keys:
        for count, selected in list(reachable.items()):
            reachable.setdefault(count + len(groups[key]), selected + (key,))
    target = len(images) * args.val_fraction
    chosen = min((n for n in reachable if 0 < n < len(images)), key=lambda n: abs(n-target))
    val_groups = set(reachable[chosen])
    previews = set(rng.sample(sorted(images), min(args.preview_count, len(images))))
    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            (output / kind / split).mkdir(parents=True, exist_ok=True)
    (output / 'previews').mkdir()
    rows = []
    for group in sorted(groups):
        split = 'val' if group in val_groups else 'train'
        for name in groups[group]:
            with Image.open(images[name]) as im, Image.open(masks[name]) as mask:
                if im.size != mask.size:
                    raise ValueError(f'Image/mask size mismatch: {name}')
                width, height = im.size
                boxes = boxes_from_mask(np.asarray(mask))
                labels = []
                for _, x1, y1, x2, y2 in boxes:
                    values = ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)
                    assert all(0 <= v <= 1 for v in values) and values[2] > 0 and values[3] > 0
                    labels.append('0 ' + ' '.join(f'{v:.8f}' for v in values))
                (output / 'labels' / split / Path(name).with_suffix('.txt')).write_text('\n'.join(labels) + ('\n' if labels else ''), encoding='utf-8')
                shutil.copy2(images[name], output / 'images' / split / name)
                if name in previews:
                    save_preview(im, np.asarray(mask), boxes, split, name, output / 'previews' / name)
                rows.append(dict(image=name, sequence=group, split=split, width=width, height=height, count=len(boxes)))
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Relative path allows moving this entire folder to Google Colab.
    (output / 'data.yaml').write_text('train: images/train\nval: images/val\nnames:\n  0: apple\n', encoding='utf-8')
    summary = dict(source=str(source), seed=args.seed, requested_val_fraction=args.val_fraction,
                   split_rule='Group by filename prefix before _image; no sequence shared between splits',
                   label_assumption='0 is background; each distinct nonzero value is one apple instance',
                   groups={s: sorted(g for g in groups if (g in val_groups) == (s == 'val')) for s in ('train', 'val')},
                   splits={s: dict(images=sum(r['split']==s for r in rows), apples=sum(r['count'] for r in rows if r['split']==s)) for s in ('train', 'val')},
                   previews=len(previews), test_included=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Prepared dataset: {output}')


if __name__ == '__main__':
    main()

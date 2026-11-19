"""Draw saved counting predictions without loading a model."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from PIL import Image, ImageDraw, ImageFont
from counting_storage import dataset_hash
from lighting_preprocess import read_processed, tag


def render(run, data=None, top=10, image_name=None):
    run = Path(run).resolve()
    summary = json.loads((run / 'summary.json').read_text())
    dataset = Path(data or summary.get('dataset') or Path(__file__).resolve().parent / 'data/apple_yolo').resolve()
    split = summary.get('split', 'val')
    expected = summary.get('dataset_sha256')
    if expected and dataset_hash(dataset, split) != expected:
        raise ValueError('Dataset changed since prediction; refusing to draw misleading comparisons')
    rows = list(csv.DictReader((run / 'counts.csv').open()))
    rows.sort(key=lambda r: int(r['absolute_error']), reverse=True)
    if image_name:
        rows = [r for r in rows if r['image'] == image_name]
        if not rows: raise ValueError('Image not found in this run')
    elif top > 0:
        rows = rows[:top]
    predictions = {r['image']: r for r in json.loads((run / 'predictions.json').read_text())}
    args = SimpleNamespace(conf=summary['conf'])
    preprocessing = summary.get('preprocessing', {'mode': 'none'})
    output = run / 'previews'
    output.mkdir(exist_ok=True)
    for row in rows:
        image_path = dataset / 'images' / split / row['image']
        if Path(row['image']).name != row['image']: raise ValueError('Invalid image name')
        predicted = predictions[row['image']]['boxes_xyxy']
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        width, height = image.size
        truth = []
        for line in (dataset / 'labels' / split / image_path.with_suffix('.txt').name).read_text().splitlines():
            _, cx, cy, w, h = map(float, line.split())
            truth.append(((cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height))
        if len(truth) != int(row['annotated']) or len(predicted) != int(row['predicted']):
            raise ValueError('Saved counts do not match boxes or current labels')
        delta = len(predicted) - len(truth)
        # Separate panels keep overlapping truth/prediction boxes readable.
        canvas = Image.new('RGB', (width*2, height+90), 'black')
        canvas.paste(image, (0, 90))
        right_image = image
        if preprocessing['mode'] != 'none':
            import cv2
            right_image = Image.fromarray(cv2.cvtColor(read_processed(image_path, preprocessing), cv2.COLOR_BGR2RGB))
        canvas.paste(right_image, (width, 90))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=22)
        draw.text((10, 8), f'Annotated: {len(truth)} | Predicted: {len(predicted)} | Difference: {delta:+d}', font=font, fill='white')
        draw.text((10, 40), f'{image_path.name} | conf={args.conf} | Ground truth follows dataset annotations', fill='white', font=ImageFont.load_default(size=16))
        draw.text((10, 66), 'GREEN: annotations', fill='lime')
        draw.text((width+10, 66), f'RED: predictions | input: {tag(preprocessing)}', fill='red')
        for boxes, offset, color in ((truth, 0, 'lime'), (predicted, width, 'red')):
            for x1, y1, x2, y2 in boxes:
                draw.rectangle((x1+offset, y1+90, x2+offset, y2+90), outline=color, width=2)
        canvas.save(output / image_path.name)
    print(f'Saved {len(rows)} comparison images: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--top', type=int, default=10, help='Worst N images; 0 means all')
    parser.add_argument('--image', help='One image filename')
    args = parser.parse_args()
    if args.top < 0: parser.error('top must be nonnegative')
    render(args.run, args.data, args.top, args.image)

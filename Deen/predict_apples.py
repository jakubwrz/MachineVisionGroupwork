"""Compare predictions with validation labels; save images and counting errors."""
import argparse
import csv
import json
import math
import hashlib
from datetime import datetime
from pathlib import Path
from counting_storage import fingerprint, find_cached, finish
from lighting_preprocess import settings as preprocessing_settings, tag, read_processed
from run_layout import resolve_weights, model_group, safe_name

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/apple_yolo')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='0', help='CUDA index, or cpu')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--conf', type=float)
    parser.add_argument('--iou', type=float, help='NMS overlap threshold, not evaluation matching IoU')
    parser.add_argument('--imgsz', type=int)
    parser.add_argument('--preprocess', choices=['none', 'gamma', 'clahe'], default='none')
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--clahe-clip', type=float, default=2.0)
    parser.add_argument('--clahe-grid', type=int, default=8)
    parser.add_argument('--force', action='store_true', help='Recompute instead of reusing a matching complete run')
    parser.add_argument('--save-images', type=int, default=0, metavar='N', help='Render worst N images; default 0 saves no images')
    parser.add_argument('--baseline', action='store_true', help='Ignore tuned defaults and use 640/conf=0.25/NMS=0.7')
    args = parser.parse_args()
    if args.save_images < 0: parser.error('save-images must be nonnegative')
    try:
        preprocessing = preprocessing_settings(args.preprocess, args.gamma, args.clahe_clip, args.clahe_grid)
    except ValueError as error:
        parser.error(str(error))
    weights = args.weights
    if weights is None:
        record = ROOT / 'last_training.json'
        if not record.is_file():
            parser.error('Train first, or supply --weights path/to/best.pt')
        weights = Path(json.loads(record.read_text())['best_weights'])
    weights = resolve_weights(weights)
    if not weights.is_file():
        parser.error(f'Weights not found: {weights}')
    settings = {}
    settings_path = ROOT / 'inference_settings.json'
    if settings_path.is_file() and not args.baseline:
        saved = json.loads(settings_path.read_text())
        if saved.get('weights_sha256') == hashlib.sha256(weights.read_bytes()).hexdigest():
            settings = saved
    args.conf = args.conf if args.conf is not None else settings.get('conf', 0.25)
    args.iou = args.iou if args.iou is not None else settings.get('nms_iou', 0.7)
    args.imgsz = args.imgsz if args.imgsz is not None else settings.get('imgsz', 640)
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1 or args.imgsz <= 0:
        parser.error('Invalid confidence, IoU or image size')
    print(f'Inference settings: imgsz={args.imgsz}, conf={args.conf}, NMS IoU={args.iou}')
    from PIL import Image
    dataset = args.data.resolve()
    images = sorted((dataset / 'images' / args.split).glob('*.png'))
    if not images:
        raise FileNotFoundError(f'No images for {args.split}')
    experiment = model_group(weights)
    print('Checking model and evaluation-data content hashes...')
    identity = fingerprint(weights, dataset, args)
    cached = None if args.force else find_cached(ROOT / 'runs/counting' / experiment, identity)
    if cached:
        print('Reusing completed analysis:', cached)
        print((cached / 'summary.json').read_text())
        if args.save_images:
            from render_counting import render
            render(cached, dataset, args.save_images)
        return
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if model.names != {0: 'apple'}:
        raise ValueError(f'Expected trained single-class apple weights, got {model.names}')
    experiment = model_group(weights)
    run_name = (datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                + f'_{args.split}_{safe_name(weights.stem)}_conf_{args.conf:g}_imgsz_{args.imgsz}_iou_{args.iou:g}_{tag(preprocessing)}')
    output = ROOT / 'runs/counting' / experiment / run_name
    output.mkdir(parents=True)
    rows = []
    predictions = []
    for image_path in images:
        prediction_source = str(image_path) if args.preprocess == 'none' else read_processed(image_path, preprocessing)
        result = model.predict(source=prediction_source, conf=args.conf, iou=args.iou,
                               imgsz=args.imgsz, device=args.device, max_det=1000, verbose=False)[0]
        predicted = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        width, height = image.size
        truth = []
        for line in (dataset / 'labels' / args.split / image_path.with_suffix('.txt').name).read_text().splitlines():
            _, cx, cy, w, h = map(float, line.split())
            truth.append(((cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height))
        delta = len(predicted) - len(truth)
        rows.append(dict(image=image_path.name, annotated=len(truth), predicted=len(predicted),
                         difference=delta, absolute_error=abs(delta)))
        predictions.append(dict(image=image_path.name, boxes_xyxy=predicted, confidence=scores))
    with (output / 'counts.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['absolute_error'], reverse=True))
    summary = dict(images=len(rows), MAE=sum(r['absolute_error'] for r in rows)/len(rows),
                   RMSE=math.sqrt(sum(r['difference']**2 for r in rows)/len(rows)),
                   mean_bias=sum(r['difference'] for r in rows)/len(rows),
                   preprocessing=preprocessing, split=args.split, dataset=str(dataset), dataset_sha256=identity['dataset_sha256'],
                   weights=str(weights.resolve()), experiment=experiment, checkpoint=weights.stem, conf=args.conf, nms_iou=args.iou, imgsz=args.imgsz,
                   note='Counts are evaluated against dataset annotations; equal counts do not imply correct localization.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'predictions.json').write_text(json.dumps(predictions), encoding='utf-8')
    finish(output, identity)
    if args.save_images:
        from render_counting import render
        render(output, dataset, args.save_images)
    print(json.dumps(summary, indent=2))
    print(f'Counting results (images are optional): {output}')


if __name__ == '__main__':
    main()

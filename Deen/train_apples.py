"""Train a local apple detector. Run with the project's .venv Python."""
import argparse
import json
from pathlib import Path
from run_layout import safe_name, timestamp, resolve_weights

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/apple_yolo')
    parser.add_argument('--device', default='0', help='CUDA index, or cpu')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--model', default='yolo11n.pt')
    parser.add_argument('--name', default=None, help='Optional label appended to automatic experiment name')
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.imgsz) <= 0:
        parser.error('epochs, batch and imgsz must be positive')
    dataset = args.data.resolve()
    for split in ('train', 'val'):
        images = sorted((dataset / 'images' / split).glob('*.png'))
        if not images:
            raise FileNotFoundError(f'No images for {split}: {dataset}')
        count = 0
        for image in images:
            label = dataset / 'labels' / split / image.with_suffix('.txt').name
            for line in label.read_text().splitlines():
                cls, x, y, w, h = map(float, line.split())
                if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f'Invalid label: {label}: {line}')
                count += 1
        print(f'{split}: {len(images)} images, {count} apple labels')
    if args.check_data:
        return
    import torch
    from ultralytics import YOLO
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable. Run .venv/Scripts/python.exe check_gpu.py')
    # Absolute root prevents dataset paths depending on the launch directory.
    config = dataset / 'data.local.yaml'
    config.write_text(f'path: {json.dumps(dataset.as_posix())}\ntrain: images/train\nval: images/val\nnames:\n  0: apple\n', encoding='utf-8')
    model_path = resolve_weights(args.model)
    model = YOLO(str(model_path))
    architecture = Path(model.model.yaml.get('yaml_file', model_path.name)).stem
    run_name = f'{timestamp()}_{safe_name(architecture)}_ep{args.epochs}_sz{args.imgsz}_b{args.batch}'
    if args.name:
        run_name += '_' + safe_name(args.name)
    model.train(data=str(config), epochs=args.epochs, batch=args.batch,
                imgsz=args.imgsz, device=args.device, workers=0, seed=42,
                project=str(ROOT / 'runs/detect'), name=run_name,
                exist_ok=False, pretrained=True, cache=False, amp=True,
                close_mosaic=0, max_det=1000, plots=True)
    best = Path(model.trainer.best).resolve()
    metadata = dict(experiment=best.parent.parent.name, architecture=architecture,
                    initial_weights=str(model_path), requested_epochs=args.epochs,
                    imgsz=args.imgsz, batch=args.batch, dataset=str(dataset))
    (best.parent.parent / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (ROOT / 'last_training.json').write_text(json.dumps({'best_weights': str(best)}, indent=2), encoding='utf-8')
    print(f'Best weights: {best}')
    print('Next: .\\.venv\\Scripts\\python.exe predict_apples.py')


if __name__ == '__main__':
    main()

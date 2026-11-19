"""Content fingerprints and integrity checks for completed counting runs."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from lighting_preprocess import settings


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hash(dataset, split='val'):
    images = sorted((dataset / 'images' / split).glob('*.png'))
    if not images:
        raise ValueError('No validation images')
    digest = hashlib.sha256()
    for image in images:
        for path in (image, dataset / 'labels' / split / image.with_suffix('.txt').name):
            digest.update(json.dumps([path.relative_to(dataset).as_posix(), sha256(path)]).encode())
    return digest.hexdigest()


def fingerprint(weights, dataset, args):
    import cv2
    preprocessing = settings(args.preprocess, args.gamma, args.clahe_clip, args.clahe_grid)
    return dict(schema=2, preprocessing=preprocessing, opencv=cv2.__version__,
                preprocessing_code=sha256(Path(__file__).with_name('lighting_preprocess.py')), weights_sha256=sha256(weights), dataset_sha256=dataset_hash(dataset, args.split), split=args.split,
                conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, max_det=1000,
                versions={name: version(name) for name in ('torch', 'torchvision', 'ultralytics')},
                implementation=sha256(Path(__file__).with_name('predict_apples.py')),
                storage_implementation=sha256(__file__))


def finish(output, identity):
    files = {name: sha256(output / name) for name in ('summary.json', 'counts.csv', 'predictions.json')}
    # Written last: a partial run never qualifies for reuse.
    (output / 'complete.json').write_text(json.dumps(dict(identity=identity, files=files), indent=2))


def find_cached(parent, identity):
    for marker in sorted(parent.glob('*/complete.json'), reverse=True):
        try:
            data = json.loads(marker.read_text())
            if data['identity'] == identity and all(
                sha256(marker.parent / name) == digest for name, digest in data['files'].items()
            ) and set(data['files']) == {'summary.json', 'counts.csv', 'predictions.json'}:
                return marker.parent
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None

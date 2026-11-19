"""Validate official COCO annotations and create a test-only YOLO dataset."""
import json
import math
import os
from pathlib import Path
from PIL import Image

ROOT = Path('E:/MV/apple_counting')
source = ROOT / 'MinneApple/detection/detection/test/images'
annotations = ROOT / 'MinneApple/test_data/test_data/detection'
truth = json.loads((annotations / 'ground_truth.json').read_text())
mapping = json.loads((annotations / 'mapping.json').read_text())
output = ROOT / 'data/apple_official_test'
if output.exists():
    raise FileExistsError(output)
assert truth['categories'] == [{'supercategory': '', 'id': 0, 'name': 'apple'}]
images = {r['id']: r for r in truth['images']}
assert len(images) == len(truth['images']) == 331
assert {r['filename']: r['id'] for r in images.values()} == mapping
assert set(mapping) == {p.name for p in source.glob('*.png')}
labels = {i: [] for i in images}
ids = set()
for a in truth['annotations']:
    assert a['id'] not in ids
    ids.add(a['id'])
    assert a['category_id'] == 0 and a['iscrowd'] == 0
    im = images[a['image_id']]
    x, y, w, h = a['bbox']
    assert all(math.isfinite(v) for v in (x,y,w,h))
    assert w > 0 and h > 0 and x >= 0 and y >= 0
    assert x+w <= im['width'] and y+h <= im['height']
    labels[a['image_id']].append(f"0 {(x+w/2)/im['width']:.10f} {(y+h/2)/im['height']:.10f} {w/im['width']:.10f} {h/im['height']:.10f}")
for im in images.values():
    with Image.open(source / im['filename']) as picture:
        assert picture.size == (im['width'],im['height'])
(output / 'images/test').mkdir(parents=True)
(output / 'labels/test').mkdir(parents=True)
for i, im in images.items():
    # Hard links reference existing image bytes on the same drive, avoiding a second copy.
    os.link(source / im['filename'], output / 'images/test' / im['filename'])
    (output / 'labels/test' / Path(im['filename']).with_suffix('.txt')).write_text('\n'.join(labels[i])+'\n', encoding='utf-8')
(output / 'summary.json').write_text(json.dumps(dict(images=len(images), apples=len(ids), source=str(annotations), image_storage='hard links; do not edit original images', split='test'), indent=2))
print(f'Validated and prepared {len(images)} test images, {len(ids)} apples: {output}')

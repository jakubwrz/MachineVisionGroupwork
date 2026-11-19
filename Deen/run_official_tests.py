"""Four fixed validation-selected settings, frozen before test evaluation."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
experiments = [
    ('M0', '20261117_144716_yolo11n_ep30_sz640_b4', .275, .65, 640),
    ('M1', '20261117_201626_043049_yolo11n_ep30_sz960_b4_M1_resolution960', .275, .65, 800),
    ('M2', '20261117_210935_674221_yolo11n_ep60_sz640_b4_M2_epochs60', .275, .6, 800),
    ('M3', '20261117_223911_576210_yolo11s_ep30_sz640_b4_M3_yolo11s', .275, .6, 640),
]
plan = [dict(model=m, experiment=e, conf=c, nms_iou=i, imgsz=z) for m,e,c,i,z in experiments]
plan_path = ROOT / 'runs/official_test_plan.json'
if plan_path.exists():
    assert json.loads(plan_path.read_text()) == plan
else:
    plan_path.write_text(json.dumps(plan, indent=2))
for model, experiment, conf, iou, size in experiments:
    print(f'Official test: {model}', flush=True)
    subprocess.run([sys.executable, str(ROOT/'predict_apples.py'),
                    '--weights', str(ROOT/'runs/detect'/experiment/'weights/best.pt'),
                    '--data', str(ROOT/'data/apple_official_test'), '--split', 'test',
                    '--baseline', '--conf', str(conf), '--iou', str(iou), '--imgsz', str(size),
                    '--save-images', '331'], cwd=ROOT, check=True)
print('All four fixed-parameter test runs complete.', flush=True)

"""Five fixed validation-only lighting ablations, with no retraining."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from lighting_preprocess import settings, tag

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-images', type=int, default=10, help='Worst N per experiment; 0 for no PNGs')
    args = parser.parse_args()
    if args.save_images < 0:
        parser.error('save-images must be nonnegative')
    experiment = '20261117_201626_043049_yolo11n_ep30_sz960_b4_M1_resolution960'
    weights = ROOT / 'runs/detect' / experiment / 'weights/best.pt'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    configs = [settings(), settings('gamma', gamma=0.8), settings('gamma', gamma=0.6),
               settings('clahe', clip=1.5), settings('clahe', clip=2.0)]
    report = ROOT / 'reports' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_M1_lighting')
    report.mkdir(parents=True)
    (report / 'plan.json').write_text(json.dumps(dict(weights=str(weights), split='val', conf=0.275,
        iou=0.65, imgsz=800, configurations=configs), indent=2), encoding='utf-8')
    results = []
    for config in configs:
        command = [sys.executable, str(ROOT / 'predict_apples.py'), '--weights', str(weights),
                   '--data', str(ROOT / 'data/apple_yolo'), '--split', 'val', '--baseline',
                   '--conf', '0.275', '--iou', '0.65', '--imgsz', '800',
                   '--save-images', str(args.save_images), '--preprocess', config['mode']]
        if config['mode'] == 'gamma':
            command += ['--gamma', str(config['gamma'])]
        elif config['mode'] == 'clahe':
            command += ['--clahe-clip', str(config['clip_limit']), '--clahe-grid', str(config['grid'])]
        print('Running lighting comparison:', tag(config), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        matches = []
        for file in (ROOT / 'runs/counting' / experiment).glob('*/summary.json'):
            data = json.loads(file.read_text())
            if (data.get('preprocessing') == config and data.get('split') == 'val'
                    and data['conf'] == .275 and data['nms_iou'] == .65 and data['imgsz'] == 800
                    and (file.parent / 'complete.json').exists()):
                matches.append((file, data))
        file, result = max(matches, key=lambda item: item[0].parent.name)
        results.append(dict(treatment=tag(config), MAE=result['MAE'], RMSE=result['RMSE'],
                            mean_bias=result['mean_bias'], result_directory=str(file.parent)))
        (report / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    lines = ['# M1 lighting comparison on validation data', '',
             '| Input treatment | MAE | RMSE | Mean bias |', '|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['treatment']} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['mean_bias']:+.4f} |")
    lines += ['', 'All 134 validation images are evaluated. These are not shadow-only metrics.',
              'The model and thresholds are fixed. No test-set selection is performed.', '']
    for r in results:
        lines.append(f"- [{r['treatment']}]({Path(r['result_directory']).as_posix()})")
    (report / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('All five experiments complete. Report:', report)


if __name__ == '__main__':
    main()

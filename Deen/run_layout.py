"""Shared experiment naming and compatibility with pre-migration paths."""
import json
import re
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._') or 'model'

def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')

def resolve_weights(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    mapping = ROOT / 'runs/layout_migration.json'
    if not path.exists() and mapping.exists():
        for item in json.loads(mapping.read_text(encoding='utf-8'))['training']:
            old = Path(item['old'])
            if path.is_relative_to(old):
                path = Path(item['new']) / path.relative_to(old)
                break
    return path.resolve()

def model_group(weights):
    weights = Path(weights).resolve()
    if weights.parent.name == 'weights' and weights.parent.parent.parent == ROOT / 'runs/detect':
        return weights.parent.parent.name
    import hashlib
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()[:12]
    return f'external_{safe_name(weights.stem)}_{digest}'

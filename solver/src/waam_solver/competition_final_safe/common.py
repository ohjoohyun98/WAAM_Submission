import hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/competition_final_safe'
REPORT = ROOT / 'reports/competition_final_safe'
TOLERANCE_S = 1e-7

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def tree_hash(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}

def selection_key(row):
    # No positive epsilon tie band: even a slightly slower high-IoU candidate
    # cannot displace candidate zero. Shape robustness breaks EXACT ties only.
    s = row['official']['summary']
    return (s['schedule']['makespan_s'], -s['shape']['iou'],
            -s['shape']['coverage'], s['shape']['overfill_ratio'], row['candidate_source'])

def eligible(row):
    official = row.get('official', {})
    s = official.get('summary', {})
    return bool(official.get('passed') and s.get('status') == 'PASS'
                and s.get('shape', {}).get('passed') and s.get('reach', {}).get('passed')
                and s.get('collision', {}).get('passed')
                and s.get('collision', {}).get('collision_event_count') == 0
                and math.isfinite(s.get('schedule', {}).get('makespan_s', float('inf'))))

def select(rows):
    valid = [r for r in rows if eligible(r)]
    return min(valid, key=selection_key) if valid else None

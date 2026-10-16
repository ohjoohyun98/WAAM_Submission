"""Release-only persistence, provenance, independent copies and timers."""
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clone_scenario(s):
    from shapely import get_precision, set_precision
    result = copy.deepcopy(s)
    result.layer_geometry = {k:set_precision(g,get_precision(s.layer_geometry[k]),mode='pointwise')
                             for k,g in result.layer_geometry.items()}
    for task in result.tasks:
        task.points.setflags(write=False)
    return result


def scenario_hash(s):
    import numpy as np
    from shapely import get_precision
    h = hashlib.sha256(json.dumps(s.config, sort_keys=True).encode())
    for k,g in sorted(s.layer_geometry.items()):
        h.update(str((k,float(get_precision(g)))).encode()+g.wkb)
    for t in s.tasks:
        h.update(str((t.id,t.layer,t.robots)).encode()+np.ascontiguousarray(t.points).tobytes())
    h.update(json.dumps(s.scores,sort_keys=True).encode())
    return h.hexdigest()


class Timers:
    def __init__(self):
        self.seconds = {}
        self.stack = []

    @contextmanager
    def stage(self, name):
        item = [perf_counter(), 0.]
        self.stack.append(item)
        try:
            yield
        finally:
            elapsed = perf_counter()-item[0]
            self.stack.pop()
            self.seconds[name] = self.seconds.get(name,0.)+elapsed-item[1]
            print(f'STAGE {name} {elapsed-item[1]:.3f}s',flush=True)
            if self.stack:
                self.stack[-1][1] += elapsed

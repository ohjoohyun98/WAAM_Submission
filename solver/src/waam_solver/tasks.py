import numpy as np
from .models import Task
from .profiling import timed,stage,count


@timed('task_generation')
def generate_tasks(layers, paths, config, residual_indices=None):
    """Split paths into maximal runs sharing a complete reach-feasible set."""
    tasks = []
    p = config['process']
    for k in sorted(layers):
        z = p['build_plane_z_mm'] + (k+(1 if p['tcp_z_reference']=='top' else .5))*p['layer_height_mm']
        for path_id,line in enumerate(paths.get(k, [])):
            residual=path_id in (residual_indices or {}).get(k,())
            # Add analytical circle intersection points before assigning segments.
            split = [line[0]]
            for a,b in zip(line[:-1], line[1:]):
                d = b-a
                aa = float(d@d)
                if aa < 1e-12:
                    continue
                ts = {1.0}
                for robot in config['robots']:
                    offset = a-np.asarray(robot['base_xyz_mm'][:2])
                    radius = robot['xy_reach_radius_mm']-0.001
                    bb, cc = 2*float(d@offset), float(offset@offset)-radius**2
                    disc = bb*bb-4*aa*cc
                    if disc > 0:
                        for t in ((-bb-np.sqrt(disc))/(2*aa),(-bb+np.sqrt(disc))/(2*aa)):
                            if 1e-9 < t < 1-1e-9:
                                ts.add(float(t))
                split.extend(a+t*d for t in sorted(ts))
            current, eligible = [], None
            for a,b in zip(split[:-1], split[1:]):
                if np.linalg.norm(b-a) < 1e-5:
                    continue
                with stage('reach_feasibility'):
                    ids = tuple(robot['id'] for robot in config['robots']
                                if max(np.linalg.norm(a-np.asarray(robot['base_xyz_mm'][:2])),
                                       np.linalg.norm(b-np.asarray(robot['base_xyz_mm'][:2])))
                                <= robot['xy_reach_radius_mm']-0.0005)
                if not ids:
                    raise ValueError(f"Unreachable deposition segment in layer {k}")
                if eligible != ids:
                    if len(current)>1:
                        tasks.append(Task(len(tasks), k, np.column_stack([current, np.full(len(current), z)]), eligible,residual))
                    current, eligible = [a,b], ids
                else:
                    current.append(b)
            if len(current)>1:
                tasks.append(Task(len(tasks), k, np.column_stack([current, np.full(len(current), z)]), eligible,residual))
    count('deposition_task_count',len(tasks))
    count('deposition_segment_count',sum(len(t.points)-1 for t in tasks))
    count('candidate_robot_task_pairs',sum(len(t.robots) for t in tasks))
    return tasks

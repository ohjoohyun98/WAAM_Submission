"""Exact vectorized form of certify_move for one robot's straight intervals."""
import numpy as np
from shapely import multipoints,convex_hull,distance,Point,MultiPoint
from .profiling import timed,count


@timed('internal_validation')
def certify_moves(robot_id,starts,ends,parking,config):
    robot=next(r for r in config['robots'] if r['id']==robot_id)
    a=np.asarray(starts,dtype=float)[:,:2]; b=np.asarray(ends,dtype=float)[:,:2]
    base=np.asarray(robot['base_xyz_mm'][:2])
    limit=robot['xy_reach_radius_mm']-1e-5
    if np.any(np.linalg.norm(a-base,axis=1)>limit) or np.any(np.linalg.norm(b-base,axis=1)>limit):
        raise ValueError(f'R{robot_id} move exceeds reach')
    swept=convex_hull(multipoints(np.stack([np.broadcast_to(base,a.shape),a,b],axis=1)))
    tcp=convex_hull(multipoints(np.stack([a,b],axis=1)))
    c=config['collision']
    for other in config['robots']:
        if other['id']==robot_id:
            continue
        parked=MultiPoint([other['base_xyz_mm'][:2],parking[other['id']][:2]]).convex_hull
        if c['check_arm_envelope']:
            count('static_detailed_collision_checks',len(a))
            required=robot['arm_envelope_radius_mm']+other['arm_envelope_radius_mm']+c['arm_clearance_mm']
            if np.any(distance(swept,parked)<=required+.01):
                raise ValueError(f'R{robot_id} swept arm conflicts with parked R{other["id"]}')
        if c['check_tcp_radius']:
            count('static_detailed_collision_checks',len(a))
            required=robot['tcp_radius_mm']+other['tcp_radius_mm']
            if np.any(distance(tcp,Point(parking[other['id']][:2]))<=required+.01):
                raise ValueError(f'R{robot_id} swept TCP conflicts with parked R{other["id"]}')

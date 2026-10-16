"""Continuous conservative swept-arm certification for serial execution."""
import numpy as np
from shapely.geometry import MultiPoint, Point
from .profiling import timed,count


@timed('internal_validation')
def certify_move(robot_id, start, end, parking, config):
    robot = next(r for r in config['robots'] if r['id']==robot_id)
    base = np.asarray(robot['base_xyz_mm'][:2])
    radius = robot['xy_reach_radius_mm']
    if max(np.linalg.norm(start[:2]-base),np.linalg.norm(end[:2]-base)) > radius-1e-5:
        raise ValueError(f"R{robot_id} move exceeds reach")
    swept = MultiPoint([base,start[:2],end[:2]]).convex_hull
    tcp_swept = MultiPoint([start[:2],end[:2]]).convex_hull
    c = config['collision']
    for other in config['robots']:
        if other['id']==robot_id:
            continue
        parked = MultiPoint([other['base_xyz_mm'][:2], parking[other['id']][:2]]).convex_hull
        arm_req = robot['arm_envelope_radius_mm']+other['arm_envelope_radius_mm']+c['arm_clearance_mm']
        tcp_req = robot['tcp_radius_mm']+other['tcp_radius_mm']
        count('static_detailed_collision_checks',int(c['check_arm_envelope'])+int(c['check_tcp_radius']))
        if c['check_arm_envelope'] and swept.distance(parked) <= arm_req+0.01:
            raise ValueError(f"R{robot_id} swept arm conflicts with parked R{other['id']}")
        if c['check_tcp_radius'] and tcp_swept.distance(Point(parking[other['id']][:2])) <= tcp_req+0.01:
            raise ValueError(f"R{robot_id} swept TCP conflicts with parked R{other['id']}")

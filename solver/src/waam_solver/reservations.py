"""Exact forbidden-time union with conservative temporal/spatial broad phases."""
import numpy as np
from shapely import STRtree,box,bounds
from .profiling import count


def merge_intervals(intervals):
    """Union closed intervals, without filling any positive-size gap.

    The execution engine alone applies its existing 1e-5 s guard. Do not add
    that guard here: preserving even tiny gaps preserves reference decisions.
    """
    data=np.asarray(intervals,dtype=float)
    if not data.size:
        return []
    data=data[np.argsort(data[:,0],kind='stable')]
    upper=np.maximum.accumulate(data[:,1])
    first=np.r_[0,1+np.flatnonzero(data[1:,0]>upper[:-1])]
    last=np.r_[first[1:]-1,len(data)-1]
    return list(zip(data[first,0].tolist(),upper[last].tolist()))


def forbidden_intervals(engine,tour,existing,ready):
    """Keep the original GEOS dwithin narrow phase and time arithmetic.

    Interval right endpoint is existing.start + other_hi - own_lo. An
    interval with right < ready cannot affect any start >= ready. The two
    temporal masks use extrema as necessary conditions, not time samples.
    """
    c=engine.config['collision']
    robots={r['id']:r for r in engine.config['robots']}
    a,b=robots[tour.robot],robots[existing.robot]
    wa=np.asarray(tour.windows); wb=np.asarray(existing.windows)
    if not len(wa) or not len(wb):
        return []
    ia=np.flatnonzero(existing.start+wb[:,1].max()-wa[:,0]>=ready)
    ib=np.flatnonzero(existing.start+wb[:,1]-wa[:,0].min()>=ready)
    count('temporal_primitive_prunes',len(wa)+len(wb)-len(ia)-len(ib))
    if not len(ia) or not len(ib):
        return []
    chunks=[]
    for enabled,geoms,other,required in (
        (c['check_arm_envelope'],tour.arm,existing.arm,a['arm_envelope_radius_mm']+b['arm_envelope_radius_mm']+c['arm_clearance_mm']),
        (c['check_tcp_radius'],tour.tcp,existing.tcp,a['tcp_radius_mm']+b['tcp_radius_mm'])):
        if not enabled:
            continue
        geoms=np.asarray(geoms,dtype=object)[ia]
        other=np.asarray(other,dtype=object)[ib]
        tree=STRtree(other)
        bb=bounds(geoms); radius=required+.02
        broad=tree.query(box(bb[:,0]-radius,bb[:,1]-radius,bb[:,2]+radius,bb[:,3]+radius))
        count('collision_candidate_pair_count',broad.shape[1])
        count('dynamic_detailed_collision_checks',broad.shape[1])
        pairs=tree.query(geoms,predicate='dwithin',distance=radius)
        count('collision_conflicting_pair_count',pairs.shape[1])
        if not pairs.size:
            continue
        own=wa[ia[pairs[0]]]; previous=wb[ib[pairs[1]]]
        intervals=np.column_stack((existing.start+previous[:,0]-own[:,1],
                                   existing.start+previous[:,1]-own[:,0]))
        intervals=intervals[intervals[:,1]>=ready]
        # Compress each channel before combining, avoiding giant Python tuples.
        chunks.extend(merge_intervals(intervals))
    compact=merge_intervals(chunks)
    count('reservation_intervals_retained',len(compact))
    return compact

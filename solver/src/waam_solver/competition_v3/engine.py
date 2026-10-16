"""Continuous conservative interval reservations with bounded spatial queries."""
from dataclasses import replace
import numpy as np
from shapely import STRtree, convex_hull, multipoints
from ..parallel_engine import ParallelExecutionEngine
from ..reservations import merge_intervals

class Engine(ParallelExecutionEngine):
    def __init__(self,config,target_top,*,local_height=True,primitive_mm=10.):
        super().__init__(config,target_top)
        self.local_height=local_height; self.primitive_mm=primitive_mm
        self.clearance=max(config['process']['bead_width_mm'],config['process']['layer_height_mm'],
                           *(r['tcp_radius_mm'] for r in config['robots']))+.02
        self.query_pairs=0

    def connection(self,a,b):
        # Global barrier: no deposited material above the current layer top.
        # Fixed high parking is retained. XY collision rules remain unchanged.
        z=self.travel_z
        if self.local_height:
            offset=self.config['process']['layer_height_mm']*(.5 if self.config['process']['tcp_z_reference']=='center' else 0.)
            z=min(z,max(a[2],b[2])+offset+self.clearance)
        return [np.array([*a[:2],z]),np.array([*b[:2],z]),b]

    def build_tour(self,robot,actions):
        tour=super().build_tour(robot,actions)
        if self.primitive_mm==10.: return tour
        base=next(r['base_xyz_mm'][:2] for r in self.config['robots'] if r['id']==robot)
        xs=[]; ys=[]; windows=[]
        for (t,a,_),(u,b,_) in zip(tour.rows[:-1],tour.rows[1:]):
            n=max(1,int(np.ceil(np.linalg.norm(b[:2]-a[:2])/self.primitive_mm)))
            for i in range(n):
                xs.append(a[:2]+(b[:2]-a[:2])*i/n); ys.append(a[:2]+(b[:2]-a[:2])*(i+1)/n)
                windows.append((t+(u-t)*i/n,t+(u-t)*(i+1)/n))
        tour.arm=convex_hull(multipoints(np.stack([np.broadcast_to(base,(len(xs),2)),xs,ys],axis=1)))
        tour.tcp=convex_hull(multipoints(np.stack([xs,ys],axis=1))); tour.windows=windows
        return tour

    def forbidden(self,tour,existing,ready=None):
        """Same continuous conservative semantics, no dense all-pair allocation.

        Spatial query chunks cap intermediate pair memory. No redundant broad
        query just for counters. Temporal filtering precedes expensive GEOS.
        """
        ready=float('-inf') if ready is None else ready
        key=(id(tour),id(existing),existing.start)
        cached=self._reservation_cache.get(key)
        if cached is not None and cached[0]<=ready: return cached[1]
        wa=np.asarray(tour.windows); wb=np.asarray(existing.windows)
        ia=np.flatnonzero(existing.start+wb[:,1].max()-wa[:,0]>=ready)
        ib=np.flatnonzero(existing.start+wb[:,1]-wa[:,0].min()>=ready)
        robots={r['id']:r for r in self.config['robots']}; a=robots[tour.robot]; b=robots[existing.robot]
        c=self.config['collision']; result=[]
        if len(ia) and len(ib):
            for enabled,own,other,radius in (
                (c['check_arm_envelope'],tour.arm,existing.arm,a['arm_envelope_radius_mm']+b['arm_envelope_radius_mm']+c['arm_clearance_mm']),
                (c['check_tcp_radius'],tour.tcp,existing.tcp,a['tcp_radius_mm']+b['tcp_radius_mm'])):
                if not enabled: continue
                tree=STRtree(np.asarray(other,dtype=object)[ib])
                for offset in range(0,len(ia),128):
                    selected=ia[offset:offset+128]
                    pairs=tree.query(np.asarray(own,dtype=object)[selected],predicate='dwithin',distance=radius+.02)
                    self.query_pairs+=pairs.shape[1]
                    if not pairs.size: continue
                    x=wa[selected[pairs[0]]]; y=wb[ib[pairs[1]]]
                    intervals=np.column_stack((existing.start+y[:,0]-x[:,1],existing.start+y[:,1]-x[:,0]))
                    result.extend(merge_intervals(intervals[intervals[:,1]>=ready]))
        result=merge_intervals(result); self._reservation_cache[key]=(ready,result)
        return result

    def commit(self,tour,release,start):
        super().commit(tour,release,start)
        # Completed tours cannot collide with any future ready time. Keep a
        # small live frontier rather than all layers' Shapely objects forever.
        floor=min(rows[-1][0] for rows in self.rows.values())
        self.tours=[t for t in self.tours if t.start+t.duration>=floor-1e-8]

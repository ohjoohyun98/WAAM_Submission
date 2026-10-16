"""Continuous swept-volume reservations for independently timed robot tours.

A tour starts/ends at base parking. Each primitive is subdivided spatially;
overlap of two swept capsules produces a forbidden *start-time interval*.
Earliest-start selection is exact within this conservative reservation model.
It does not claim a globally minimal continuous-time collision schedule.
"""
from dataclasses import dataclass
import numpy as np
from shapely import MultiPoint, STRtree, multipoints, convex_hull
from .collision import certify_move
from .batch_collision import certify_moves
from .reservations import forbidden_intervals
from .profiling import timed,stage,count


@dataclass
class Tour:
    robot: int
    rows: list
    actions: list
    arm: list
    tcp: list
    windows: list
    start: float = 0.0

    @property
    def duration(self):
        return self.rows[-1][0]


class ParallelExecutionEngine:
    def __init__(self, config, target_top, *, reservation_mode='optimized'):
        self.config=config
        if reservation_mode not in ('optimized','reference'):
            raise ValueError('Unknown reservation mode')
        self.reservation_mode=reservation_mode
        # Entire target bounds, not merely currently deposited layers. Horizontal
        # TCP travel clears all target material even for changing cross sections.
        self.travel_z=target_top+max(config['process']['bead_width_mm'],config['process']['layer_height_mm'],
                                    *(r['tcp_radius_mm'] for r in config['robots']))+0.02
        self.parking={r['id']:np.array([*r['base_xyz_mm'][:2],self.travel_z]) for r in config['robots']}
        self.rows={r:[(0.,p.copy(),'W')] for r,p in self.parking.items()}
        self.tours=[]
        self.clock=0.
        self.decisions=[]
        self.waits=[]
        self._reservation_cache={}
        self.robot_stats={r:{'deposition_time_s':0.,'travel_time_s':0.,'wait_time_s':0.,
                             'last_layer':None,'last_task':None} for r in self.parking}
        for r,p in self.parking.items():
            certify_move(r,p,p,self.parking,config)

    def connection(self, a,b):
        return [np.array([*a[:2],self.travel_z]),np.array([*b[:2],self.travel_z]),b]

    def travel_length(self,a,b):
        pts=[a,*self.connection(a,b)]
        return sum(float(np.linalg.norm(y-x)) for x,y in zip(pts[:-1],pts[1:]))

    @timed('trajectory_generation')
    def build_tour(self,robot,actions):
        if not actions or len({a.task.id for a in actions})!=len(actions):
            raise ValueError('Tour requires nonempty, unique tasks')
        if len({a.task.layer for a in actions})!=1:
            raise ValueError('Tour crosses layer barrier')
        rows=[(0.,self.parking[robot].copy(),'W')]
        proc=self.config['process']
        def append(end,duration,mode):
            if duration<=1e-10:
                return
            t,a,_=rows[-1]
            rows[-1]=(t,a,mode)
            rows.append((t+duration,end.copy(),'W'))
        def travel(end):
            for point in self.connection(rows[-1][1],end):
                append(point,float(np.linalg.norm(point-rows[-1][1]))/proc['travel_speed_mm_s'],'T')
        for action in actions:
            if action.robot!=robot or robot not in action.task.robots:
                raise ValueError('Unreachable tour action')
            pts=action.task.points[::-1] if action.reverse else action.task.points
            travel(pts[0])
            append(pts[0],proc.get('arc_on_time_s') or 0.,'W')
            for b in pts[1:]:
                append(b,float(np.linalg.norm(b-rows[-1][1]))/proc['deposition_speed_mm_s'],'D')
            append(pts[-1],proc.get('arc_off_time_s') or 0.,'W')
        travel(self.parking[robot])
        certify_moves(robot,[r[1] for r in rows[:-1]],[r[1] for r in rows[1:]],self.parking,self.config)
        base=next(r['base_xyz_mm'][:2] for r in self.config['robots'] if r['id']==robot)
        xs,ys,windows=[],[],[]
        # 10 mm swept primitives; continuous conservative checks, not point samples.
        for (t,a,mode),(u,b,_) in zip(rows[:-1],rows[1:]):
            n=max(1,int(np.ceil(np.linalg.norm(b[:2]-a[:2])/10.)))
            for i in range(n):
                x=a[:2]+(b[:2]-a[:2])*i/n; y=a[:2]+(b[:2]-a[:2])*(i+1)/n
                xs.append(x); ys.append(y)
                windows.append((t+(u-t)*i/n,t+(u-t)*(i+1)/n))
        arm=convex_hull(multipoints(np.stack([np.broadcast_to(base,(len(xs),2)),xs,ys],axis=1)))
        tcp=convex_hull(multipoints(np.stack([xs,ys],axis=1)))
        return Tour(robot,rows,actions,arm,tcp,windows)

    @staticmethod
    def earliest(release, intervals):
        start=release
        for left,right in sorted(intervals):
            if right<start:
                continue
            if left>start:
                break
            start=max(start,right+1e-5)
        return start

    @timed('collision_candidate_generation')
    def forbidden(self,tour,existing,ready=None):
        key=(id(tour),id(existing),existing.start)
        bound=float('-inf') if ready is None else ready
        cached=self._reservation_cache.get(key)
        if cached is not None and cached[0]<=bound:
            return cached[1]
        if self.reservation_mode=='optimized':
            intervals=forbidden_intervals(self,tour,existing,bound)
            self._reservation_cache[key]=(bound,intervals)
            return intervals
        c=self.config['collision']
        robots={r['id']:r for r in self.config['robots']}
        a,b=robots[tour.robot],robots[existing.robot]
        intervals=[]
        for enabled,geoms,other,required in (
            (c['check_arm_envelope'],tour.arm,existing.arm,a['arm_envelope_radius_mm']+b['arm_envelope_radius_mm']+c['arm_clearance_mm']),
            (c['check_tcp_radius'],tour.tcp,existing.tcp,a['tcp_radius_mm']+b['tcp_radius_mm'])):
            if not enabled:
                continue
            tree=STRtree(other)
            # Instrument exact narrow phase without changing the accepted pairs.
            from shapely import box,bounds
            bb=bounds(geoms)
            expanded=box(bb[:,0]-required-.02,bb[:,1]-required-.02,bb[:,2]+required+.02,bb[:,3]+required+.02)
            broad=tree.query(expanded)
            count('collision_candidate_pair_count',broad.shape[1])
            count('dynamic_detailed_collision_checks',broad.shape[1])
            pairs=tree.query(geoms,predicate='dwithin',distance=required+0.02)
            count('collision_conflicting_pair_count',pairs.shape[1])
            for i,j in pairs.T:
                lo,hi=tour.windows[i]; elo,ehi=existing.windows[j]
                intervals.append((existing.start+elo-hi,existing.start+ehi-lo))
        self._reservation_cache[key]=(bound,intervals)
        return intervals

    @timed('parallel_reservation')
    def earliest_start(self,tour,release):
        count('reservation_iterations')
        intervals=[]
        ready=max(release,self.rows[tour.robot][-1][0])
        for other in self.tours:
            if other.robot!=tour.robot and other.start+other.duration>=ready:
                intervals.extend(self.forbidden(tour,other,ready))
        return self.earliest(ready,intervals)

    @timed('wait_insertion')
    def commit(self,tour,release,start):
        r=tour.robot
        previous=self.rows[r][-1][0]
        if start<max(previous,release)-1e-9:
            raise ValueError('Tour violates robot/layer precedence')
        if self.earliest_start(tour,start)>start+1e-9:
            raise ValueError('Tour start conflicts with existing reservations')
        barrier=max(0.,release-previous)
        collision=max(0.,start-max(previous,release))
        for reason,begin,duration in [('layer_precedence',previous,barrier),
                                      ('collision_reservation',max(previous,release),collision)]:
            if duration>1e-10:
                self.waits.append({'robot':r,'reason':reason,'start_s':begin,'duration_s':duration})
        if start>previous+1e-10:
            self.rows[r].append((start,self.parking[r].copy(),'W'))
        # Replace existing terminal W with the outgoing mode of the new tour.
        self.rows[r][-1]=(start,self.parking[r].copy(),tour.rows[0][2])
        self.rows[r].extend((start+t,p,m) for t,p,m in tour.rows[1:])
        tour.start=start
        self.tours.append(tour)
        # Cache holds tour references indirectly through live self.tours/queues;
        # discard after each commit to bound memory and avoid object-id reuse.
        self._reservation_cache.clear()
        self.clock=max(self.clock,start+tour.duration)
        stats=self.robot_stats[r]
        stats['wait_time_s']+=start-previous
        for a,b in zip(tour.rows[:-1],tour.rows[1:]):
            stats[{'D':'deposition_time_s','T':'travel_time_s','W':'wait_time_s'}[a[2]]]+=b[0]-a[0]
        stats['last_layer']=tour.actions[-1].task.layer
        stats['last_task']=tour.actions[-1].task.id
        for action in tour.actions:
            self.decisions.append({'task':action.task.id,'layer':action.task.layer,
                                   'robot':r,'reverse':action.reverse,'tour_start_s':start})

    def finish(self):
        for r,rows in self.rows.items():
            if len(rows)<2:
                rows.append((max(self.clock,1e-6),self.parking[r].copy(),'W'))
                self.waits.append({'robot':r,'reason':'unused_robot_padding','start_s':0.,'duration_s':self.clock})
                self.robot_stats[r]['wait_time_s']+=rows[-1][0]
        return self.rows

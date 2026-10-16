"""Load-aware greedy assignment, nearest-neighbor routing, earliest reservation."""
import numpy as np
from .models import Action
from .profiling import timed


class GreedyParallelScheduler:
    @timed('greedy_assignment')
    def assign(self,tasks,engine):
        c=engine.config; p=c['process']
        queues={r:[] for r in engine.parking}
        ends={r:x.copy() for r,x in engine.parking.items()}
        load={r:0. for r in queues}
        # Longest tasks first prevent short-task nomination from starving robots.
        for task in sorted(tasks,key=lambda t:(-t.length,t.id)):
            choices=[]
            for r in task.robots:
                for reverse in (False,True):
                    pts=task.points[::-1] if reverse else task.points
                    extra=(engine.travel_length(ends[r],pts[0])+engine.travel_length(pts[-1],engine.parking[r])
                           -engine.travel_length(ends[r],engine.parking[r]))/p['travel_speed_mm_s']
                    cost=task.length/p['deposition_speed_mm_s']+(p.get('arc_on_time_s') or 0)+(p.get('arc_off_time_s') or 0)+extra
                    projected=dict(load); projected[r]+=cost
                    # Primary global load lower bound, secondary imbalance, no
                    # arbitrary weighted sum and no robot-ID quota.
                    choices.append((max(projected.values()),max(projected.values())-min(projected.values()),
                                    projected[r],r,reverse,cost))
            _,_,_,r,reverse,cost=min(choices)
            queues[r].append(task); load[r]+=cost
            ends[r]=task.points[0 if reverse else -1]
        return queues

    @timed('nearest_neighbor_ordering')
    def order(self,r,remaining,engine):
        ordered=[]; current=engine.parking[r]
        while remaining:
            choices=[(engine.travel_length(current,t.points[-1 if reverse else 0]),t.id,reverse,t)
                     for t in remaining for reverse in (False,True)]
            _,_,reverse,task=min(choices,key=lambda x:x[:3])
            ordered.append(Action(task,r,reverse)); current=task.points[0 if reverse else -1]
            remaining=[t for t in remaining if t.id!=task.id]
        return ordered

    @timed('parallel_reservation')
    def plan_layer(self,tasks,engine,release):
        queues=self.assign(tasks,engine)
        tours=[]
        for r,remaining in queues.items():
            ordered=self.order(r,remaining,engine)
            if ordered:
                tours.append(engine.build_tour(r,ordered))
        # Scheduling decisions use actual collision-resolved global makespan.
        while tours:
            candidates=[]
            for tour in tours:
                start=engine.earliest_start(tour,release)
                candidates.append((max(engine.clock,start+tour.duration),start,tour.robot,tour))
            _,start,_,tour=min(candidates,key=lambda x:x[:3])
            engine.commit(tour,release,start)
            tours=[x for x in tours if x is not tour]

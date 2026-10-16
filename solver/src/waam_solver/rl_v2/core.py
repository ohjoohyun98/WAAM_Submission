"""Complete plans, exact layer deltas and bounded tour/reservation caches.

Layer additivity requires every tour to end at parking and a global barrier.
It is NOT valid if a future implementation relaxes that barrier.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
import copy
import json
import pickle
import numpy as np
from ..models import Action
from ..parallel_engine import ParallelExecutionEngine
from ..reservations import forbidden_intervals
from ..rl.scenario_cache import digest, load_or_build_scenario, CACHE_SCHEMA


def load_scenario(source, root):
    """Read matching frozen cache only; all new cache writes go to v2."""
    source=Path(source).absolute()
    hashes={n:digest(source/n) for n in ('config.yaml','target.stl')}
    for meta in sorted((root/'checkpoints/rl_v1/cache').glob('*.json')):
        info=json.loads(meta.read_text(encoding='utf-8'))
        if info.get('source_sha256')==hashes and info.get('schema_version')==CACHE_SCHEMA:
            with meta.with_suffix('.pkl').open('rb') as stream: cache=pickle.load(stream)
            if cache.source_sha256==hashes: return cache
    return load_or_build_scenario(source,root/'checkpoints/rl_v2/cache')[0]


@dataclass(frozen=True)
class LayerPlan:
    # Ordered sub-tours. Same-robot sub-tours keep their specified order.
    tours: tuple
    priority: bool=False


def initial_plan(cache, decisions=None):
    decisions=cache.greedy_decisions if decisions is None else decisions
    result={}
    for layer in cache.layers:
        tours=[]
        for robot in (r['id'] for r in cache.config['robots']):
            queue=tuple((int(d['task']),bool(d['reverse'])) for d in decisions
                        if d['layer']==layer and d['robot']==robot)
            if queue: tours.append((robot,queue))
        result[layer]=LayerPlan(tuple(tours))
    return result


class BoundedCache:
    def __init__(self,limit): self.data=OrderedDict(); self.limit=limit; self.hits=0; self.misses=0
    def get(self,key,build):
        if key in self.data:
            self.hits+=1; self.data.move_to_end(key); return self.data[key]
        self.misses+=1; value=build(); self.data[key]=value
        if len(self.data)>self.limit: self.data.popitem(last=False)
        return value


class DeltaEngine(ParallelExecutionEngine):
    def __init__(self,owner):
        super().__init__(owner.cache.config,owner.cache.target_top_mm); self.owner=owner
    def forbidden(self,tour,existing,ready=None):
        # Relative forbidden intervals are translation invariant. Cache only
        # immutable action signatures, never Python object IDs.
        key=(signature(tour),signature(existing))
        def build():
            return forbidden_intervals(self,replace(tour,start=0.),replace(existing,start=0.),float('-inf'))
        relative=self.owner.pairs.get(key,build)
        bound=float('-inf') if ready is None else ready
        return [(a+existing.start,b+existing.start) for a,b in relative if b+existing.start>=bound]


def signature(tour): return (tour.robot,tuple((a.task.id,a.reverse) for a in tour.actions))


class Evaluator:
    def __init__(self,cache,use_cache=True):
        self.cache=cache; self.tasks={t.id:t for t in cache.tasks}; self.use_cache=use_cache
        self.geometry=ParallelExecutionEngine(cache.config,cache.target_top_mm)
        self.tours=BoundedCache(96); self.pairs=BoundedCache(192); self.layers=BoundedCache(256)

    def validate(self,layer,plan):
        expected={t.id for t in self.cache.tasks_for_layer(layer)}; seen=[]
        for robot,queue in plan.tours:
            if not queue: raise ValueError('Empty sub-tour')
            for task_id,rev in queue:
                if task_id not in expected or robot not in self.tasks[task_id].robots:
                    raise ValueError('Infeasible move mask')
                if type(rev) is not bool: raise ValueError('Direction must be bool')
                seen.append(task_id)
        if len(seen)!=len(expected) or set(seen)!=expected: raise ValueError('Tasks must occur exactly once')

    def travel(self,robot,queue):
        position=self.geometry.parking[robot]; distance=0.
        for tid,rev in queue:
            pts=self.tasks[tid].points[::-1] if rev else self.tasks[tid].points
            distance+=self.geometry.travel_length(position,pts[0]); position=pts[-1]
        return distance+self.geometry.travel_length(position,self.geometry.parking[robot])

    def duration(self,robot,queue):
        p=self.cache.config['process']
        return self.travel(robot,queue)/p['travel_speed_mm_s']+sum(self.tasks[t].length for t,_ in queue)/p['deposition_speed_mm_s']+len(queue)*((p.get('arc_on_time_s') or 0)+(p.get('arc_off_time_s') or 0))

    def commit_layer(self,engine,layer,plan,release):
        self.validate(layer,plan); pending=[]
        for robot,queue in plan.tours:
            def build(robot=robot,queue=queue):
                return engine.build_tour(robot,[Action(self.tasks[t],robot,d) for t,d in queue])
            tour=self.tours.get((robot,queue),build) if self.use_cache else build()
            pending.append(copy.copy(tour))
        while pending:
            # Cannot reorder the remaining sub-tours of one robot.
            ready=[t for i,t in enumerate(pending) if not any(x.robot==t.robot for x in pending[:i])]
            if plan.priority: tour=ready[0]; start=engine.earliest_start(tour,release)
            else:
                _,start,_,tour=min((max(engine.clock,(s:=engine.earliest_start(t,release))+t.duration),s,t.robot,t) for t in ready)
            engine.commit(tour,release,start); pending=[t for t in pending if t is not tour]

    def layer(self,layer,plan):
        def build():
            engine=DeltaEngine(self) if self.use_cache else ParallelExecutionEngine(self.cache.config,self.cache.target_top_mm)
            self.commit_layer(engine,layer,plan,0.)
            loads={r:0. for r in engine.parking}
            for tour in engine.tours: loads[tour.robot]+=tour.duration
            return {'duration':engine.clock,'base':max(loads.values()),'collision':engine.clock-max(loads.values()),
                    'loads':loads,'completion':{r:rows[-1][0] for r,rows in engine.rows.items()},
                    'travel':sum(self.travel(r,q) for r,q in plan.tours),
                    'waits':engine.waits,'tour_starts':[t.start for t in engine.tours],
                    'rows':engine.rows,'trace':engine.decisions}
        return self.layers.get((layer,plan),build) if self.use_cache else build()

    def full(self,plan):
        engine=DeltaEngine(self) if self.use_cache else ParallelExecutionEngine(self.cache.config,self.cache.target_top_mm)
        for layer in self.cache.layers: self.commit_layer(engine,layer,plan[layer],engine.clock)
        rows=engine.finish()
        return {'makespan_s':float(engine.clock),'rows':rows,'waits':engine.waits,'trace':engine.decisions,
                'robot_stats':engine.robot_stats}

    def stats(self):
        return {k:{'hits':v.hits,'misses':v.misses,'entries':len(v.data)} for k,v in
                [('layer',self.layers),('tour',self.tours),('reservation',self.pairs)]}

    def assemble(self,plan):
        """Reuse unchanged, hash-validated Greedy layer trajectories exactly.

        Changed layers use exact reservations. Parking and the global barrier
        make their translation independent. Full replay remains available for
        equivalence tests; this is not used with a relaxed barrier.
        """
        baseline=initial_plan(self.cache); rows={r:[(0.,p.copy(),'W')] for r,p in self.geometry.parking.items()}
        release=0.; original_release=0.; waits=[]; trace=[]; base_sum=0.; errors=[]
        for layer in self.cache.layers:
            current=plan[layer]
            if current==baseline[layer]:
                duration=self.cache.greedy_layer_durations_s[layer]; local={r:[(0.,p.copy(),'W')] for r,p in self.geometry.parking.items()}
                decisions=[d for d in self.cache.greedy_decisions if d['layer']==layer]
                for robot,queue in current.tours:
                    start=min(d['tour_start_s'] for d in decisions if d['robot']==robot)
                    end=start+self.duration(robot,queue)
                    segment=[(t-original_release,p,m) for t,p,m in self.cache.greedy_rows[robot] if start-1e-7<=t<=end+1e-7]
                    if not segment: raise ValueError('Frozen Greedy layer extraction failed')
                    segment[-1]=(segment[-1][0],segment[-1][1],'W')
                    if abs(segment[0][0])<1e-7:
                        segment[0]=(0.,segment[0][1],segment[0][2]); local[robot]=segment
                    else: local[robot].extend(segment)
                    errors.append(abs(segment[-1][0]-(end-original_release)))
                base=max((self.duration(r,q) for r,q in current.tours),default=0.)
                for wait in self.cache.greedy_waits:
                    if wait['reason']=='collision_reservation' and original_release-1e-7<=wait['start_s']<original_release+duration-1e-7:
                        waits.append({**wait,'start_s':release+wait['start_s']-original_release})
            else:
                result=self.layer(layer,current); duration=result['duration']; local=result['rows']; base=result['base']; decisions=result['trace']
                waits.extend({**w,'start_s':release+w['start_s']} for w in result['waits'] if w['reason']=='collision_reservation')
            base_sum+=base
            for robot,items in local.items():
                if len(items)<2: continue
                # The first item can be a parking wait before the first tour.
                shifted=[(release+t,p,m) for t,p,m in items]
                if shifted[0][0]>rows[robot][-1][0]+1e-9:
                    waits.append({'robot':robot,'reason':'layer_precedence','start_s':rows[robot][-1][0],
                                  'duration_s':shifted[0][0]-rows[robot][-1][0]})
                    rows[robot].extend(shifted)
                else: rows[robot][-1]=shifted[0]; rows[robot].extend(shifted[1:])
            for d in decisions:
                item=dict(d)
                item['tour_start_s']=release+d.get('tour_start_s',0.)-(original_release if current==baseline[layer] else 0.)
                trace.append(item)
            release+=duration; original_release+=self.cache.greedy_layer_durations_s[layer]
        stats={r:{'deposition_time_s':0.,'travel_time_s':0.,'wait_time_s':0.} for r in rows}
        for robot,items in rows.items():
            if len(items)<2: items.append((release,self.geometry.parking[robot].copy(),'W'))
            for a,b in zip(items[:-1],items[1:]): stats[robot][{'D':'deposition_time_s','T':'travel_time_s','W':'wait_time_s'}[a[2]]]+=b[0]-a[0]
        return {'makespan_s':release,'rows':rows,'trace':trace,'waits':waits,'robot_stats':stats,
                'critical_collision_s':release-base_sum,'noncollision_critical_s':base_sum,
                'frozen_extraction_max_error_s':max(errors,default=0.)}


def dump_plan(plan):
    return {str(k):{'tours':p.tours,'priority':p.priority} for k,p in plan.items()}


def read_plan(data):
    return {int(k):LayerPlan(tuple((int(r),tuple((int(t),bool(d)) for t,d in q)) for r,q in p['tours']),p['priority']) for k,p in data.items()}

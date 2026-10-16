"""Shared exact execution, barrier deltas, and experimental local-support DAG."""
from collections import defaultdict
import copy
import itertools
from time import perf_counter
import numpy as np
from shapely import LineString, STRtree
from ..greedy_parallel_scheduler import GreedyParallelScheduler
from ..rl_v2.core import Evaluator as ReferenceEvaluator, BoundedCache, LayerPlan
from .engine import Engine

class Evaluator(ReferenceEvaluator):
    def __init__(self,scenario,*,local_height=True,primitive_mm=10.,timeline=False,certified_timeline=False):
        self.cache=scenario; self.tasks={t.id:t for t in scenario.tasks}; self.use_cache=True
        self.options={'local_height':local_height,'primitive_mm':primitive_mm}
        from .timeline import TimelineEngine,CertifiedTimelineEngine
        self.engine_class=CertifiedTimelineEngine if certified_timeline else TimelineEngine if timeline else Engine
        self.geometry=self.engine_class(scenario.config,scenario.target_top_mm,**self.options)
        self.tours=BoundedCache(18); self.layers=BoundedCache(12)
        self.graphs=BoundedCache(4)
        self.exact_seconds=0.; self.evaluations=0

    def new_engine(self): return self.engine_class(self.cache.config,self.cache.target_top_mm,**self.options)

    def conflict_graph(self,layer):
        from .graphs import ConflictIndex
        return self.graphs.get(layer,lambda:ConflictIndex(self.cache.tasks_for_layer(layer),self.cache.config))

    def layer(self,layer,plan):
        def build():
            begin=perf_counter(); engine=self.new_engine()
            self.commit_layer(engine,layer,plan,0.)
            loads={r:sum(self.duration(r,q) for rr,q in plan.tours if rr==r) for r in engine.parking}
            result={'duration':engine.clock,'base':max(loads.values()),'collision':engine.clock-max(loads.values()),
                    'loads':loads,'completion':{r:rows[-1][0] for r,rows in engine.rows.items()},
                    'travel':sum(self.travel(r,q) for r,q in plan.tours),'waits':engine.waits,
                    'rows':engine.rows,'trace':engine.decisions,'query_pairs':engine.query_pairs}
            self.exact_seconds+=perf_counter()-begin; self.evaluations+=1
            return result
        return self.layers.get((layer,plan),build)

    def full(self,plan):
        engine=self.new_engine()
        for layer in self.cache.layers: self.commit_layer(engine,layer,plan[layer],engine.clock)
        return {'makespan_s':engine.clock,'rows':engine.finish(),'waits':engine.waits,'trace':engine.decisions}

def greedy_plan(scenario,evaluator):
    policy=GreedyParallelScheduler(); plan={}
    for layer in scenario.layers:
        queues=policy.assign(scenario.tasks_for_layer(layer),evaluator.geometry)
        tours=[]
        for r,q in queues.items():
            actions=policy.order(r,q,evaluator.geometry)
            if actions: tours.append((r,tuple((a.task.id,a.reverse) for a in actions)))
        plan[layer]=LayerPlan(tuple(tours))
    return plan

def assemble(scenario,evaluator,results):
    rows={r:[(0.,p.copy(),'W')] for r,p in evaluator.geometry.parking.items()}
    release=0.; waits=[]; trace=[]; critical_collision=0.; base=0.
    for layer in scenario.layers:
        result=results[layer]
        for robot,items in result['rows'].items():
            if len(items)<2: continue
            shifted=[(release+t,p,m) for t,p,m in items]
            previous=rows[robot][-1][0]
            if release>previous+1e-9:
                waits.append({'robot':robot,'reason':'layer_precedence','start_s':previous,'duration_s':release-previous})
                rows[robot].extend(shifted)
            else:
                rows[robot][-1]=shifted[0]; rows[robot].extend(shifted[1:])
        waits.extend({**w,'start_s':release+w['start_s']} for w in result['waits'])
        trace.extend({**d,'tour_start_s':release+d['tour_start_s']} for d in result['trace'])
        release+=result['duration']; critical_collision+=result['collision']; base+=result['base']
    for robot,items in rows.items():
        if len(items)<2:
            items.append((release,evaluator.geometry.parking[robot].copy(),'W'))
            waits.append({'robot':robot,'reason':'unused_robot_padding','start_s':0.,'duration_s':release})
    return {'makespan_s':release,'rows':rows,'waits':waits,'trace':trace,
            'critical_collision_s':critical_collision,'collision_free_barrier_s':base}

def partition(plan,size):
    return {k:LayerPlan(tuple((r,q[i:i+size]) for r,q in p.tours for i in range(0,len(q),size)),p.priority)
            for k,p in plan.items()}

def reduced_priorities(plan):
    """Exact reduced integer optimization: enumerate <= 6 tour permutations.

    Equivalent finite priority variable search; not marketed as CP-SAT (the
    optional OR-Tools package is absent). Original queues stay unchanged.
    """
    if len(plan.tours)>6: return []
    return [LayerPlan(tuple(tours),True) for tours in itertools.permutations(plan.tours)]

def support_schedule(scenario,plan):
    """Experimental bead-overlap support DAG. Not a thermal/process certificate.

    Every intersecting prior-layer bead is a predecessor, including boundary
    contacts. Missing support falls back to the entire previous active layer.
    Global target-height travel is mandatory while layers overlap.
    """
    evaluator=Evaluator(scenario,local_height=False)
    nodes=[]; owner={}; by_layer={}; dependencies={}; support_edges=0; fallback=0
    for layer in scenario.layers:
        indices=[]
        for r,q in plan[layer].tours:
            node=len(nodes); nodes.append((layer,r,q)); indices.append(node)
            for tid,_ in q: owner[tid]=node
        by_layer[layer]=indices
    previous=None
    radius=scenario.config['process']['bead_width_mm']
    for layer in scenario.layers:
        tasks=scenario.tasks_for_layer(layer)
        if previous is None:
            for node in by_layer[layer]: dependencies[node]=set()
        else:
            lower=scenario.tasks_for_layer(previous)
            tree=STRtree([LineString(t.points[:,:2]) for t in lower])
            for node in by_layer[layer]: dependencies[node]=set()
            for task in tasks:
                hits=tree.query(LineString(task.points[:,:2]),predicate='dwithin',distance=radius+.001)
                deps={owner[lower[i].id] for i in hits}
                if not deps:
                    deps=set(by_layer[previous]); fallback+=1
                dependencies[owner[task.id]].update(deps); support_edges+=len(hits)
        previous=layer
    # Same-robot task/layer order is retained. This guarantees an acyclic DAG
    # and prevents depositing an upper task before the robot's earlier layer.
    last={}
    for n,(_,r,_) in enumerate(nodes):
        if r in last: dependencies[n].add(last[r])
        last[r]=n
    successors=defaultdict(list); indegree={n:len(d) for n,d in dependencies.items()}
    for n,deps in dependencies.items():
        for d in deps: successors[d].append(n)
    ready={n for n,d in indegree.items() if not d}; completion={}; engine=evaluator.new_engine()
    while ready:
        candidates=[]
        shortlist=sorted(ready,key=lambda n:(max((completion[d] for d in dependencies[n]),default=0.),nodes[n][0],n))[:3]
        for n in shortlist:
            layer,r,q=nodes[n]; release=max((completion[d] for d in dependencies[n]),default=0.)
            from ..models import Action
            tour=engine.build_tour(r,[Action(evaluator.tasks[t],r,d) for t,d in q])
            start=engine.earliest_start(tour,release)
            candidates.append((start+tour.duration,start,n,release,tour))
        end,start,n,release,tour=min(candidates,key=lambda x:x[:3])
        engine.commit(tour,release,start); completion[n]=end; ready.remove(n)
        for other in successors[n]:
            indegree[other]-=1
            if indegree[other]==0: ready.add(other)
    if len(completion)!=len(nodes): raise ValueError('SUPPORT_DAG_CYCLE')
    proc=scenario.config['process']; arc=(proc.get('arc_on_time_s') or 0)+(proc.get('arc_off_time_s') or 0)
    bound={}
    for n,(_,r,q) in enumerate(nodes):
        cost=sum(evaluator.tasks[t].length/proc['deposition_speed_mm_s']+arc for t,_ in q)
        bound[n]=cost+max((bound[d] for d in dependencies[n]),default=0.)
    for wait in engine.waits:
        if wait['reason']=='layer_precedence': wait['reason']='local_support_or_robot_precedence'
    return {'makespan_s':engine.clock,'rows':engine.finish(),'waits':engine.waits,'trace':engine.decisions,
            'support_edges':support_edges,'dependency_edges':sum(map(len,dependencies.values())),
            'fixed_assignment_support_precedence_lower_bound_s':max(bound.values(),default=0.),
            'missing_support_fallback_tasks':fallback,'process_scope':'bead overlap ordering only; thermal/overhang physics unmodeled'}

"""Spatially coherent assignment, representative screening, exact layer deltas."""
import numpy as np
from time import perf_counter
from ..competition_v3.scheduling import Evaluator as BaseEvaluator,greedy_plan as old_greedy,reduced_priorities
from ..greedy_parallel_scheduler import GreedyParallelScheduler
from ..rl_v2.core import LayerPlan
from ..competition_v3.search import optimize as old_optimize

AUDIT={}
SPATIAL=True
CACHE_CERTIFICATES=True
class Evaluator(BaseEvaluator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if CACHE_CERTIFICATES and kwargs.get('certified_timeline',False):
            from .engine import CertifiedEngine
            self.engine_class=CertifiedEngine;self.geometry=self.new_engine()
def signature(tasks):
    # Index for transfer of a POLICY, never for reuse of safety results.
    # Each chosen queue is rebuilt and exactly collision-evaluated at actual Z.
    return tuple((t.robots,round(t.length,1),tuple(np.round(t.points[:,:2].mean(axis=0),1))) for t in tasks)

def spatial_plan(tasks,evaluator,reverse=False):
    queues={r:[] for r in evaluator.geometry.parking};bases={r['id']:np.asarray(r['base_xyz_mm'][:2]) for r in evaluator.cache.config['robots']}
    for task in tasks:
        center=task.points[:,:2].mean(axis=0)
        robot=min(task.robots,key=lambda r:(np.linalg.norm(center-bases[r]),r))
        queues[robot].append(task)
    policy=GreedyParallelScheduler();tours=[]
    for robot,q in queues.items():
        actions=policy.order(robot,q,evaluator.geometry)
        queue=tuple((a.task.id,a.reverse) for a in actions)
        if reverse:queue=tuple((tid,not rev) for tid,rev in reversed(queue))
        if queue:tours.append((robot,queue))
    return LayerPlan(tuple(tours))

def greedy_plan(scenario,evaluator):
    start=perf_counter();baseline=old_greedy(scenario,evaluator);selected={};policies={};records=[]
    if not SPATIAL:return baseline
    for layer in scenario.layers:
        tasks=scenario.tasks_for_layer(layer);key=signature(tasks)
        if key not in policies:
            if len(policies)>=4 or perf_counter()-start>12.:
                # A bounded pilot chooses a general assignment POLICY only;
                # execution at every later layer still receives exact replay.
                totals={name:sum(row['options'][name] for row in records) for name in ('load','spatial','spatial_reverse')}
                label=min(totals,key=totals.get) if records else 'load'
                selected[layer]=baseline[layer] if label=='load' else spatial_plan(tasks,evaluator,label=='spatial_reverse')
                continue
            options=[('load',baseline[layer]),('spatial',spatial_plan(tasks,evaluator)),('spatial_reverse',spatial_plan(tasks,evaluator,True))]
            # Small priority problem: no giant MILP and no changed collision rules.
            measured=[]
            for label,plan in options:
                result=evaluator.layer(layer,plan);measured.append((result['duration'],label,plan))
            _,label,plan=min(measured,key=lambda x:x[0]);policies[key]=label
            records.append({'layer':layer,'options':{label:d for d,label,_ in measured},'selected':label})
        label=policies[key]
        selected[layer]=baseline[layer] if label=='load' else spatial_plan(tasks,evaluator,label=='spatial_reverse')
    AUDIT.clear();AUDIT.update(assignment_screening=records,signature_groups=len(policies),screening_s=perf_counter()-start)
    return selected

def optimize(evaluator,plan,results,*,algorithm,budget,seed=11,max_steps=None,ranker=None):
    start=perf_counter()
    if all(len(evaluator.cache.tasks_for_layer(k))==1 for k in plan):
        # With mandatory global barriers and one fixed task per layer, all
        # other robots remain parked. Enumerating robot/direction is exact;
        # once the incumbent attains that bound, LNS cannot improve it.
        proved=True
        for k in plan:
            task=evaluator.cache.tasks_for_layer(k)[0]
            optimum=min(evaluator.duration(r,((task.id,reverse),)) for r in task.robots for reverse in (False,True))
            if results[k]['duration']>optimum+1e-8:proved=False;break
        if proved:
            return dict(plan),dict(results),{'algorithm':'exact_single_task_certificate','steps':0,'records':[],
                'accepted':[],'exact_candidates':0,'errors':[],'neural_seconds':0.,'optimization_seconds':perf_counter()-start,
                'scope':'Optimal robot/direction for fixed single-task layers with mandatory global barriers; not a geometry or barrier-free optimum.'}
    # Exact affected-layer replay already makes all other layers immutable.
    # Critical LNS is inherited; lower cold budget buys structural screening.
    return old_optimize(evaluator,plan,results,algorithm=algorithm,budget=min(budget,45.),seed=seed,max_steps=max_steps,ranker=ranker)

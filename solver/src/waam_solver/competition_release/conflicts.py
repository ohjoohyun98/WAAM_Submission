"""Sparse reservation-decision graph and bounded critical-block beam search.

Edges mean reservation priority, not whole-tour non-overlap. Exact continuous
capsule replay decides the allowable lag/overlap after every ordering change.
Mandatory robot queue order and the global layer barrier are never relaxed.
"""
import itertools,copy
from time import perf_counter
from collections import defaultdict
import numpy as np
from ..rl_v2.core import LayerPlan
from ..models import Action
from ..competition_v3.timeline import pair_safe

def graph(evaluator,layer,plan,result):
    starts={x['task']:x['tour_start_s'] for x in result['trace']};order={x['task']:i for i,x in enumerate(result['trace'])}
    nodes=[];engine=evaluator.new_engine()
    for i,(r,q) in enumerate(plan.tours):
        t=engine.build_tour(r,[Action(evaluator.tasks[tid],r,rev) for tid,rev in q]);t.start=starts[q[0][0]]
        nodes.append({'id':i,'robot':r,'start':t.start,'end':t.start+t.duration,'tour':t,'order':order[q[0][0]],'slack':result['duration']-t.start-t.duration})
    edges=[];last={};ordered=sorted(nodes,key=lambda n:n['order'])
    for n in ordered:
        prior=last.get(n['robot']);ready=prior['end'] if prior else 0.
        if prior:edges.append({'from':prior['id'],'to':n['id'],'kind':'robot_sequence'})
        for other in ordered:
            if other['order']>=n['order']:break
            if other['robot']==n['robot'] or other['end']<ready:continue
            safe,skip=pair_safe(evaluator.cache.config,n['tour'],other['tour'],ready,return_skip=True)
            if not safe:
                edges.append({'from':other['id'],'to':n['id'],'kind':'reservation_priority','witness_advance':skip,
                    'witness_confirmed':skip>0,'delay':max(0.,n['start']-ready),'critical':n['slack']<max(1.,result['duration']*.03)})
        last[n['robot']]=n
    return {'layer':layer,'nodes':[{k:v for k,v in n.items() if k!='tour'} for n in nodes],'edges':edges,
        'layer_precedence':'global barrier before and after this graph'}

def neighbors(evaluator,layer,plan,result):
    g=graph(evaluator,layer,plan,result);critical={n['id'] for n in g['nodes'] if n['slack']<max(1.,result['duration']*.03)}
    tours=plan.tours;candidates=[]
    # Adjacent and critical-block flips, plus small exact order enumeration.
    if len(tours)<=4:
        for perm in itertools.permutations(range(len(tours))):
            p=LayerPlan(tuple(tours[i] for i in perm),True)
            if p!=plan:candidates.append((p,'priority'))
    for i,(r,q) in enumerate(tours):
        if i not in critical or len(q)<2:continue
        variants=[q[1:]+q[:1],q[-1:]+q[:-1],(q[-1],)+q[1:-1]+(q[0],),tuple((tid,not rev) for tid,rev in reversed(q))]
        for queue in variants:
            p=list(tours);p[i]=(r,queue);candidates.append((LayerPlan(tuple(p),plan.priority),'critical_block'))
    seen={plan};ranked=[]
    for p,kind in candidates:
        if p in seen:continue
        seen.add(p);loads=defaultdict(float)
        for r,q in p.tours:loads[r]+=evaluator.duration(r,q)
        lower=max(loads.values())
        if lower>=result['duration']-1e-8:continue
        ranked.append((lower,p,kind))
    ranked.sort(key=lambda x:x[0]);return ranked,g

def optimize(evaluator,plan,results,budget=20.,width=3,depth=3,max_evaluations=48):
    start=perf_counter();best=dict(plan);out=dict(results);accepted=[];graphs=[];tested=0;errors=[]
    layers=sorted(plan,key=lambda k:-(results[k]['collision']+.02*results[k]['duration']))
    for k in layers:
        if tested>=max_evaluations:break
        if out[k]['collision']/max(out[k]['duration'],1.)<.08:continue
        beam=[(best[k],out[k])];seen={best[k]}
        for level in range(depth):
            pool=[]
            for p,result in beam:
                if tested>=max_evaluations:break
                candidates,g=neighbors(evaluator,k,p,result)
                if len(graphs)<8:graphs.append(g)
                for _,candidate,kind in candidates[:8]:
                    if candidate in seen or tested>=max_evaluations:continue
                    seen.add(candidate)
                    tested+=1
                    try:exact=evaluator.layer(k,candidate)
                    except ValueError as exc:
                        errors.append(str(exc));continue
                    pool.append((candidate,exact))
                    if exact['duration']<out[k]['duration']-1e-8:
                        accepted.append({'layer':k,'kind':kind,'depth':level,'gain_s':out[k]['duration']-exact['duration']})
                        best[k]=candidate;out[k]=exact
            if not pool:break
            pool.sort(key=lambda x:x[1]['duration']);beam=pool[:width]
    return best,out,{'algorithm':'critical_reservation_beam','accepted':accepted,'exact_candidates':tested,'graphs':graphs,
        'optimization_seconds':perf_counter()-start,'records':[],'neural_seconds':0.,'errors':errors,
        'delta_scope':'Exact affected layer only; all downstream global barriers shift additively. No cross-layer overlap.'}

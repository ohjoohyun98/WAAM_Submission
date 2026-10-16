"""Budgeted critical-layer portfolio; all accepted moves use exact evaluation."""
import math
import random
from time import perf_counter
from ..rl_v2.moves import propose, cheap_features, insert, KINDS, FEATURE_NAMES as V2_FEATURE_NAMES
from ..rl_v2.core import LayerPlan
from ..competition_v3.scheduling import reduced_priorities

# Keep the frozen v2 builder as an implementation helper, but expose the
# actual v3 meanings. The constant critical flag is replaced by scale/OOD
# context; no v2 checkpoint is silently loaded into this feature contract.
FEATURE_NAMES=list(V2_FEATURE_NAMES[:28])
FEATURE_NAMES[12]='sampled_assigned_arm_tcp_conflict_density'
FEATURE_NAMES[18]='log_layer_task_count'
FEATURE_NAMES[19]='log_total_layer_count'
FEATURE_NAMES=tuple(FEATURE_NAMES)

def ruin_repair(evaluator,layer,plan,rng):
    tours=[(r,list(q)) for r,q in plan.tours]
    flat=[(i,item) for i,(_,q) in enumerate(tours) for item in q]
    if len(flat)<2: return None
    removed=rng.sample(flat,min(len(flat),max(2,int(math.sqrt(len(flat))))))
    for i,item in removed: tours[i][1].remove(item)
    for _,item in sorted(removed,key=lambda x:-evaluator.tasks[x[1][0]].length):
        choices=[]
        for robot in evaluator.tasks[item[0]].robots:
            idx=next((i for i,(r,q) in enumerate(tours) if r==robot),None)
            q=list(tours[idx][1]) if idx is not None else []
            insert(evaluator,robot,q,item)
            load=evaluator.duration(robot,q)
            other=max((evaluator.duration(r,x) for r,x in tours if r!=robot),default=0.)
            choices.append((max(load,other),load,robot,idx,q))
        _,_,r,idx,q=min(choices,key=lambda x:x[:3])
        if idx is None: tours.append((r,q))
        else: tours[idx]=(r,q)
    return LayerPlan(tuple((r,tuple(q)) for r,q in tours if q),plan.priority)

def optimize(evaluator,plan,results,*,algorithm,budget,seed=11,max_steps=None,ranker=None):
    rng=random.Random(seed); begin=perf_counter(); records=[]; accepted=[]; evaluations=0
    best=dict(plan); best_results=dict(results); current=dict(plan); current_results=dict(results)
    neural_seconds=0.; errors=[]; step=0
    initial=sum(x['duration'] for x in results.values())
    while step < (max_steps if max_steps is not None else max(1,int(budget))):
        layers=list(best)
        # Critical layer extension dominates; uniform exploration prevents
        # starvation. No public identity or known dimensions enter selection.
        weights=[math.sqrt(max(best_results[k]['collision'],.01)+.05*best_results[k]['duration']) for k in layers]
        layer=rng.choices(layers,weights=weights,k=1)[0] if step%5 else rng.choice(layers)
        before=current[layer] if algorithm=='sa' else best[layer]
        base=current_results[layer] if algorithm=='sa' else best_results[layer]
        candidates=[]; seen={before}; kindlist=[]
        chosen_kind=('priority','relocate','reinsert','reverse','swap','cluster','split')[step%7]
        if algorithm=='reduced' or (algorithm=='portfolio' and step%7==0):
            candidates=[(p,'priority') for p in reduced_priorities(before) if p!=before]
        else:
            for attempt in range(12):
                if False: break  # Wall clock is telemetry only; bounded outer steps.
                kind=chosen_kind if algorithm=='critical' else rng.choice(KINDS)
                if algorithm in ('lns','portfolio','learned','hybrid','ppo') and attempt%3==0:
                    candidate=ruin_repair(evaluator,layer,before,rng); kind='cluster'
                else: candidate=propose(evaluator,layer,before,rng,kind,critical=attempt%4!=0,analysis=base)
                if candidate is not None and candidate not in seen:
                    seen.add(candidate); candidates.append((candidate,kind))
        featured=[]
        for candidate,kind in candidates:
            if False: break  # Wall clock is telemetry only; bounded outer steps.
            f=cheap_features(evaluator,layer,before,candidate,kind,initial,True,base=base)
            f[12]=evaluator.conflict_graph(layer).assigned_density(candidate)
            f[13]=f[12]*base['collision']/max(1.,base['duration'])
            f[18]=math.log1p(len(evaluator.cache.tasks_for_layer(layer)))
            f[19]=math.log1p(len(evaluator.cache.layers))
            featured.append((candidate,kind,f))
        if ranker is not None and algorithm in ('learned','hybrid','ppo','portfolio') and featured:
            t=perf_counter(); scores,ood=ranker.score([x[2] for x in featured]); neural_seconds+=perf_counter()-t
            if not ood: featured=[x for _,x in sorted(zip(scores,featured),key=lambda x:-x[0])]
            else: featured.sort(key=lambda x:x[2][9]+x[2][10],reverse=True)
        else: featured.sort(key=lambda x:x[2][9]+x[2][10],reverse=True)
        # Bounded exact screening. Reduced optimization enumerates all small
        # permutations; others retain a random exploratory candidate.
        selected=featured if algorithm=='reduced' else featured[:2]+(rng.sample(featured[2:],1) if len(featured)>2 else [])
        best_candidate=None
        for candidate,kind,f in selected:
            if False: break  # Wall clock is telemetry only; bounded outer steps.
            evaluations+=1
            try: exact=evaluator.layer(layer,candidate)
            except ValueError as exc:
                errors.append(str(exc)); continue
            gain=(base['duration']-exact['duration'])/initial
            records.append({'features':f,'relative_gain':gain,'layer':layer,'kind':kind,'context':step})
            if best_candidate is None or exact['duration']<best_candidate[1]['duration']:
                best_candidate=(candidate,exact)
        if best_candidate:
            candidate,exact=best_candidate
            if exact['duration']<best_results[layer]['duration']-1e-8:
                accepted.append({'step':step,'layer':layer,'gain_s':best_results[layer]['duration']-exact['duration']})
                best[layer]=candidate; best_results[layer]=exact
            if algorithm=='sa':
                delta=exact['duration']-base['duration']; temperature=max(.01,.01*base['duration']*(1-step/max(1,max_steps if max_steps is not None else int(budget))))
                if delta<=0 or rng.random()<math.exp(-min(700.,delta/temperature)):
                    current[layer]=candidate; current_results[layer]=exact
        step+=1
    return best,best_results,{'algorithm':algorithm,'steps':step,'exact_candidates':evaluations,'accepted':accepted,
                            'records':records,'errors':errors,'neural_seconds':neural_seconds,'optimization_seconds':perf_counter()-begin}

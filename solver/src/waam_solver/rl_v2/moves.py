"""Geometry hierarchy and critical/exploratory complete-plan moves."""
import random
import numpy as np
from .core import LayerPlan

KINDS=('relocate','swap','reinsert','reverse','split','merge','cluster','priority','critical')
FEATURES=32
FEATURE_NAMES=tuple('action_'+k for k in KINDS)+(
    'safe_z_travel_delta','critical_load_delta','current_collision_extension','bbox_conflict_degree',
    'cheap_future_extension','barrier_blocked_robot_seconds','barrier_blocked_robot_count',
    'subtour_count','layer_task_count','total_layer_count','critical_candidate','layer_fraction_of_global',
    'candidate_load_R1','candidate_load_R2','candidate_load_R3','slack_R1','slack_R2','slack_R3',
    'mean_feasible_robot_fraction','exact_relative_gain_percent','exact_collision_delta',
    'exact_candidate_duration_ratio','exact_candidate_extension_ratio')


def clusters(evaluator,layer,max_size=6):
    tasks=evaluator.cache.tasks_for_layer(layer); remaining={t.id for t in tasks}; result=[]
    centers={t.id:t.points[:,:2].mean(axis=0) for t in tasks}
    while remaining:
        first=min(remaining); signature=evaluator.tasks[first].robots
        near=sorted((t for t in remaining if evaluator.tasks[t].robots==signature),
                    key=lambda t:(float(np.linalg.norm(centers[t]-centers[first])),t))[:max_size]
        result.append(tuple(near)); remaining.difference_update(near)
    return tuple(result)


def insert(evaluator,robot,queue,item,heuristic=True):
    directions=(False,True) if heuristic else (item[1],)
    ends=[]
    for tid,rev in queue:
        points=evaluator.tasks[tid].points
        ends.append((points[-1],points[0]) if rev else (points[0],points[-1]))
    park=evaluator.geometry.parking[robot]; points=evaluator.tasks[item[0]].points
    choices=[]
    for i in range(len(queue)+1):
        a=park if i==0 else ends[i-1][1]; b=park if i==len(queue) else ends[i][0]
        direct=evaluator.geometry.travel_length(a,b)
        for d in directions:
            first,last=(points[-1],points[0]) if d else (points[0],points[-1])
            delta=evaluator.geometry.travel_length(a,first)+evaluator.geometry.travel_length(last,b)-direct
            choices.append((delta,i,d))
    _,index,rev=min(choices)
    queue.insert(index,(item[0],rev))


def propose(evaluator,layer,plan,rng,kind,critical=True,direction='heuristic',analysis=None):
    tours=[(r,list(q)) for r,q in plan.tours]
    durations=[evaluator.duration(r,q) for r,q in tours]
    completion=(analysis if analysis is not None else evaluator.layer(layer,plan))['completion']
    a=max(range(len(tours)),key=lambda i:(completion[tours[i][0]],durations[i])) if critical else rng.randrange(len(tours))
    robot,queue=tours[a]; idx=rng.randrange(len(queue)); item=queue[idx]
    heuristic=direction=='heuristic'
    if kind=='priority':
        rng.shuffle(tours)
        return LayerPlan(tuple((r,tuple(q)) for r,q in tours),True)
    if kind=='split':
        if len(queue)<2: return None
        cut=rng.randrange(1,len(queue)); tours[a]=(robot,queue[:cut]); tours.insert(a+1,(robot,queue[cut:]))
    elif kind=='merge':
        peers=[i for i,(r,q) in enumerate(tours) if r==robot and i!=a]
        if not peers: return None
        b=rng.choice(peers); lo,hi=sorted((a,b)); tours[lo]=(robot,tours[lo][1]+tours[hi][1]); tours.pop(hi)
    elif kind=='reverse': queue[idx]=(item[0],not item[1])
    elif kind=='reinsert': queue.pop(idx); insert(evaluator,robot,queue,item,heuristic)
    elif kind in ('relocate','critical','cluster'):
        items=[item]
        if kind=='cluster':
            cluster=next(c for c in clusters(evaluator,layer) if item[0] in c)
            items=[x for x in queue if x[0] in cluster]
        feasible=set.intersection(*(set(evaluator.tasks[x[0]].robots) for x in items))-{robot}
        if not feasible: return None
        target=rng.choice(sorted(feasible))
        for x in items: queue.remove(x)
        peers=[i for i,(r,q) in enumerate(tours) if r==target]
        if peers: target_queue=tours[rng.choice(peers)][1]
        else: target_queue=[]; tours.append((target,target_queue))
        for x in items: insert(evaluator,target,target_queue,x,heuristic)
    elif kind=='swap':
        peers=[i for i,(r,q) in enumerate(tours) if r!=robot and q]
        if not peers: return None
        b=rng.choice(peers); other,queue_b=tours[b]; j=rng.randrange(len(queue_b)); second=queue_b[j]
        if other not in evaluator.tasks[item[0]].robots or robot not in evaluator.tasks[second[0]].robots: return None
        queue[idx],queue_b[j]=second,item
    else: raise ValueError(kind)
    result=LayerPlan(tuple((r,tuple(q)) for r,q in tours if q),plan.priority)
    evaluator.validate(layer,result)
    return result if result!=plan else None


def cheap_features(evaluator,layer,before,after,kind,initial_m,critical,base=None):
    base=evaluator.layer(layer,before) if base is None else base; scale=max(base['duration'],1.)
    loads={r:0. for r in evaluator.geometry.parking}
    for r,q in after.tours: loads[r]+=evaluator.duration(r,q)
    travel=sum(evaluator.travel(r,q) for r,q in after.tours)
    speed=evaluator.cache.config['process']['travel_speed_mm_s']
    # Candidate-specific spatial graph (arm hull envelope, not just TCP bbox).
    boxes=[]
    for r,q in after.tours:
        pts=np.concatenate([evaluator.tasks[t].points[:,:2] for t,_ in q]+[evaluator.geometry.parking[r][None,:2]])
        boxes.append((r,pts.min(axis=0),pts.max(axis=0)))
    pairs=[(a,b) for i,a in enumerate(boxes) for b in boxes[i+1:] if a[0]!=b[0]]
    conflicts=sum(bool(np.all(a[1]<=b[2]) and np.all(b[1]<=a[2])) for a,b in pairs)
    degree=conflicts/max(1,len(pairs))
    future=degree*base['collision'] # Explicit cheap heuristic; exact preview replaces this.
    features=[float(kind==k) for k in KINDS]
    features += [(base['travel']-travel)/speed/scale,(base['base']-max(loads.values()))/scale,
                 base['collision']/scale,degree,future/scale,
                 sum(base['duration']-v for v in base['completion'].values())/scale,
                 sum(v<base['duration']-1e-6 for v in base['completion'].values())/3.,
                 len(after.tours)/6.,len(evaluator.cache.tasks_for_layer(layer))/100.,
                 len(evaluator.cache.layers)/350.,float(critical),base['duration']/initial_m,
                 *[v/scale for v in loads.values()],
                 *[(base['duration']-v)/scale for v in base['completion'].values()],
                 sum(len(evaluator.tasks[t].robots)/3 for r,q in after.tours for t,d in q)/max(1,sum(len(q) for r,q in after.tours)),
                 0.,0.,0.,0.]
    assert len(features)==FEATURES
    return features


def candidate_pool(evaluator,layer,plan,rng,initial_m,pool_size=20,top_k=4,kinds=KINDS,direction='heuristic',critical_fraction=.8):
    seen={plan}; pool=[]; old=evaluator.layer(layer,plan)
    for attempt in range(pool_size*8):
        if len(pool)>=pool_size: break
        critical=(len(pool)%10)<round(10*critical_fraction)
        kind=rng.choice(kinds)
        changed=propose(evaluator,layer,plan,rng,kind,critical,direction,analysis=old)
        if changed is None or changed in seen: continue
        seen.add(changed)
        f=cheap_features(evaluator,layer,plan,changed,kind,initial_m,critical,base=old)
        pool.append({'plan':changed,'kind':kind,'critical':critical,'features':f})
    # Preserve exploration in exact top-K too. Never score model IDs.
    pool.sort(key=lambda c:c['features'][9]+c['features'][10]-.1*c['features'][13],reverse=True)
    if top_k<1: raise ValueError('top_k must be positive')
    chosen=pool[:max(1,top_k-1)]
    remaining=pool[len(chosen):]
    if remaining and top_k>1: chosen.append(rng.choice(remaining))
    result=[{'plan':plan,'kind':'noop','features':[0.]*FEATURES,'reward':0.,'duration':old['duration']}]
    for c in chosen:
        exact=evaluator.layer(layer,c['plan']); gain=(old['duration']-exact['duration'])/initial_m
        c.update(reward=max(0.,gain),raw_gain=gain,duration=exact['duration'])
        # Exact lookahead is intentionally visible. Its benefit is separated
        # from the neural selector by an exact-argmax search control.
        c['features'][-4:]=[gain*100.,(old['collision']-exact['collision'])/max(1.,old['duration']),
                             exact['duration']/max(1.,old['duration']),exact['collision']/max(1.,old['duration'])]
        result.append(c)
    return result

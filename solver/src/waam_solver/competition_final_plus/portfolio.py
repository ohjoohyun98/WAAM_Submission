"""Exact layer-wise geometry portfolio under unchanged global barriers."""
from ..models import Task
from ..rl_v2.core import LayerPlan
from ..competition_v3.scenario import Scenario
from ..competition_v3.scheduling import assemble
from ..competition_final.scheduling import Evaluator
from ..shape_validation import aggregate

def layerwise(reference,alternative):
    original=reference['scenario'];tasks=[];by={};scores={};plan={};results={};selected=[]
    for k in original.layers:
        take=k in alternative['results'] and alternative['results'][k]['duration']<reference['results'][k]['duration']-1e-8
        source=alternative if take else reference
        if take:selected.append(k)
        local=[];mapping={}
        for task in source['scenario'].by_layer[k]:
            mapping[task.id]=len(tasks);new=Task(len(tasks),k,task.points,task.robots,task.residual_fill);tasks.append(new);local.append(new)
        by[k]=local;scores[k]=source['scenario'].scores[k];old=source['plan'][k]
        plan[k]=LayerPlan(tuple((r,tuple((mapping[tid],reverse) for tid,reverse in q)) for r,q in old.tours),old.priority)
        results[k]={**source['results'][k],'trace':[{**x,'task':mapping[x['task']]} for x in source['results'][k]['trace']]}
    if not selected:return reference,[]
    gate=aggregate(scores,original.config);before=aggregate(original.scores,original.config)
    floor=min(before['iou'],original.config['shape_validation']['minimum_overall_iou']+.005)
    if not gate['passed'] or gate['iou']<floor-1e-8:return reference,[]
    scenario=Scenario(original.config,original.target_top_mm,tuple(tasks),original.layers,original.layer_geometry,by,scores,
        original.geometry_seconds+alternative['scenario'].geometry_seconds)
    evaluator=Evaluator(scenario,certified_timeline=True)
    return {'scenario':scenario,'evaluator':evaluator,'plan':plan,'results':results,
        'search':{'algorithm':'exact_layerwise_geometry_portfolio','selected_geometry_layers':selected,'optimization_seconds':0.},
        'trajectory':assemble(scenario,evaluator,results)},selected

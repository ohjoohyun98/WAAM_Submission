"""Isolated Core: existing pruning/contiguous merge, exact affected layers."""
from collections import defaultdict
import copy
from shapely import normalize
from ..models import Task
from ..tasks import generate_tasks
from ..shape_validation import aggregate
from ..competition_v3.scenario import Scenario
from ..competition_v3.scheduling import assemble
from ..competition_final_plus.geometry import merge_contiguous
from ..competition_final.geometry import cost
from ..competition_final_plus.portfolio import layerwise
from . import scheduling, conflicts, pruning
from .common import clone_scenario


def schedule(scenario, steps, seed):
    evaluator=scheduling.Evaluator(scenario,certified_timeline=True)
    plan=scheduling.greedy_plan(scenario,evaluator)
    assignment_policy=copy.deepcopy(scheduling.AUDIT)
    results={k:evaluator.layer(k,plan[k]) for k in scenario.layers}
    assignment_evaluations=evaluator.evaluations
    info={'steps':0,'exact_candidates':0}
    if steps:
        plan,results,info=scheduling.optimize(evaluator,plan,results,algorithm='lns',
                                             budget=steps,max_steps=steps,seed=seed)
    info={**info,'assignment_exact_evaluations':assignment_evaluations,'total_layer_evaluations':evaluator.evaluations,
          'assignment_policy':assignment_policy}
    return {'scenario':scenario,'evaluator':evaluator,'plan':plan,'results':results,
            'search':info,'trajectory':assemble(scenario,evaluator,results)}


def improve(reference, settings, seed, timers):
    scenario=clone_scenario(reference['scenario'])
    incumbent={**reference,'scenario':scenario,'plan':copy.deepcopy(reference['plan']),
               'results':copy.deepcopy(reference['results'])}
    incumbent['evaluator']=scheduling.Evaluator(scenario,certified_timeline=True)
    paths={};scores={};memo={};changed=[];prune_details={};calls=0;pilots=[];pilot_candidates=[]
    gate=aggregate(scenario.scores,scenario.config)
    pruning_enabled=gate['iou']-scenario.config['shape_validation']['minimum_overall_iou']>=.03
    # Two bounded exact manufacturing pilots. D length alone is not a benefit:
    # compare pruned paths against original A *and* geometry-preserving merge.
    # This is heuristic screening, not a proof about every other layer.
    if pruning_enabled:
        with timers.stage('core_pruning_schedule_pilot'):
            seen=set()
            for k in scenario.layers:
                key=normalize(scenario.layer_geometry[k].simplify(0.,preserve_topology=True)).wkb
                if key in seen:continue
                seen.add(key);original=[t.points[:,:2] for t in scenario.by_layer[k]]
                p,s,d=pruning.prune(scenario.layer_geometry[k],original,scenario.config,
                                    budget=settings['core_prune_evaluations']/32)
                memo[key]=(p,s,d);calls+=1
                record={'layer':k,'A_duration':reference['results'][k]['duration'],'length_saved':cost(original)-cost(p),
                        'pruning':d,'useful':False}
                if cost(p)<cost(original)-1e-6:
                    local_tasks,_=merge_contiguous(scenario.by_layer[k])
                    merged=Scenario(scenario.config,scenario.target_top_mm,tuple(local_tasks),(k,),scenario.layer_geometry,
                                      {k:local_tasks},{k:scenario.scores[k]},0.)
                    merged_candidate=schedule(merged,0,seed);pilot_candidates.append(merged_candidate)
                    new_tasks,_=merge_contiguous(generate_tasks({k:scenario.layer_geometry[k]},{k:p},scenario.config))
                    pruned=Scenario(scenario.config,scenario.target_top_mm,tuple(new_tasks),(k,),scenario.layer_geometry,
                                      {k:new_tasks},{k:s},0.)
                    pruned_candidate=schedule(pruned,0,seed);pilot_candidates.append(pruned_candidate)
                    merged_m=merged_candidate['results'][k]['duration'];pruned_m=pruned_candidate['results'][k]['duration']
                    record.update(merged_duration=merged_m,pruned_duration=pruned_m,
                                  merged_exact_evaluations=merged_candidate['evaluator'].evaluations,
                                  pruned_exact_evaluations=pruned_candidate['evaluator'].evaluations,
                                  useful=pruned_m<min(record['A_duration'],merged_m)-1e-8)
                pilots.append(record)
                if len(pilots)>=2:break
        pruning_enabled=any(r['useful'] for r in pilots)
    with timers.stage('core_geometry'):
        for k in scenario.layers:
            original=[t.points[:,:2] for t in scenario.by_layer[k]]
            key=normalize(scenario.layer_geometry[k].simplify(0.,preserve_topology=True)).wkb
            if pruning_enabled and key in memo:
                p,s,d=memo[key]
            elif pruning_enabled and calls<settings['unique_prune_layers']:
                p,s,d=pruning.prune(scenario.layer_geometry[k],original,scenario.config,
                                     budget=settings['core_prune_evaluations']/32)
                memo[key]=(p,s,d);calls+=1
            else:p,s,d=original,scenario.scores[k],{'reason':'geometry_preserved_or_unproductive_policy_screen'}
            if cost(p)>=cost(original)-1e-6:
                # No-gain proposals can merely reorder/split identical D paths.
                # Preserve original tasks instead of rebuilding every layer.
                p,s=original,scenario.scores[k];d={**d,'no_length_gain_original_paths_retained':True}
            paths[k]=p;scores[k]=s;prune_details[k]=d
        combined=aggregate(scores,scenario.config)
        if not combined['passed'] or combined['iou']<min(gate['iou'],scenario.config['shape_validation']['minimum_overall_iou']+.005)-1e-8:
            paths={k:[t.points[:,:2] for t in scenario.by_layer[k]] for k in scenario.layers}
            scores=dict(scenario.scores)
        # Task generation is only repeated when path geometry really changes.
        import numpy as np
        geometry_changed=any(len(paths[k])!=len(scenario.by_layer[k]) or
             any(not np.array_equal(p,t.points[:,:2]) for p,t in zip(paths[k],scenario.by_layer[k])) for k in scenario.layers)
        tasks,merges=merge_contiguous(generate_tasks(scenario.layer_geometry,paths,scenario.config)
                                       if geometry_changed else scenario.tasks)
        by=defaultdict(list)
        for t in tasks:by[t.layer].append(t)
        for k in scenario.layers:
            old=scenario.by_layer[k];new=by[k]
            if len(old)!=len(new) or any(not np.array_equal(a.points,b.points) for a,b in zip(old,new)):
                changed.append(k)
    if changed:
        variant=Scenario(scenario.config,scenario.target_top_mm,tuple(t for t in tasks if t.layer in changed),
                         tuple(changed),scenario.layer_geometry,{k:by[k] for k in changed},
                         {k:scores[k] for k in changed},0.)
        with timers.stage('core_assignment_and_replay'):
            alternative=schedule(variant,0,seed)
            incumbent,selected=layerwise(incumbent,alternative)
    else:selected=[]
    pilot_selected=[]
    with timers.stage('core_preserve_best_pilot'):
        for candidate in pilot_candidates:
            incumbent,taken=layerwise(incumbent,candidate);pilot_selected.extend(taken)
    after_merge_m=incumbent['trajectory']['makespan_s']
    with timers.stage('core_critical_search'):
        p,r,info=conflicts.optimize(incumbent['evaluator'],incumbent['plan'],incumbent['results'],
                                    width=2,depth=2,max_evaluations=settings['critical_evaluations'])
        incumbent={**incumbent,'plan':p,'results':r,'search':info,
                   'trajectory':assemble(incumbent['scenario'],incumbent['evaluator'],r)}
    after_critical_m=incumbent['trajectory']['makespan_s']
    if settings['core_steps']:
        with timers.stage('core_lns'):
            p,r,info2=scheduling.optimize(incumbent['evaluator'],incumbent['plan'],incumbent['results'],
                        algorithm='lns',budget=settings['core_steps'],max_steps=settings['core_steps'],seed=seed+1)
            incumbent={**incumbent,'plan':p,'results':r,
                       'trajectory':assemble(incumbent['scenario'],incumbent['evaluator'],r)}
    else:info2={'steps':0,'exact_candidates':0}
    return incumbent,{'contiguous_merges':merges,'geometry_changed':geometry_changed,
                      'unique_pruning_calls':calls,'pruning_enabled_after_pilots':pruning_enabled,
                      'pruning_pilots':pilots,'selected_pilot_layers':pilot_selected,
                      'pruning_screen_scope':'Two exact original/merged/pruned schedule pilots decide whether to expand geometry search. Heuristic may miss gains in other layers; not a lower-bound certificate.',
                      'changed_layers':changed,'selected_layers':selected,'pruning':prune_details,
                      'assignment_audit':alternative['search'] if changed else {'assignment_exact_evaluations':0},
                      'critical':info,'lns':info2,
                      'paired_stage_makespan':{'A':reference['trajectory']['makespan_s'],
                          'geometry_merge_assignment':after_merge_m,'critical':after_critical_m,
                          'lns':incumbent['trajectory']['makespan_s']},
                      'D_length_before':sum(t.length for t in reference['scenario'].tasks),
                      'D_length_after':sum(t.length for t in incumbent['scenario'].tasks),
                      'shape_before':aggregate(reference['scenario'].scores,reference['scenario'].config),
                      'shape_after':aggregate(incumbent['scenario'].scores,incumbent['scenario'].config),
                      'delta_scope':'Unchanged global barriers; downstream release/parking shifted by assemble; differential full replay tests required.'}

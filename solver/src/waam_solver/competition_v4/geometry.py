"""Bead-aware bounded refinement. No file names or instance identities enter here."""
from collections import Counter
from time import perf_counter
import math
import numpy as np
from shapely import LineString, Point, Polygon, MultiPoint, union_all, make_valid, voronoi_polygons
from shapely.ops import substring
from ..geometry import polygons, toolpaths
from ..residual import lines
from ..shape_validation import normalize, deposit_polygon, measure, aggregate


def cost(paths):
    return sum(LineString(p).length for p in paths)


def skeleton(region, bead):
    """Bounded boundary Voronoi medial graph (approximation, not exact skeleton)."""
    points=[]
    for poly in polygons(region):
        for ring in [poly.exterior,*poly.interiors]:
            n=min(512,max(8,math.ceil(ring.length/max(bead/3,1e-3))))
            points.extend(ring.interpolate(i/n,normalized=True).coords[0] for i in range(n))
    points=np.unique(points,axis=0)
    if len(points)<4 or len(points)>4096: return [],{'vertices':0,'branches':0,'terminals':0,'bounded_out':True}
    try: vor=voronoi_polygons(MultiPoint(points),only_edges=True)
    except Exception: return [],{'vertices':0,'branches':0,'terminals':0}
    edges=[]; degree=Counter()
    for edge in lines(vor):
        if edge.length>1e-5 and region.covers(edge):
            a,b=tuple(edge.coords[0]),tuple(edge.coords[-1])
            edges.append(edge); degree[a]+=1; degree[b]+=1
    return edges,{'vertices':len(degree),'branches':sum(v>=3 for v in degree.values()),
                  'terminals':sum(v==1 for v in degree.values())}


def topology(region,bead):
    components=polygons(make_valid(region)); descriptions=[]
    for p in components[:64]:
        rect=np.asarray(p.minimum_rotated_rectangle.exterior.coords)[:4]
        widths=np.linalg.norm(np.roll(rect,-1,axis=0)-rect,axis=1)
        lo,hi=float(min(widths)),float(max(widths))
        edges,graph=skeleton(p,bead)
        distances=[Point(e.interpolate(.5,normalized=True)).distance(p.boundary) for e in edges]
        if not distances: distances=[p.representative_point().distance(p.boundary)]
        compact=4*math.pi*p.area/max(p.length*p.length,1e-12)
        labels=[]
        if lo<bead: labels.append('narrow strip' if hi>2*lo else 'small island')
        else: labels.append('wide residual')
        if p.area<bead*bead*.05: labels.append('tiny residual')
        if graph['branches']>2: labels.append('branched residual')
        if p.interiors: labels.append('hole-adjacent residual')
        if p.area/p.convex_hull.area<.85: labels.append('concave pocket')
        if hi/max(lo,1e-12)>3 and compact<.3: labels.append('curved channel')
        descriptions.append({'area':p.area,'perimeter':p.length,'bbox':p.bounds,'aspect_ratio':hi/max(lo,1e-12),
            'minimum_width_approx':2*min(distances),'maximum_width_approx':2*max(distances),
            'bbox_min_width':lo,'bbox_max_width':hi,'hole_count':len(p.interiors),
            'compactness':compact,'elongation':hi/max(lo,1e-12), 'skeleton':graph,
            'distance_transform_approx':{'min':min(distances),'mean':float(np.mean(distances)),'max':max(distances)},
            'local_width_over_bead':2*float(np.mean(distances))/bead,'classes':labels})
    return {'connected_component_count':len(components),'hole_count':sum(len(p.interiors) for p in components),
            'classes':['multiple islands','disconnected region'] if len(components)>1 else [],'components':descriptions,
            'diagnostic_component_limit':64,'omitted_diagnostic_components':max(0,len(components)-64)}


def contours(target,config,fraction):
    r=config['process']['bead_width_mm']/2; depth=r*fraction; paths=[]
    q=config['shape_validation']['polygon_buffer_resolution']
    # Maximum inset depth bounded by bbox, not a fixed model dimension.
    bounds=target.bounds; limit=math.hypot(bounds[2]-bounds[0],bounds[3]-bounds[1])+r
    while depth<limit:
        inset=target.buffer(-depth,quad_segs=q)
        if inset.is_empty: break
        for p in polygons(inset):
            for ring in [p.exterior,*p.interiors]:
                line=LineString(ring.coords).simplify(r*.01)
                if line.length>1e-5: paths.append(np.asarray(line.coords))
        depth+=r*1.8
    return paths


def sweeps(region,bead,axis_only=False):
    result=[]
    for p in polygons(region):
        rect=np.asarray(p.minimum_rotated_rectangle.exterior.coords)[:4]
        edges=np.roll(rect,-1,axis=0)-rect
        axis=edges[np.argmax(np.linalg.norm(edges,axis=1))]; axis=axis/np.linalg.norm(axis)
        normal=np.array([-axis[1],axis[0]]); center=np.asarray(p.centroid.coords[0])
        width=np.ptp((rect-center)@normal); length=np.ptp((rect-center)@axis)
        n=1 if axis_only else max(1,min(512,math.ceil(width/(bead*.8))))
        offsets=[0.] if axis_only else np.linspace(-width/2+width/(2*n),width/2-width/(2*n),n)
        for offset in offsets:
            c=center+offset*normal
            for line in lines(LineString([c-axis*(length+bead),c+axis*(length+bead)]).intersection(p)):
                # Rounded end-caps cover half a bead beyond the TCP segment.
                trim=min(bead*.45,line.length*.45)
                short=substring(line,trim,line.length-trim)
                if isinstance(short,LineString): result.append(np.asarray(short.coords))
    return result


def additions(target,residual,config,level):
    bead=config['process']['bead_width_mm']; r=bead/2
    allowed=target.buffer(-r*.98)
    # No area-based island dropping. Candidates are admitted only by bead metrics.
    for p in polygons(residual):
        for a in sweeps(p,bead,True): yield 'centerline/longest-axis',a
        center=np.asarray(p.representative_point().coords[0]); d=max(1e-4,bead*.001)
        yield 'short-segment',np.array([center-[d,0],center+[d,0]])
        if level>=1:
            for a in contours(p,config,.8): yield 'local-contour',a
        if level>=2:
            # Medial graph of the target neighborhood also extends through gaps
            # whose residual boundary alone has an unhelpful medial axis.
            neighborhood=target.intersection(p.buffer(bead))
            for edge in skeleton(neighborhood,bead)[0]:
                for line in lines(edge.intersection(p.buffer(r*.8))):
                    yield 'medial-axis',np.asarray(line.coords)
        if level>=3:
            for a in sweeps(p.buffer(r*.5).intersection(target),bead):
                clipped=LineString(a).intersection(allowed)
                for line in lines(clipped): yield 'raster/hatch',np.asarray(line.coords)
                if level>=4: yield 'dense-hatch',a


def layer_refine(target,initial,config,budget=8.,max_iterations=5,componentwise=True):
    start=perf_counter(); target=normalize(target,config); paths=list(initial)
    deposited=deposit_polygon(paths,config); score=measure(target,deposited,config)
    trace=[]; tested=0; methods=Counter(); initial_length=cost(paths)
    thresholds=config['shape_validation']
    def passed(s): return aggregate({0:s},config)['passed']
    def rank(s,p):
        # Before feasibility: official acceptance, IoU, coverage, overfill.
        # Once feasible, manufacturing cost wins; margin is a tie-breaker only.
        return (1,-cost(p),-len(p),s['iou']) if passed(s) else (0,s['iou'],s['coverage'],-(s['overfill'] or 0))
    if passed(score): return paths,score,{'seconds':perf_counter()-start,'trace':[],'added_length_mm':0.,'reason':'already_full_shape_feasible'}
    components=polygons(target)
    if componentwise and len(components)>1:
        # Each component receives attention, including tiny islands. Translation
        # canonicalization is an in-call geometry memo, never an STL lookup.
        from shapely.affinity import translate
        from shapely import normalize as canonical
        memo={};combined=[];detail=[]
        for component in components:
            x,y=component.bounds[:2];local=translate(component,xoff=-x,yoff=-y)
            key=canonical(local).wkb
            if key not in memo:
                seed=contours(local,config,.99)
                memo[key]=layer_refine(local,seed,config,max(.05,budget/max(1,min(len(components),16))),max_iterations,False)
            candidate,s,d=memo[key]
            combined.extend(p+np.array([x,y]) for p in candidate)
            if len(detail)<64:detail.append({'iou':s['iou'],'reason':d['reason']})
        combined_score=measure(target,deposit_polygon(combined,config),config)
        if rank(combined_score,combined)>rank(score,paths):paths=combined;score=combined_score
        trace.append({'method':'componentwise','components':len(components),'unique_components':len(memo),'iou':score['iou'],'details':detail})
        if passed(score) or perf_counter()-start>budget:
            return paths,score,{'seconds':perf_counter()-start,'trace':trace,'added_length_mm':cost(paths)-initial_length,
                               'reason':'feasible' if passed(score) else 'component_budget_exhausted'}
    # Replacement, not just additions, is essential when the initial bead
    # contour overfills islands. Evaluate the same portfolio for every shape.
    for fraction in (.95,.85,.7,.5,.25):
        if perf_counter()-start>budget: break
        candidate=contours(target,config,fraction)
        trial=measure(target,deposit_polygon(candidate,config),config); tested+=1
        trace.append({'method':'inset-contour','fraction':fraction,'iou':trial['iou'],'length_mm':cost(candidate),'passed':passed(trial)})
        if rank(trial,candidate)>rank(score,paths): paths=candidate;score=trial
    deposited=deposit_polygon(paths,config)
    if not passed(score):
        for iteration in range(max_iterations):
            old_iou=score['iou']; old_residual=score['residual_area_mm2']
            residual=target.difference(deposited)
            # Holes remain an explicit constraint. The admissible penalty is
            # configured overall overfill, never a changed Validator threshold.
            holes=union_all([Polygon(r) for p in polygons(target) for r in p.interiors])
            for method,candidate in additions(target,residual,config,iteration):
                if perf_counter()-start>budget or tested>=2048: break
                addition=deposit_polygon([candidate],config); updated=deposited.union(addition)
                trial=measure(target,updated,config);tested+=1
                hole_penalty=addition.intersection(holes).area
                if (trial['iou']>score['iou']+1e-7 and trial['overfill']<=thresholds['maximum_overall_overfill_ratio']
                    and hole_penalty<=target.area*thresholds['maximum_overall_overfill_ratio']):
                    paths.append(candidate);deposited=updated;score=trial;methods[method]+=1
                if passed(score): break
            trace.append({'iteration':iteration,'iou':score['iou'],'coverage':score['coverage'],'overfill':score['overfill'],
                          'residual_area':score['residual_area_mm2'],'length_mm':cost(paths),'passed':passed(score)})
            if passed(score) or perf_counter()-start>budget: break
            if iteration>=3 and score['iou']-old_iou<1e-6 and old_residual-score['residual_area_mm2']<1e-5: break
    score=measure(target,deposit_polygon(paths,config),config)
    return paths,score,{'seconds':perf_counter()-start,'trace':trace,'candidate_count':tested,'accepted_methods':dict(methods),
        'added_length_mm':cost(paths)-initial_length,'residual_topology':topology(target.difference(deposited),config['process']['bead_width_mm']),
        'reason':'feasible' if passed(score) else 'budget_or_no_improving_candidate'}

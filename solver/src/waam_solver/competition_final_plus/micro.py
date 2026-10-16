"""Bounded continuous coordinate descent, using the actual nominal bead union."""
from time import perf_counter
import numpy as np
from shapely import LineString
from ..geometry import polygons
from ..shape_validation import measure,deposit_polygon,aggregate
from ..competition_final.geometry import micro_candidates,contours,cost

def refine(target,initial,config,budget=1.5):
    start=perf_counter();paths=list(initial);score=measure(target,deposit_polygon(paths,config),config);before=score['iou'];tested=0
    bead=config['process']['bead_width_mm'];center=np.asarray(target.centroid.coords[0])
    def key(s):
        t=config['shape_validation']
        legal=s['coverage']>=t['minimum_overall_coverage'] and s['overfill']<=t['maximum_overall_overfill_ratio']
        return (int(legal),s['iou'])
    def consider(trial):
        nonlocal paths,score,tested
        if not trial or any(len(p)<2 or not np.isfinite(p).all() for p in trial):return False
        s=measure(target,deposit_polygon(trial,config),config);tested+=1
        if key(s)>key(score) and s['iou']>score['iou']+1e-9:
            paths=[p.copy() for p in trial];score=s;return True
        return False
    # Continuous inset is especially useful between the old finite grid points.
    for name,trial in micro_candidates(target,config):
        if perf_counter()-start>budget*.35:break
        consider(trial)
    # Disconnected short strokes can avoid the edge overfill of a closed loop.
    # Centers come from actual inset geometry; no shape-name/coordinate lookup.
    for inset in np.linspace(.25,.65,17):
        if perf_counter()-start>=budget*.6:break
        pieces=polygons(target.buffer(-bead*inset))
        centers=[v for p in pieces for v in np.asarray(p.simplify(bead*.03).exterior.coords)[:-1]]
        if 1<=len(centers)<=12:
            consider([np.array([v-np.array([bead*.001,0]),v+np.array([bead*.001,0])]) for v in centers])
    step=bead*.12
    while step>bead*.0005 and perf_counter()-start<budget:
        improved=False
        # Translation, anisotropic scale and rotation are continuous variables.
        for dx,dy,sx,sy,theta in [(step,0,1,1,0),(-step,0,1,1,0),(0,step,1,1,0),(0,-step,1,1,0),
            (0,0,1+step/bead,1,0),(0,0,1-step/bead,1,0),(0,0,1,1+step/bead,0),(0,0,1,1-step/bead,0),
            (0,0,1,1,step/bead*.25),(0,0,1,1,-step/bead*.25)]:
            if perf_counter()-start>=budget:break
            mat=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            trial=[((p-center)*np.array([sx,sy]))@mat.T+center+np.array([dx,dy]) for p in paths]
            improved=consider(trial) or improved
        # Endpoint / vertex descent covers trim, lateral shift and short bends.
        for i,p in enumerate(list(paths)):
            if len(p)>24:continue
            closed=np.linalg.norm(p[0]-p[-1])<1e-7
            for j in range(len(p)-int(closed)):
                for axis in (0,1):
                    for sign in (-1,1):
                        if perf_counter()-start>=budget:break
                        trial=[q.copy() for q in paths];trial[i][j,axis]+=sign*step
                        if closed and j==0:trial[i][-1]=trial[i][0]
                        improved=consider(trial) or improved
        if not improved:step*=.5
    return paths,score,{'initial_iou':before,'final_iou':score['iou'],'evaluations':tested,'seconds':perf_counter()-start,
        'length_mm':cost(paths),'scope':'bounded coordinate descent, not a continuous global optimum'}

"""Synchronized adaptive continuous capsule certificates (not point sampling).

At a candidate start, each shared time interval is enclosed by both moving
Base--TCP convex hulls. Only ambiguous intervals are subdivided. Instantaneous
midpoint collisions reject quickly, but acceptance ALWAYS requires disjoint
continuous hulls. No Z-separation shortcut or sample-only safety assumption.
"""
import numpy as np
from shapely import convex_hull,multipoints,linestrings,distance
from .engine import Engine

def motion(tour):
    cached=getattr(tour,'_timeline_motion',None)
    if cached is None:
        knots=np.asarray([r[0] for r in tour.rows]); xy=np.asarray([r[1][:2] for r in tour.rows])
        cached=(knots,xy,np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))])
        tour._timeline_motion=cached
    return cached

def position(tour,times):
    knots,xy,_=motion(tour)
    i=np.clip(np.searchsorted(knots,times,side='right')-1,0,len(knots)-2)
    f=np.clip((times-knots[i])/(knots[i+1]-knots[i]),0.,1.)
    return xy[i]+(xy[i+1]-xy[i])*f[:,None]

def excluded_start_advance(tour,relative_times,penetration):
    """A collision witness excludes a provable interval of later start times.

    A moving Base--TCP segment's Hausdorff displacement is at most its TCP
    displacement. Distance to the other fixed-at-witness-time capsule is
    1-Lipschitz. While backwards TCP arclength stays below penetration, the
    same witness therefore remains colliding. This is NOT a monotonic-safety
    assumption; only that certified local forbidden interval is skipped.
    """
    knots,xy,cumulative=motion(tour)
    cur=np.interp(relative_times,knots,cumulative)
    target=np.maximum(0.,cur-np.maximum(0.,penetration-1e-8)*.999999)
    k=np.clip(np.searchsorted(cumulative,target,side='left'),1,len(knots)-1)
    denominator=cumulative[k]-cumulative[k-1]
    fraction=np.divide(target-cumulative[k-1],denominator,out=np.ones(len(k)),where=denominator>1e-12)
    earlier=knots[k-1]+np.clip(fraction,0,1)*(knots[k]-knots[k-1])
    return max(0.,float(np.max(relative_times-earlier)))

def pair_safe(config,tour,other,start,max_depth=12,return_skip=False):
    def verdict(ok,skip=0.): return (ok,skip) if return_skip else ok
    left=max(start,other.start); right=min(start+tour.duration,other.start+other.duration)
    if right<left: return verdict(True)
    robots={r['id']:r for r in config['robots']}; a=robots[tour.robot]; b=robots[other.robot]
    ba=np.asarray(a['base_xyz_mm'][:2]); bb=np.asarray(b['base_xyz_mm'][:2]); c=config['collision']
    arm=a['arm_envelope_radius_mm']+b['arm_envelope_radius_mm']+c['arm_clearance_mm']+.02
    tcp=a['tcp_radius_mm']+b['tcp_radius_mm']+.02
    knots=np.unique(np.r_[left,right,[start+r[0] for r in tour.rows if left<start+r[0]<right],
                           [other.start+r[0] for r in other.rows if left<other.start+r[0]<right]])
    lo=knots[:-1]; hi=knots[1:]
    if not len(lo): lo=np.asarray([left]); hi=np.asarray([right])
    for depth in range(max_depth+1):
        if len(lo)>65536: return verdict(False)  # bounded-memory conservative fallback
        aa=position(tour,lo-start); az=position(tour,hi-start)
        ab=position(other,lo-other.start); bz=position(other,hi-other.start)
        ambiguous=np.zeros(len(lo),dtype=bool)
        if c['check_arm_envelope']:
            ga=convex_hull(multipoints(np.stack([np.broadcast_to(ba,aa.shape),aa,az],axis=1)))
            gb=convex_hull(multipoints(np.stack([np.broadcast_to(bb,ab.shape),ab,bz],axis=1)))
            ambiguous|=distance(ga,gb)<=arm
        if c['check_tcp_radius']:
            ga=convex_hull(multipoints(np.stack([aa,az],axis=1)))
            gb=convex_hull(multipoints(np.stack([ab,bz],axis=1)))
            ambiguous|=distance(ga,gb)<=tcp
        if not ambiguous.any(): return verdict(True)
        lo=lo[ambiguous]; hi=hi[ambiguous]; mid=(lo+hi)/2
        pa=position(tour,mid-start); pb=position(other,mid-other.start)
        penetration=np.full(len(mid),-np.inf)
        if c['check_tcp_radius']: penetration=np.maximum(penetration,tcp-np.linalg.norm(pa-pb,axis=1))
        if c['check_arm_envelope']:
            ga=linestrings(np.stack([np.broadcast_to(ba,pa.shape),pa],axis=1))
            gb=linestrings(np.stack([np.broadcast_to(bb,pb.shape),pb],axis=1))
            penetration=np.maximum(penetration,arm-distance(ga,gb))
        colliding=penetration>=0
        if colliding.any():
            skip=excluded_start_advance(tour,mid[colliding]-start,penetration[colliding]) if return_skip else 0.
            return verdict(False,skip)
        if depth==max_depth: return verdict(False)  # unresolved is unsafe, never accepted
        lo,hi=np.r_[lo,mid],np.r_[mid,hi]
    return verdict(False)

class TimelineEngine(Engine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.timeline_cache={}; self.timeline_checks=0

    def safe(self,tour,start):
        key=(id(tour),float(start),len(self.decisions))
        if key in self.timeline_cache: return self.timeline_cache[key]
        for other in self.tours:
            if other.robot==tour.robot or other.start+other.duration<start: continue
            self.timeline_checks+=1
            if not pair_safe(self.config,tour,other,start):
                self.timeline_cache[key]=False; return False
        self.timeline_cache[key]=True; return True

    def earliest_start(self,tour,release):
        ready=max(release,self.rows[tour.robot][-1][0])
        if self.safe(tour,ready): return ready
        # Serial end is a certified upper bound and needs no quadratic spatial
        # pair enumeration. Search earlier windows; feasibility is NOT assumed
        # monotone. This is a feasible start heuristic, not an earliest proof.
        upper=max([ready]+[t.start+t.duration+1e-4 for t in self.tours if t.robot!=tour.robot])
        low=ready
        for fraction in (.25,.5,.75):
            candidate=ready+(upper-ready)*fraction
            if self.safe(tour,candidate): upper=candidate; break
            low=candidate
        for _ in range(12):
            candidate=(low+upper)/2
            if self.safe(tour,candidate): upper=candidate
            else: low=candidate
        return upper

    def commit(self,*args,**kwargs):
        super().commit(*args,**kwargs); self.timeline_cache.clear()

class CertifiedTimelineEngine(TimelineEngine):
    """Witness-driven start search, with original exact interval fallback."""
    def earliest_start(self,tour,release):
        ready=max(release,self.rows[tour.robot][-1][0]); start=ready
        for iteration in range(96):
            skip=0.; safe=True; unresolved=False
            for other in self.tours:
                if other.robot==tour.robot or other.start+other.duration<start: continue
                self.timeline_checks+=1
                passed,advance=pair_safe(self.config,tour,other,start,return_skip=True)
                if not passed:
                    safe=False; skip=max(skip,advance)
                    unresolved|=advance<=1e-8
            if safe: return start
            if unresolved or skip<1e-7: break
            start+=skip+1e-7
        # Each skipped interval contains a real collision witness. The fallback
        # cannot skip an earlier feasible window due to an unsafe bisection.
        return Engine.earliest_start(self,tour,start)

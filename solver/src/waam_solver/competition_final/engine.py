"""Reuse a proved start certificate only until the reservation frontier changes."""
from ..competition_v3.timeline import CertifiedTimelineEngine

class CertifiedEngine(CertifiedTimelineEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.proved_starts=set();self.certificate_hits=0;self.held_tours={}
    def earliest_start(self,tour,release):
        ready=max(release,self.rows[tour.robot][-1][0])
        # This is a per-commit ephemeral cache. All geometry and other starts
        # remain immutable between candidate probing and commit certification.
        key=(id(tour),ready)
        if key in self.proved_starts:
            self.certificate_hits+=1;return ready
        start=super().earliest_start(tour,release)
        if len(self.held_tours)>=64:
            self.proved_starts.clear();self.held_tours.clear()
        self.held_tours[id(tour)]=tour
        self.proved_starts.add((id(tour),start))
        return start
    def commit(self,*args,**kwargs):
        super().commit(*args,**kwargs)
        self.proved_starts.clear()
        self.held_tours.clear()

"""Exclusive wall-clock stage timing: nested stages are not double counted."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter

STAGES=('config_loading','stl_loading','stl_slicing','polygon_preprocessing',
        'base_contour_generation','residual_fill_generation','task_generation',
        'reach_feasibility','greedy_assignment','nearest_neighbor_ordering',
        'collision_candidate_generation','parallel_reservation','wait_insertion',
        'trajectory_generation','csv_export','internal_validation')
ACTIVE=ContextVar('waam_profile',default=None)


class Profile:
    def __init__(self):
        self.seconds=Counter({s:0. for s in STAGES})
        self.counts=Counter()
        self.stack=[]

    def report(self,total):
        return {'exclusive_wall_seconds':dict(self.seconds),'counts':dict(self.counts),
                'total_solver_seconds':total,'unattributed_seconds':total-sum(self.seconds.values())}


@contextmanager
def stage(name):
    p=ACTIVE.get()
    if p is None:
        yield
        return
    item=[perf_counter(),0.]
    p.stack.append(item)
    try:
        yield
    finally:
        elapsed=perf_counter()-item[0]
        p.stack.pop()
        p.seconds[name]+=elapsed-item[1]
        if p.stack:
            p.stack[-1][1]+=elapsed


def timed(name):
    def decorate(func):
        @wraps(func)
        def call(*args,**kwargs):
            with stage(name):
                return func(*args,**kwargs)
        return call
    return decorate


def count(name,value=1):
    if (p:=ACTIVE.get()) is not None:
        p.counts[name]+=value

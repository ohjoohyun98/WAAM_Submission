"""Lazy spatial/arm/TCP conflict-option graph, with bounded neighborhood cache.

Edges are potential spatial conflicts, NOT a claim that the tasks collide at
the same time. Temporal narrow-phase execution remains the acceptance gate.
"""
from collections import OrderedDict
import numpy as np
from shapely import MultiPoint,STRtree,distance

class ConflictIndex:
    def __init__(self,tasks,config):
        self.config=config; self.robots={r['id']:r for r in config['robots']}
        self.options=[]; arms=[]; tcps=[]
        for task in tasks:
            tcp=MultiPoint(task.points[:,:2]).convex_hull
            for robot in task.robots:
                self.options.append((task.id,robot)); tcps.append(tcp)
                arms.append(MultiPoint(np.vstack([task.points[:,:2],self.robots[robot]['base_xyz_mm'][:2]])).convex_hull)
        self.ids={key:i for i,key in enumerate(self.options)}
        self.arm=np.asarray(arms,dtype=object); self.tcp=np.asarray(tcps,dtype=object)
        self.arm_tree=STRtree(self.arm); self.tcp_tree=STRtree(self.tcp); self.cache=OrderedDict()

    def neighbors(self,task,robot):
        key=(task,robot)
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        i=self.ids[key]; a=self.robots[robot]; c=self.config['collision']; found={}
        for channel,tree,geoms,enabled,radius_name,extra in (
            ('arm',self.arm_tree,self.arm,c['check_arm_envelope'],'arm_envelope_radius_mm',c['arm_clearance_mm']),
            ('tcp',self.tcp_tree,self.tcp,c['check_tcp_radius'],'tcp_radius_mm',0.),
            ('spatial',self.tcp_tree,self.tcp,True,None,self.config['process']['bead_width_mm'])):
            if not enabled: continue
            radius=extra+(a[radius_name]+max(r[radius_name] for r in self.robots.values()) if radius_name else 0.)
            for j in tree.query(geoms[i],predicate='dwithin',distance=radius+.02):
                other,rr=self.options[j]
                if other==task or rr==robot: continue
                actual=extra+(a[radius_name]+self.robots[rr][radius_name] if radius_name else 0.)
                if distance(geoms[i],geoms[j])<=actual+.02:
                    found.setdefault((other,rr),set()).add(channel)
        self.cache[key]=found
        if len(self.cache)>32: self.cache.popitem(last=False)
        return found

    def sample_density(self,limit=12):
        if not self.options: return 0.
        ids=np.unique(np.linspace(0,len(self.options)-1,min(limit,len(self.options))).astype(int))
        degree=[]
        for i in ids:
            task,robot=self.options[i]
            denominator=sum(t!=task and r!=robot for t,r in self.options)
            relevant=sum(bool(channels & {'arm','tcp'}) for channels in self.neighbors(task,robot).values())
            degree.append(relevant/max(1,denominator))
        return float(np.mean(degree))

    def assigned_density(self,plan,limit=12):
        assigned={task:robot for robot,queue in plan.tours for task,_ in queue}
        tasks=sorted(assigned)
        indices=np.unique(np.linspace(0,len(tasks)-1,min(limit,len(tasks))).astype(int))
        values=[]
        for i in indices:
            task=tasks[i]; robot=assigned[task]
            denominator=sum(r!=robot for r in assigned.values())
            count=sum(assigned.get(t)==r and bool(ch & {'arm','tcp'}) for (t,r),ch in self.neighbors(task,robot).items())
            values.append(count/max(1,denominator))
        return float(np.mean(values)) if values else 0.

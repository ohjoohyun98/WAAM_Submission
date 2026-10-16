import ctypes
from ctypes import wintypes
from collections import defaultdict
import numpy as np

def peak_memory_mb():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    data=Counters(); data.cb=ctypes.sizeof(data)
    kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=wintypes.HANDLE
    query=ctypes.WinDLL('psapi').GetProcessMemoryInfo
    query.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    query.restype=wintypes.BOOL
    ok=query(kernel.GetCurrentProcess(),ctypes.byref(data),data.cb)
    return data.PeakWorkingSetSize/1024**2 if ok else None

def measure(result,scenario):
    makespan=result['makespan_s']; stats={}
    p=scenario.config['process']; arc=(p.get('arc_on_time_s') or 0.)+(p.get('arc_off_time_s') or 0.)
    clearance=max(p['bead_width_mm'],p['layer_height_mm'],*(r['tcp_radius_mm'] for r in scenario.config['robots']))
    parking={r['id']:np.array([*r['base_xyz_mm'][:2],scenario.target_top_mm+clearance+.02]) for r in scenario.config['robots']}
    for robot,rows in result['rows'].items():
        entry={'D':0.,'T':0.,'W':0.,'deposition_distance_mm':0.,'travel_distance_mm':0.,'W_intervals':0,
               'completion_s':rows[-1][0],'idle_after_completion_s':max(0.,makespan-rows[-1][0]),
               'parking_travel_s':0.,'inter_task_travel_s':0.}
        seen_D=False; pending_T=0.
        for (t,a,m),(u,b,_) in zip(rows[:-1],rows[1:]):
            if np.array_equal(a,parking[robot]):
                entry['parking_travel_s']+=pending_T; pending_T=0.; seen_D=False
            entry[m]+=u-t
            if m in ('D','T'): entry['deposition_distance_mm' if m=='D' else 'travel_distance_mm']+=float(np.linalg.norm(b-a))
            else: entry['W_intervals']+=1
            if m=='T': pending_T+=u-t
            elif m=='D':
                entry['inter_task_travel_s' if seen_D else 'parking_travel_s']+=pending_T
                pending_T=0.; seen_D=True
        entry['parking_travel_s']+=pending_T
        entry['mandatory_arc_s']=sum(d['robot']==robot for d in result['trace'])*arc
        entry['nonwaiting_load_s']=entry['D']+entry['T']+entry['mandatory_arc_s']
        stats[robot]=entry
    waits=defaultdict(float)
    for wait in result['waits']: waits[wait['reason']]+=wait['duration_s']
    return {'makespan_s':makespan,'robots':stats,'wait_robot_seconds':dict(waits),
            'wait_percent_of_makespan':{k:v/max(makespan,1e-9)*100 for k,v in waits.items()},
            'total_travel_s':sum(x['T'] for x in stats.values()),'total_wait_robot_s':sum(x['W'] for x in stats.values()),
            'total_deposition_robot_s':sum(x['D'] for x in stats.values()),
            'deposition_normalized_makespan':makespan/max(1e-9,sum(x['D'] for x in stats.values())/3),
            'critical_collision_s':result.get('critical_collision_s'),
            'collision_free_barrier_s':result.get('collision_free_barrier_s'),
            'completion_load_imbalance_s':max(s['nonwaiting_load_s'] for s in stats.values())-min(s['nonwaiting_load_s'] for s in stats.values()),
            'assignment_counts':{r:sum(d['robot']==r for d in result['trace']) for r in stats}}

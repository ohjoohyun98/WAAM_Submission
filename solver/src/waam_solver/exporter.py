import csv
import numpy as np


def export_csv(rows,path):
    if sorted(rows)!=[1,2,3]:
        raise ValueError('Exactly three robot trajectories required')
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['robot_id','time_s','x_mm','y_mm','z_mm','mode'])
        for robot,items in sorted(rows.items()):
            if len(items)<2 or items[0][0]!=0 or items[-1][2]!='W':
                raise ValueError('Invalid start/end trajectory contract')
            previous=-1
            for time,point,mode in items:
                # Serialize enough digits to keep small, positive intervals distinct.
                time_text=format(time,'.15g')
                if not np.isfinite([time,*point]).all() or float(time_text)<=previous:
                    raise ValueError('Invalid or duplicate serialized timestamp')
                writer.writerow([robot,time_text,*(format(x,'.12g') for x in point),mode])
                previous=float(time_text)

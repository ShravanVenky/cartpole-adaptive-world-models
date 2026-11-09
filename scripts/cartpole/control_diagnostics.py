"""Saved-trajectory backbone coverage and failure-event diagnostics."""
import json,math
import numpy as np
import pandas as pd

def diagnostics(run):
    coverage=json.loads((run/'backbone_coverage.json').read_text());results=[]
    for path in sorted((run/'episodes').glob('*.json')):
        packet=json.loads(path.read_text());data=np.load(path.with_suffix('.npz'))
        states=data['states'];actions=data['actions'];pred=data['one_step_prediction']
        for i,row in enumerate(packet['rows']):
            end=max(1,int(row['survival_steps']));s=states[i,:end];u=actions[i,:end,0]
            z=np.column_stack([s[:,0],s[:,1],np.sin(s[:,2]),np.cos(s[:,2]),s[:,3],u])
            q=np.array(coverage[str(row['seed'])]['quantiles']);margin=1e-5+.01*(q[-1]-q[0])
            outside=((z<q[0]-margin)|(z>q[-1]+margin)).any(1)
            central=((z<q[1])|(z>q[3])).any(1)
            err=pred[i,:end]-states[i,1:end+1];err[:,2]=np.arctan2(np.sin(err[:,2]),np.cos(err[:,2]))
            squared=np.mean((err/np.array([2.4,3.,math.pi,4.]))**2,axis=-1)
            late=squared[max(0,end-10):];early=squared[max(0,end-60):max(1,end-10)]
            if row['failed']:reason='rail_or_numerical_constraint'
            elif not row['reached_upright']:reason='never_stabilized'
            elif row['settled_then_lost']:reason='lost_stabilization'
            elif not row['success']:reason='task_stability_criterion'
            else:reason='success'
            results.append({k:row[k] for k in ['seed','trial','environment','category','task','horizon','protocol','method','budget','success']}|{
                'failure_category':reason,'outside_training_marginal_bounds_fraction':float(outside.mean()),
                'outside_training_central98_fraction':float(central.mean()),'large_angle_fraction':float((np.abs(np.arctan2(np.sin(s[:,2]),np.cos(s[:,2])))>math.pi/2).mean()),
                'near_force_limit_fraction':float((np.abs(u)>9.5).mean()),'pre_event_residual_mean':float(np.mean(late)),
                'earlier_residual_mean':float(np.mean(early)),'residual_spike_ratio':float(np.mean(late)/max(np.mean(early),1e-12))})
    frame=pd.DataFrame(results);frame.to_csv(run/'coverage_and_failure_events.csv',index=False)
    frame.groupby(['task','method','success']).agg(outside_bounds=('outside_training_marginal_bounds_fraction','mean'),
        outside_central98=('outside_training_central98_fraction','mean'),near_force_limit=('near_force_limit_fraction','mean'),
        median_spike_ratio=('residual_spike_ratio','median'),episodes=('success','size')).to_csv(run/'coverage_failure_summary.csv')
    return frame

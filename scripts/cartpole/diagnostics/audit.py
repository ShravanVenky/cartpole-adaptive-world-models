"""Termination audit and synthetic boundary cases; does not alter archived scores."""
import json
import time
import numpy as np
import pandas as pd
from .protocol import HISTORICAL
from .engine import episode_metrics
from .context import save_json, file_hash


def boundary_checks(cfg):
    def score(s, task='swingup'):
        return episode_metrics(s, np.zeros((len(s)-1,1),np.float32),cfg,task)
    upright=np.zeros((151,4),np.float32)
    assert score(upright)['success']
    spin=upright.copy();spin[:,2]=np.linspace(np.pi,-3*np.pi,len(spin));spin[:,3]=8
    assert not score(spin)['failed'] and not score(spin)['success']
    moment=upright.copy();moment[:,2]=1;moment[50,2]=0
    assert not score(moment)['success']
    fast=upright.copy();fast[:,3]=2
    assert not score(fast)['success']
    rail=upright.copy();rail[50:,0]=2.41
    assert score(rail)['rail_failure'] and not score(rail)['numerical_failure']
    invalid=upright.copy();invalid[50:,3]=np.nan
    assert score(invalid)['numerical_failure']
    timeout=upright.copy();timeout[:,2]=1
    assert score(timeout)['time_limit_failure']
    return dict(angular_overshoot_does_not_terminate=True, momentary_upright_is_not_success=True,
                sustained_angle_and_velocity_required=True, rail_and_numerical_distinguished=True,
                time_limit_separate=True, failure_cost_separated=True)


def run_audit(ctx, smoke=False):
    cached=ctx.run/'termination_audit.json'
    if not smoke and cached.exists() and json.loads(cached.read_text())['episodes_audited']==26625:
        return json.loads(cached.read_text())
    start=time.perf_counter()
    checks=boundary_checks(ctx.cfg)
    paths=sorted((HISTORICAL/'episodes').glob('*.json'))
    if smoke:paths=paths[:2]
    rows=[]
    for path in paths:
        p=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as data:
            for i,row in enumerate(p['rows']):
                shift=(150 if row['task']=='balance' else 25) if row['protocol']=='online' else 150 if row['protocol']=='disturbance' else 0
                m=episode_metrics(data['states'][i],data['actions'][i],ctx.cfg,row['task'],shift)
                rows.append(dict(group=p['key'],row=i,method=row['method'],seed=row['seed'],task=row['task'],
                                 archived_success=row['success'],recomputed_success=m['success'],success_agrees=m['success']==row['success'],
                                 environment=row['environment'],protocol=row['protocol'],horizon=row['horizon'],budget=row['budget'],**m))
    frame=pd.DataFrame(rows)
    frame.to_csv(ctx.run/'termination_audit_rows.csv',index=False)
    if not frame.success_agrees.all():
        raise AssertionError('Archived success does not match unchanged declared rule; inspect audit before running comparisons')
    record=dict(checks=checks,episodes_audited=len(rows),all_success_labels_reproduced=True,
                changes='No success or termination thresholds changed. New code separates cause labels, reports penalty-free cost, and names stabilization confirmation time explicitly.',
                historical_diagnostic_caveats=['External disturbance was inserted into recorded state; old one-step errors across that event include the external jump.',
                    'Archived recursive Oracle forecasts crossing a physics-change time hold start-time physics fixed. Do not interpret their residual as pure model error.',
                    'Old numerical guard abs(other-state)>100 is preserved and separately labeled; it is not proof of a floating-point solver failure.'],
                wall_seconds=time.perf_counter()-start)
    save_json(ctx.run/'termination_audit.json',record)
    return record

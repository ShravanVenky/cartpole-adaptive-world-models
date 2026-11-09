"""Explicit archived development strata and prospective confirmation; unavailable slots stay visible."""
import json
import numpy as np
import pandas as pd
from .protocol import REFERENCE,METAS,ORIGINAL,ROLLOUT
from .common import checkpoint_K,save_json


def record(ctx,packet,row,step,slot,source=REFERENCE,split='development'):
    r=packet['rows'][row];data=ctx.arrays(packet['key'],source)
    end=int(r['survival_steps']);usable=int(r['usable_postchange_observations'])
    if step>usable or step<0:return None
    state=data['states'][row,step];angle=abs(np.arctan2(np.sin(state[2]),np.cos(state[2])))
    operating='swingup' if angle>.5 else 'stabilization' if angle<.15 and abs(state[3])<.8 else 'capture'
    return dict(id=slot,key=packet['key'],row=row,step=int(step),seed=int(r['seed']),trial=int(r['episode_trial']),
        source_dir=str(source),collector_method=r['method'],source_success=bool(r['success']),source_failure=r['failure_category'],
        usable=usable,support_K=checkpoint_K(step,usable),state=data['states'][row,step].tolist(),
        remaining=int(data['remaining'][row])-int(step),condition=r['condition'],phase=r['phase'],task=r['task'],
        environment=r['environment'],horizon=packet['case']['horizon'],split=split,source_end=end,
        source_cost=float(r['cost']),source_rail=bool(r['rail_failure']),
        operating_phase=operating,
        support_is_actual_past=True,cold_start_search=True)


def saved_manifest(ctx,bounded=False):
    target=ctx.run/'saved_manifest.json'
    if target.exists():return json.loads(target.read_text())
    cases=[];fixed=[];pre=[];unavailable=[];seeds=[11] if bounded else ctx.cfg['model_seeds']
    conditions=ctx.cfg['conditions'][:1] if bounded else ctx.cfg['conditions']
    for seed in seeds:
        for condition in conditions:
            key=f'confirmation_{seed}_{ctx.cfg["archived_trial"]}_{condition["name"]}'
            packet=ctx.packet(key)
            rows={r['method']:i for i,r in enumerate(packet['rows']) if r['safety']=='margin' and r['regularization']=='existing'}
            methods=METAS+(['oracle'] if condition['environment']=='combined_geometry' else [])
            times=ctx.cfg['replay_times_geometry'] if condition['environment']=='combined_geometry' else ctx.cfg['replay_times_other']
            if bounded:methods=METAS[:1];times=[50]
            for method in methods:
                for step in times:
                    slot=f'dev_{seed}_{condition["name"]}_{method}_t{step}'
                    r=record(ctx,packet,rows[method],step,slot)
                    if r:cases.append(r)
                    else:unavailable.append(dict(id=slot,reason='collector failed before selected state'))
            qkey=f'confirmation_{seed}_{ctx.cfg["query_trial"]}_{condition["name"]}'
            query=ctx.packet(qkey);qr=next(i for i,r in enumerate(query['rows']) if r['method']==ROLLOUT and r['safety']=='margin')
            h=record(ctx,packet,rows[ORIGINAL],50,f'fixed_{seed}_{condition["name"]}')
            q=record(ctx,query,qr,50,'query')
            if h and q:
                h.update(state=q['state'],query_key=qkey,query_row=qr,query_step=50,support_is_actual_past=False,
                         source_success=q['source_success'],source_cost=q['source_cost'],source_rail=q['source_rail'],operating_phase=q['operating_phase'],
                         query_independent_episode=True);fixed.append(h)
            else:unavailable.append(dict(id=f'fixed_{seed}_{condition["name"]}',reason='independent query or50transition support unavailable'))
            if condition['environment']=='combined_geometry':
                for method in [ORIGINAL,'ridge']:
                    i=next(i for i,r in enumerate(packet['rows']) if r['method']==method and r['safety']=='original' and r['regularization']=='existing')
                    row=packet['rows'][i]
                    for lead in ([10] if bounded else ctx.cfg['pre_failure_leads']):
                        slot=f'prefail_{seed}_{method}_lead{lead}'
                        r=record(ctx,packet,i,int(row['survival_steps'])-lead,slot) if row['rail_failure'] else None
                        if r:r['lead']=lead;pre.append(r)
                        else:unavailable.append(dict(id=slot,reason='no qualifying archived rail failure or insufficient past'))
    result=dict(cases=cases,fixed_queries=fixed,pre_failure=pre,unavailable=unavailable,
                role='Known V2 episodes are development diagnostics. New trials900/901 are the only independent confirmation.')
    save_json(target,result);return result


def behavior(states,actions,cfg):
    angle=np.arctan2(np.sin(states[:,2]),np.cos(states[:,2]));omega=np.abs(states[:,3]);p=cfg['phase_definitions']
    upright=np.abs(angle)<p['upright_angle'];stable=upright&(omega<cfg['success_omega'])
    transitions=np.diff(np.r_[False,upright,False].astype(int));starts=np.flatnonzero(transitions==1);ends=np.flatnonzero(transitions==-1)
    durations=ends-starts
    capture=(np.abs(angle)<=p['capture_angle'])&~stable;swing=np.abs(angle)>p['capture_angle']
    stable_delta=np.diff(np.r_[False,stable,False].astype(int));lengths=np.flatnonzero(stable_delta==-1)-np.flatnonzero(stable_delta==1)
    good_streak=int(lengths.max()) if len(lengths) else 0
    return dict(upright_visits=len(starts),first_upright_step=int(starts[0]) if len(starts) else None,
        longest_upright_visit=int(durations.max()) if len(durations) else 0,
        mean_entry_omega=float(omega[starts].mean()) if len(starts) else None,
        max_entry_omega=float(omega[starts].max()) if len(starts) else None,
        high_speed_entry_fraction=float((omega[starts]>p['high_omega']).mean()) if len(starts) else None,
        longest_stable_streak=good_streak,ever_sustained_stable=good_streak>=cfg['sustained_steps'],
        swingup_fraction=float(swing.mean()),capture_fraction=float(capture.mean()),stable_fraction=float(stable.mean()),
        low_speed_before_capture_fraction=float(((np.abs(angle)>.5)&(np.abs(angle)<1.57)&(omega<1.)).mean()),
        minimum_rail_clearance=float(2.4-np.abs(states[:,0]).max()),
        force_saturation_fraction=float((np.abs(actions)>=p['high_force']).mean()),
        force_reversals=int((actions[1:,0]*actions[:-1,0]<0).sum()),
        max_angular_velocity=float(omega.max()))


def saved_analysis(ctx,manifest):
    target=ctx.run/'saved_analysis.json'
    if target.exists():return json.loads(target.read_text())
    d=pd.read_csv(REFERENCE/'episodes.csv',low_memory=False);q=d[d.stage.eq('confirmation')&d.condition.isin([c['name'] for c in ctx.cfg['conditions']])]
    rows=[]
    for _,row in q.iterrows():
        data=ctx.arrays(row.key);i=int(row.row);n=int(row.survival_steps)
        rows.append(dict(seed=int(row.seed),trial=int(row.episode_trial),condition=row.condition,method=row.method,safety=row.safety,regularization=row.regularization,
            success=bool(row.success),failure_category=row.failure_category,cost=float(row.cost),
            **behavior(data['states'][i,:n+1],data['actions'][i,:n],ctx.cfg)))
    pd.DataFrame(rows).to_csv(ctx.run/'saved_behavior.csv',index=False)
    result=dict(records=len(rows),source='Only selected conditions from latest final V2, no new control executed',
        selected_replay_states=len(manifest['cases']),fixed_query_groups=len(manifest['fixed_queries']),pre_failure_states=len(manifest['pre_failure']),
        unavailable=manifest['unavailable'])
    save_json(target,result);return result

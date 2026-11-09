"""Sequential A decision gate, then information, recovery and untouched physics."""
import json
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from adaptive_world_model.utils.randomness import stable_seed
from diagnostics.context import transition_batch
from .protocol import METAS, optimizer, screen, disturbance_cases, unseen_cases, REFERENCE
from .common import save_json, save_npz, save_csv, digest, file_hash, query_record, support_from_episode, coverage
from .optimization import audit
from .offline import evaluate
from .simulation import control, policies, diagnostic_policies


def offline_rows(ctx,tag):
    return [r for p in sorted((ctx.run/'offline').glob(f'{tag}_*.json'))
            for r in json.loads(p.read_text())['rows']]


def select_offline(rows,method,allowed):
    frame=pd.DataFrame(rows); frame=frame[frame.method.isin([method,'frozen'])].copy()
    if not (frame.split=='development').all():raise RuntimeError('Confirmation data cannot select optimizer settings')
    frozen=frame[frame.method=='frozen'].drop_duplicates(['query_id','support_K']).set_index(['query_id','support_K']).one_step_nmse
    d=frame[frame.method==method].copy()
    d['frozen_nmse']=[frozen.loc[(r.query_id,r.support_K)] for r in d.itertuples()]
    d['selection_score']=np.log(np.maximum(d.one_step_nmse,1e-12)/np.maximum(d.frozen_nmse,1e-12))+np.log1p(d.planning_regret)
    mean=d.groupby('optimizer').agg(score=('selection_score','mean'),accepted=('accepted','mean'),fit_ms=('fit_ms','mean')).reset_index()
    mean=mean[(mean.optimizer.isin(allowed))&(mean.accepted>=.95)].sort_values(['score','fit_ms','optimizer'])
    return mean


def case_for(env,split,name=None):
    return dict(name=name or f'active_swingup__{env}',environment=env,phase='active_swingup',task='swingup',split=split)


def choose_control_setting(frame,method,shortlist):
    if not (frame.split=='development').all():raise RuntimeError('Confirmation control outcomes cannot select settings')
    baseline=frame[frame.policy==method];chosen=optimizer();selection=[];candidates=[]
    reason='Baseline retained; no admissible development control gain.'
    for c in shortlist:
        group=frame[frame.policy==f'{method}__{c["name"]}']
        if len(group)!=len(baseline) or not len(group):continue
        nom=group[group.environment=='nominal'].success.mean()-baseline[baseline.environment=='nominal'].success.mean()
        gain=group.success.mean()-baseline.success.mean();cost_ratio=group.cost.mean()/max(baseline.cost.mean(),1e-8)
        eligible=nom>=-.10 and gain>=0 and (gain>0 or cost_ratio<=.90)
        candidates.append((eligible,gain,-cost_ratio,-c['steps'],c))
        selection.append(dict(method=method,config=c,success_gain=gain,cost_ratio=cost_ratio,nominal_success_delta=nom,eligible=eligible))
    acceptable=[c for c in candidates if c[0]]
    if acceptable:chosen=max(acceptable,key=lambda x:x[:4])[-1];reason='Development control gain with nominal regression guard.'
    selection.append(dict(method=method,selected=chosen,reason=reason));return chosen,selection


def collection(ctx,seed,trial,environment,strategy='ordinary',split='confirmation'):
    case=case_for(environment,split,f'B_collect_{environment}_{strategy}')
    case.update(collection_only=True,steps=50,excitation=strategy=='excitation',match_key=f'B_collection_{environment}')
    key=f'collect_{seed}_{trial}_{environment}_{strategy}'
    packet=control(ctx,key,seed,trial,case,[policies(dense=False)[1]],online=True)
    if not packet['rows'] or packet['rows'][0]['actual_interactions']<50:
        return None,dict(key=key,available=False,actual_interactions=packet.get('actual_interactions',0),reason='collector failed before 50 transitions')
    with np.load(ctx.run/'collections'/f'{key}.npz') as a:
        support=transition_batch(a['observations'][0,:50].copy(),a['actions'][0,:50].copy(),a['observations'][0,1:51].copy(),
                                 f'hardening/support_collector/{key}')
    return support,dict(key=key,available=True,actual_interactions=50,collector_cost=packet['rows'][0]['cost'],
                        rail_failure=packet['rows'][0]['rail_failure'],privileged=False)


def diverse(ctx,seed,environment):
    path=REFERENCE/'dense'/f'{seed}_{environment}_data.npz'
    with np.load(path) as a:
        states=torch.from_numpy(a['states'].copy());actions=torch.from_numpy(a['actions'].copy());x=torch.from_numpy(a['x'].copy());y=torch.from_numpy(a['y'].copy())
    base,_=ctx.templates(seed)
    # Select using observed features only; no query outcomes, target residuals or physics scoring.
    distance=torch.full((len(x),),float('inf'));chosen=[];ix=int((x-x.mean(0)).square().sum(1).argmin())
    for _ in range(50):
        chosen.append(ix);distance=torch.minimum(distance,(x-x[ix]).square().sum(1));distance[chosen]=-1.;ix=int(distance.argmax())
    indices=torch.tensor(chosen)
    with torch.no_grad():nxt=states[indices]+base.normalizer.decode_y(base.network(x[indices])+y[indices])
    support=transition_batch(states[indices],actions[indices],nxt,f'hardening/privileged_development_pool/{seed}/{environment}')
    return support,dict(available=True,privileged=True,labelled_pool_access=len(x),selected_count=50,indices=chosen,
                         note='Not 50-interaction online identification; pre-existing broad development labels')


def A(ctx):
    """Screen development; validate control; freeze; then independent A confirmation."""
    save_json(ctx.run/'optimizer_audit.json',audit(ctx))
    configs={c['name']:c for c in screen()};all_groups=[]
    for seed in ctx.cfg['model_seeds']:
        for env in ctx.cfg['dev_environments']:
            support=support_from_episode(seed,env)
            for step in ctx.cfg['query_times']:
                record=query_record(seed,env,step)
                evaluate(ctx,record,support,[(m,c) for m in METAS for c in screen()],'A12')
            all_groups.append((seed,env,support))
    rows=offline_rows(ctx,'A12');extras={}
    # A3/A4 follow A1/A2, using only those development query scores.
    for method in METAS:
        table=select_offline(rows,method,list(configs));base=configs[table.iloc[0].optimizer] if len(table) else optimizer()
        variants=[]
        for clip in [.25,5.,None]:
            c=dict(base,name=f'clip_{clip}',gradient_clip=clip);variants.append(c)
        variants.append(dict(base,name='relaxed_guards',gradient_clip=5.,max_parameter_delta=10.,step_norm_bound=.5,isolated_diagnostic=True))
        variants.append(dict(base,name='adam_matched_updates',algorithm='adam'))
        extras[method]=variants
    for seed,env,support in all_groups:
        for step in ctx.cfg['query_times']:
            evaluate(ctx,query_record(seed,env,step),support,[(m,c) for m in METAS for c in extras[m]],'A34')
    rows+=offline_rows(ctx,'A34');shortlist={};config_map={}
    for method in METAS:
        byname={**configs,**{c['name']:c for c in extras[method]}}
        valid=[n for n,c in byname.items() if not c['isolated_diagnostic'] and n!='baseline']
        ranking=select_offline(rows,method,valid)
        shortlist[method]=[byname[n] for n in ranking.optimizer.head(2)]
        config_map[method]=byname
        if len(shortlist[method])<2:
            # Failed treatments stay in tables; baseline is the legitimate fallback.
            shortlist[method]+=[optimizer(f'fallback_baseline_{i}') for i in range(2-len(shortlist[method]))]
    save_json(ctx.run/'A_shortlist_development.json',dict(shortlist=shortlist,confirmation_used=False))
    for seed,env,support in all_groups:
        variants=[]
        for method in METAS:
            options=[optimizer(),optimizer('steps_100',steps=100),*shortlist[method]]
            options={digest({k:v for k,v in c.items() if k!='name'}):c for c in options}
            variants += [(method,c) for c in options.values()]
        for k in ctx.cfg['budget_K']:
            for step in ctx.cfg['query_times']:
                evaluate(ctx,query_record(seed,env,step),support.select(torch.arange(k)),variants,'A5')
    dev=[]
    plist=policies(shortlist=shortlist)+diagnostic_policies(extras)
    for seed,env,support in all_groups:
        for trial in ctx.cfg['dev_trials']:
            for protocol in ['snapshot','online']:
                packet=control(ctx,f'A_dev_{seed}_{trial}_{env}_{protocol}',seed,trial,
                    case_for(env,'development'),plist,fixed_support=support if protocol=='snapshot' else None,online=protocol=='online')
                dev += packet['rows']
    selected={};selection=[]
    frame=pd.DataFrame(dev)
    for method in METAS:
        selected[method],decisions=choose_control_setting(frame,method,shortlist[method]);selection+=decisions
    from .timing import deployment_profile
    deployment_profile(ctx,selected)
    # Report is written BEFORE the freeze and before any A confirmation.
    from .report import generate
    generate(ctx,only_A=True,development_only=True,selection_override=dict(selected=selected,selection=selection,confirmation_used=False))
    latency={}
    for method in METAS:
        matching=[]
        for path in (ctx.run/'controls').glob('A_dev_*.json'):
            item=json.loads(path.read_text())
            for row,initial in zip(item['rows'],item.get('initial_fits',[])):
                if initial and row['method']==method and row['optimizer']==selected[method]:matching.append(initial['log']['adaptation_seconds'])
            for update in item.get('updates',[]):
                row=item['rows'][update['row']]
                if row['method']==method and row['optimizer']==selected[method]:matching.append(update['adaptation_seconds'])
        latency[method]=max(1,int(np.ceil(np.quantile(matching,.95)/ctx.cfg['dt']))) if matching else 1
    frozen=dict(selected=selected,selection=selection,thresholds=ctx.cfg['exit_criteria'],delay_steps_p95_development=latency,
        confirmation_used=False,development_controls_sha256=digest(dev),optimizer_audit_sha256=file_hash(ctx.run/'optimizer_audit.json'),
        development_report_sha256=file_hash(ctx.run/'report_A_development.md'),
        protocol_sha256=ctx.digest,mode=ctx.mode)
    freeze_path=ctx.run/'A_selection_frozen.json'
    if freeze_path.exists():
        if json.loads(freeze_path.read_text())!=frozen:raise RuntimeError('Frozen A selection changed on resume')
    else: save_json(freeze_path,frozen)
    save_json(ctx.run/'C_severities_frozen.json',dict(impulse_Ns=ctx.cfg['impulse_levels'],angular_delta_rad_s=ctx.cfg['angular_velocity_levels'],
        external_force_N=ctx.cfg['force_pulse_levels'],duration_steps=ctx.cfg['pulse_steps'],selection='All three predeclared levels retained; no outcome-based severity selection',mode=ctx.mode))
    for seed in ctx.cfg['model_seeds']:
        for env in ctx.cfg['dev_environments']:
            for trial in ctx.cfg['confirmation_trials']:
                support,info=collection(ctx,seed,trial+10000,env)
                for protocol in ['snapshot','online']:
                    case=case_for(env,'confirmation');case['support_collection']=info
                    if protocol=='snapshot' and support is None:
                        ctx.save_job(ctx.run/'controls'/f'A_confirm_{seed}_{trial}_{env}_{protocol}.json',
                            dict(rows=[],unavailable_slots=9,reason='independent 50-support collection failed',case=case,split='confirmation'),unavailable=np.ones(1));continue
                    key=f'A_confirm_{seed}_{trial}_{env}_{protocol}'
                    packet=control(ctx,key,seed,trial,case,
                            policies(selected=selected),fixed_support=support if protocol=='snapshot' else None,online=protocol=='online')
                    if protocol=='snapshot' and packet['rows']:
                        for target in ctx.cfg['confirmation_query_times']:
                            usable=packet['rows'][0]['survival_steps']-int(packet['rows'][0]['failed'])
                            at=min(target,usable)
                            with np.load(ctx.run/'controls'/f'{key}.npz') as a:state=a['states'][0,at].tolist()
                            record=dict(id=f'A_confirm_query_{seed}_{trial}_{env}_{target}',seed=seed,trial=trial,
                                state=state,environment=env,condition=case['name'],task='swingup',horizon=30,step=at,
                                requested_query_step=target,split='confirmation',query_independent_episode=True)
                            variants=[(m,c) for m in METAS for c in [optimizer(),selected[m]]]
                            evaluate(ctx,record,support,variants,'A_confirmation')
    generate(ctx,only_A=True)


def selected(ctx):
    p=ctx.run/'A_selection_frozen.json'
    if not p.exists():raise RuntimeError('Experiment A development/report/freeze gate not completed')
    value=json.loads(p.read_text())
    if value['confirmation_used'] or value['protocol_sha256']!=ctx.digest or value['mode']!=ctx.mode:
        raise RuntimeError('Invalid A selection provenance')
    audit=ctx.run/'completion_A.json'
    if not audit.exists() or json.loads(audit.read_text()).get('selection_sha256')!=file_hash(p):
        raise RuntimeError('A scientific completion/freeze audit missing or changed')
    return value['selected']


def B(ctx):
    chosen=selected(ctx)
    for split,trials in [('development',ctx.cfg['dev_trials']),('confirmation',ctx.cfg['confirmation_trials'])]:
        for seed in ctx.cfg['model_seeds']:
            for env in ctx.cfg['b_environments']:
                for trial in trials:
                    # Reuse the independent Frozen A trajectory; availability of one B support
                    # strategy must not decide whether other strategies can be evaluated.
                    prefix='dev' if split=='development' else 'confirm'
                    query_source=None
                    for protocol in ['snapshot','online']:
                        candidate=ctx.run/'controls'/f'A_{prefix}_{seed}_{trial}_{env}_{protocol}.json'
                        ap=json.loads(candidate.read_text())
                        if ap['rows']:
                            query_source=candidate;break
                    for strategy in ['ordinary','excitation','diverse']:
                        support,info=diverse(ctx,seed,env) if strategy=='diverse' else collection(ctx,seed,trial+20000,env,strategy,split)
                        case=case_for(env,split,f'B_{env}_{strategy}');case.update(support_strategy=strategy,support_information=info,match_key=f'B_{env}')
                        key=f'B_{seed}_{trial}_{env}_{strategy}'
                        if support is None:
                            ctx.save_job(ctx.run/'controls'/f'{key}.json',dict(rows=[],unavailable_slots=9,reason=info['reason'],case=case,split=split),unavailable=np.ones(1));continue
                        base,_=ctx.templates(seed)
                        with torch.no_grad():
                            x=base.normalizer.encode_x(support.features());residual=base.normalizer.encode_y(support.delta())-base.network(x)
                            singular=torch.linalg.svdvals(residual-residual.mean(0));weights=singular/singular.sum().clamp_min(1e-12)
                            residual_rank=float(torch.exp(-(weights*weights.clamp_min(1e-12).log()).sum()))
                        save_json(ctx.run/'support_coverage'/f'{key}.json',dict(**coverage(base,support),collection=info,seed=seed,environment=env,strategy=strategy,
                            residual_std_mean=float(residual.std(0,unbiased=False).mean()),residual_effective_rank=residual_rank,
                            positive_force_fraction=float((support.action>0).float().mean()),negative_force_fraction=float((support.action<0).float().mean()),
                            distinct_forces=len(torch.unique(support.action)),state_span=(support.state.max(0).values-support.state.min(0).values).tolist()))
                        packet=control(ctx,key,seed,trial,case,policies(selected=chosen),fixed_support=support,online=False)
                        if not packet['rows']:continue
                        if query_source is None:raise RuntimeError('B has an eligible start but no independent A Frozen query trajectory')
                        op=json.loads(query_source.read_text())
                        available=op['rows'][0]['survival_steps']-int(op['rows'][0]['failed'])
                        query_step=min(50,available)
                        with np.load(query_source.with_suffix('.npz')) as a:state=a['states'][0,query_step].tolist()
                        record=dict(id=f'B_query_{seed}_{trial}_{env}',seed=seed,trial=trial,state=state,environment=env,
                            condition=f'B_{env}',task='swingup',horizon=30,step=query_step,split=split,
                            query_source=str(query_source.relative_to(ctx.run)))
                        variants=[(m,c) for m in METAS for c in [optimizer(),chosen[m]]]
                        evaluate(ctx,record,support,variants,'B_'+strategy)


def C(ctx):
    chosen=selected(ctx)
    latency=json.loads((ctx.run/'A_selection_frozen.json').read_text())['delay_steps_p95_development']
    if not (ctx.run/'C_severities_frozen.json').exists():raise RuntimeError('Severity freeze missing')
    for split,trials in [('development',ctx.cfg['dev_trials']),('confirmation',ctx.cfg['confirmation_trials'])]:
        for seed in ctx.cfg['model_seeds']:
            for trial in trials:
                for spec in disturbance_cases():
                    case=dict(spec,phase='stable_balance',task='balance',split=split,steps=ctx.cfg['disturbance_steps'])
                    plist=policies(selected=chosen,disabled=True)
                    if spec['kind'] in ['persistent','combined']:
                        plist += [dict(label=f'delayed_{m}',method=m,optimizer=chosen[m],online=True,delay_steps=latency[m]) for m in METAS]
                    control(ctx,f'C_{seed}_{trial}_{case["name"]}',seed,trial,case,
                            plist,online=True)


def D(ctx):
    chosen=selected(ctx)
    for seed in ctx.cfg['model_seeds']:
        for trial in ctx.cfg['confirmation_trials']:
            for spec in unseen_cases():
                case=dict(spec,task='balance' if spec['phase']=='stable_balance' else 'swingup',
                          split='confirmation',steps=ctx.cfg['unseen_steps'],match_key=f'D_{spec["environment"]}_{spec["phase"]}_{spec["sequence"]}')
                control(ctx,f'D_{seed}_{trial}_{case["name"]}',seed,trial,case,policies(selected=chosen,dense=False),online=True)

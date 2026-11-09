"""Experiments1-6 and final confirmation from cached results, including censoring denominators."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import save_json
from .protocol import workload,live_specs,confirmation_specs,shadow_specs,fork_specs,spec,CHECKPOINTS,PREVIOUS


def bootstrap(values,cfg):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return None,None,None
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    draws=x[rng.integers(len(x),size=(cfg['bootstrap_samples'],len(x)))].mean(1)
    return float(x.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))


def as_table(frame):
    return '```csv\n'+frame.to_csv(index=False)+'```'


def load(ctx):
    episode_rows=[];updates=[];snapshot_rows=[];profiles=[];timings=[];checkpoint_rows=[];live_coverage=[]
    for path in sorted((ctx.run/'episodes').glob('*.json')):
        p=json.loads(path.read_text());episode_rows.extend(p['rows'])
        timings.append(dict(key=p['key'],setup=p['setup_seconds'],loop=p['loop_seconds'],wall=p['wall_seconds'],
                            mpc_calls=p['mpc_batch_calls'],mpc_row_calls=p['mpc_row_calls'],actual_interactions=p['actual_interactions']))
        updates.extend([dict(key=p['key'],**u) for u in p['updates']])
        live_coverage.extend([dict(key=p['key'],**u) for u in p.get('coverage',[])])
        for cp in p['checkpoints']:
            for i,r in enumerate(p['rows']):
                checkpoint_rows.append(dict(seed=r['seed'],trial=r['episode_trial'],condition=r['condition'],stage=r.get('stage','unknown'),
                    method=r['method'],regularization=r['regularization'],safety=r['safety'],K=cp['K'],
                    reached=cp['valid'][i],simulated_observation_seconds=cp['simulated_postchange_seconds'],
                    elapsed_wall_seconds=cp['elapsed_loop_wall_seconds'],key=p['key'],row=i,
                    actual_online_tail_success=r['success'],actual_observations_available=r['usable_postchange_observations']))
        # Log censored checkpoints that never occurred because the batch failed early.
        logged={cp['K'] for cp in p['checkpoints']}
        if p['checkpoints']:
            for k in CHECKPOINTS:
                if k in logged:continue
                for i,r in enumerate(p['rows']):
                    checkpoint_rows.append(dict(seed=r['seed'],trial=r['episode_trial'],condition=r['condition'],stage=r.get('stage','unknown'),
                        method=r['method'],regularization=r['regularization'],safety=r['safety'],K=k,reached=False,
                        simulated_observation_seconds=k*.02,elapsed_wall_seconds=None,key=p['key'],row=i,
                        actual_online_tail_success=r['success'],actual_observations_available=r['usable_postchange_observations']))
    for path in sorted((ctx.run/'snapshots').glob('*.json')):
        p=json.loads(path.read_text());snapshot_rows.extend(p['rows']);profiles.extend(p['horizon_profiles'])
    censored=[r for path in (ctx.run/'censoring').glob('*.json') for r in json.loads(path.read_text())['rows']]
    return [pd.DataFrame(x) for x in [episode_rows,updates,snapshot_rows,profiles,timings,checkpoint_rows,censored,live_coverage]]


def paired_snapshot_effects(snap,cfg):
    valid=snap[~snap.censored].copy() if len(snap) else snap
    if valid.empty:return pd.DataFrame()
    keys=['seed','episode_trial','condition','K','collector']
    ref=valid[valid.method=='frozen'][keys+['local_one_step_nmse','normalized_regret','long_recursive_100_nmse']]
    j=valid.merge(ref,on=keys,suffixes=('','_frozen'),validate='many_to_one');rows=[]
    for ids,g in j.groupby(['condition','collector','method','regularization','alpha','K']):
        row=dict(zip(['condition','collector','method','regularization','alpha','K'],ids));row['pairs']=len(g);row['seeds']=g.seed.nunique()
        row['reach_probability']=len(g)/(len(cfg['model_seeds'])*len(cfg['confirmation_trials']))
        for metric in ['local_one_step_nmse','normalized_regret','long_recursive_100_nmse']:
            delta=g[metric]-g[metric+'_frozen'];est,lo,hi=bootstrap(delta.groupby(g.seed).mean(),cfg)
            row[metric+'_difference']=est;row[metric+'_ci_low']=lo;row[metric+'_ci_high']=hi
            row[metric+'_reference_mean']=float(g[metric+'_frozen'].mean())
            relative=(g[metric+'_frozen']-g[metric])/np.maximum(g[metric+'_frozen'].abs(),cfg['usefulness']['baseline_error_floor'])
            row[metric+'_relative_benefit']=float(relative.groupby(g.seed).mean().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def safety_interactions(episodes,cfg):
    live=episodes[episodes.stage.eq('confirmation')].copy() if 'stage' in episodes else pd.DataFrame()
    if live.empty and 'stage' in episodes:live=episodes[episodes.stage.eq('smoke')].copy()
    rows=[]
    if live.empty:return pd.DataFrame()
    for method in ['meta_residual_1step','meta_residual_rollout']:
        for condition,c in live.groupby('condition'):
            keys=['seed','episode_trial']
            frames=[]
            for m,s,label in [('frozen','original','A'),('frozen','margin','B'),(method,'original','C'),(method,'margin','D')]:
                f=c[(c.method==m)&(c.safety==s)&(c.regularization=='existing')]
                frames.append(f[keys+['success','cost','actual_postchange_transitions','usable_postchange_observations']].rename(columns={v:v+'_'+label for v in ['success','cost','actual_postchange_transitions','usable_postchange_observations']}))
            j=frames[0]
            for f in frames[1:]:j=j.merge(f,on=keys,validate='one_to_one')
            if j.empty:continue
            row=dict(method=method,condition=condition,pairs=len(j))
            for metric in ['success','cost','actual_postchange_transitions','usable_postchange_observations']:
                cast={k:j[metric+'_'+k].astype(float) for k in ['A','B','C','D']}
                interaction=cast['D']-cast['C']-cast['B']+cast['A']
                est,lo,hi=bootstrap(interaction.groupby(j.seed).mean(),cfg)
                row.update({metric+'_interaction':est,metric+'_ci_low':lo,metric+'_ci_high':hi})
                for k in cast:row[metric+'_'+k]=float(cast[k].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def classify_recovery(episodes):
    if 'stage' not in episodes:return pd.DataFrame()
    f=episodes[episodes.stage=='oracle_recovery'];rows=[]
    for ident,g in f.groupby('source_case'):
        baseline=g[(g.search=='baseline')&(g.safety=='original')]
        stronger=g[(g.search=='high')|(g.safety=='margin')]
        a=bool(baseline.success.any());b=bool(stronger.success.any())
        rows.append(dict(case=ident,existing_success=a,stronger_or_safer_success=b,
                         category='A: Oracle recovery demonstrated' if a else 'B: only stronger/safer recovery demonstrated' if b else 'C: no recovery demonstrated under tested controllers'))
    return pd.DataFrame(rows)


def fork_effects(episodes,cfg):
    if 'stage' not in episodes:return pd.DataFrame()
    f=episodes[episodes.stage=='snapshot_control'].copy()
    if f.empty:return pd.DataFrame()
    keys=['seed','episode_trial','condition','collector','K']
    ref=f[f.method=='frozen'][keys+['success','cost']]
    j=f.merge(ref,on=keys,suffixes=('','_frozen'),validate='many_to_one');rows=[]
    for ids,g in j.groupby(['condition','collector','method','regularization','alpha','K']):
        row=dict(zip(['condition','collector','method','regularization','alpha','K'],ids));row['pairs']=len(g);row['seeds']=g.seed.nunique()
        row['reach_probability']=len(g)/(len(cfg['model_seeds'])*len(cfg['confirmation_trials']))
        for metric in ['success','cost']:
            delta=g[metric].astype(float)-g[metric+'_frozen'].astype(float)
            est,lo,hi=bootstrap(delta.groupby(g.seed).mean(),cfg);row.update({metric+'_difference':est,metric+'_ci_low':lo,metric+'_ci_high':hi})
            if metric=='cost':row['relative_cost_benefit']=float(((g.cost_frozen-g.cost)/np.maximum(g.cost_frozen.abs(),1)).groupby(g.seed).mean().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def usefulness(snapshot_effects,control_effects,cfg):
    rows=[]
    if snapshot_effects.empty:return pd.DataFrame()
    for ids,g in snapshot_effects.groupby(['condition','collector','method','regularization','alpha']):
        if ids[2] in ['frozen','oracle']:continue
        g=g.sort_values('K');predicate=g.local_one_step_nmse_ci_high<0;decision=g.normalized_regret_ci_high<0
        def first(mask):
            values=g.loc[mask,'K'];return int(values.iloc[0]) if len(values) else None
        row=dict(zip(['condition','collector','method','regularization','alpha'],ids),earliest_detectable_prediction_K=first(predicate),earliest_detectable_decision_K=first(decision))
        qualified=(g.seeds==len(cfg['model_seeds']))&(g.reach_probability>=cfg['usefulness']['minimum_reach_probability'])
        row['earliest_useful_prediction_K']=first(predicate&qualified&(g.local_one_step_nmse_relative_benefit>=cfg['usefulness']['prediction_fraction'])&(g.local_one_step_nmse_reference_mean>=cfg['usefulness']['baseline_error_floor']))
        row['earliest_useful_decision_K']=first(decision&qualified&(g.normalized_regret_relative_benefit>=cfg['usefulness']['decision_fraction']))
        controls=control_effects
        if len(controls):
            for field,value in zip(['condition','collector','method','regularization','alpha'],ids):controls=controls[controls[field]==value]
        sustained=[]
        if len(controls):
            controls=controls.sort_values('K')
            good=((controls.success_ci_low>0)&(controls.success_difference>=cfg['usefulness']['control_success_points']))|((controls.cost_ci_high<0)&(controls.relative_cost_benefit>=cfg['usefulness']['control_cost_fraction'])&(controls.success_difference>=0))
            good &= (controls.seeds==len(cfg['model_seeds']))&(controls.reach_probability>=cfg['usefulness']['minimum_reach_probability'])
            for i in range(len(controls)-1):
                if good.iloc[i] and good.iloc[i+1]:sustained.append(int(controls.K.iloc[i]))
        row['earliest_two_checkpoint_control_evidence_K']=sustained[0] if sustained else None
        row['interpretation']='Exploratory seed intervals; no universal kick-in step; compare reach-K/censoring and nominal controls.'
        rows.append(row)
    return pd.DataFrame(rows)


def matrix_check(ctx,episodes,snap,censored):
    expected=workload(ctx.cfg);stages=episodes.groupby('stage').size().to_dict() if 'stage' in episodes else {}
    unavailable=sum(3 for _ in (ctx.run/'recovery_unavailable').glob('*.json'))
    ineligible=sum(len(live_specs()) for _ in (ctx.run/'ineligible').glob('*.json'))
    live=stages.get('development',0)+stages.get('confirmation',0)
    checks=dict(live=live+ineligible==expected['live_episodes'],
                snapshot_controls=stages.get('snapshot_control',0)+len(censored)==expected['snapshot_control_records'],
                snapshot_metrics=len(snap)==expected['snapshot_ranking_records'],
                fixed_confirmation=stages.get('fixed_confirmation',0)==expected['fixed_confirmation_episodes'],
                failure_replays=stages.get('oracle_recovery',0)+unavailable==expected['oracle_failure_replay_records'])
    planned_live=set()
    for phase,trials in [('development',ctx.cfg['development_trials']),('confirmation',ctx.cfg['confirmation_trials'])]:
        for seed in ctx.cfg['model_seeds']:
            for trial in trials:
                for case in ctx.cfg['conditions']:
                    for s in live_specs():planned_live.add((phase,seed,trial,case['name'],s['id']))
    live_frame=episodes[episodes.stage.isin(['development','confirmation'])] if 'stage' in episodes else pd.DataFrame()
    actual_live=set(tuple(x) for x in live_frame[['stage','seed','episode_trial','condition','id']].to_numpy()) if len(live_frame) else set()
    for p in (ctx.run/'ineligible').glob('*.json'):
        data=json.loads(p.read_text())
        actual_live.update((data['stage'],data['seed'],data['trial'],data['case']['name'],s['id']) for s in live_specs())
    checks['live_identities']=actual_live==planned_live
    planned_fixed={(seed,trial,c['name'],s['id']) for seed in ctx.cfg['model_seeds'] for trial in ctx.cfg['fixed_confirmation_trials'] for c in ctx.cfg['fixed_conditions'] for s in confirmation_specs()}
    fixed=episodes[episodes.stage.eq('fixed_confirmation')]
    actual_fixed=set(tuple(x) for x in fixed[['seed','episode_trial','condition','id']].to_numpy()) if len(fixed) else set()
    checks['fixed_identities']=actual_fixed==planned_fixed
    missing_snapshot_ids=[];missing_fork_ids=[]
    selection=ctx.run/'selection_locked.json'
    if selection.exists():
        chosen=json.loads(selection.read_text())['conditions']
        planned_snap={(seed,trial,c,collector,k,s['id']) for seed in ctx.cfg['model_seeds'] for trial in ctx.cfg['confirmation_trials'] for c in chosen for collector in ['original','margin'] for k in CHECKPOINTS for s in shadow_specs()}
        actual_snap=set(tuple(x) for x in snap[['seed','episode_trial','condition','collector','K','id']].to_numpy()) if len(snap) else set()
        checks['snapshot_identities']=actual_snap==planned_snap
        planned_fork={(seed,trial,c,collector,s['K'],s['id']) for seed in ctx.cfg['model_seeds'] for trial in ctx.cfg['confirmation_trials'] for c in chosen for collector in ['original','margin'] for s in fork_specs()}
        fork=episodes[episodes.stage.eq('snapshot_control')]
        actual_fork=set(tuple(x) for x in fork[['seed','episode_trial','condition','collector','K','id']].to_numpy()) if len(fork) else set()
        if len(censored):actual_fork.update(tuple(x) for x in censored[['seed','episode_trial','condition','collector','K','id']].to_numpy())
        checks['fork_identities']=actual_fork==planned_fork
        missing_snapshot_ids=list(planned_snap-actual_snap)[:20];missing_fork_ids=list(planned_fork-actual_fork)[:20]
    else:checks['selection_lock']=False
    result=dict(complete=all(checks.values()),checks=checks,expected=expected,actual_stages=stages,
                censored_snapshot_controls=len(censored),ineligible_live_records=ineligible,unavailable_recovery_records=unavailable)
    result['missing_snapshot_examples']=missing_snapshot_ids;result['missing_fork_examples']=missing_fork_ids
    save_json(ctx.run/'completion_registry.json',result);return result


def plots(ctx,episodes,snap,effects,interactions):
    folder=ctx.run/'figures';folder.mkdir(exist_ok=True)
    def save(name):plt.tight_layout();plt.savefig(folder/name,dpi=140);plt.close()
    if len(snap) and 'censored' in snap:
        valid=snap[~snap.censored]
        for metric in ['local_one_step_nmse','normalized_regret','long_recursive_100_nmse','correction_worsens_prediction_fraction']:
            fig,ax=plt.subplots(figsize=(9,4))
            for (m,r,a),g in valid.groupby(['method','regularization','alpha']):
                t=g.groupby('K')[metric].median();ax.plot(t.index,t.values,'o-',label=f'{m}/{r}/a{a}')
            ax.set_xlabel('Actual post-change support observations K');ax.set_ylabel(metric);ax.set_xscale('symlog',linthresh=3);ax.legend(fontsize=6,ncol=2)
            save(f'exp1_4_{metric}.png')
        fig,ax=plt.subplots(figsize=(9,4))
        coverage=valid[(valid.method=='ridge')&(valid.alpha==1)]
        for collector,g in coverage.groupby('collector'):ax.scatter(g.coverage_feature_variance,g.normalized_regret,s=8,alpha=.4,label=collector)
        ax.set_xlabel('Observed support feature variance');ax.set_ylabel('Planning regret');ax.legend();save('exp6_coverage_regret.png')
        fig,ax=plt.subplots(figsize=(9,4));s=valid[(valid.method=='ridge')&(valid.alpha==1)&valid.K.isin([5,10,20,50])]
        for r,g in s.groupby('regularization'):
            t=g.groupby('K').coefficient_norm.median();ax.plot(t.index,t.values,'o-',label=r)
        ax.set_xlabel('K');ax.set_ylabel('Ridge coefficient norm');ax.legend();save('exp5_regularization.png')
    if len(effects):
        fig,ax=plt.subplots(figsize=(10,4))
        for (m,r,a),g in effects.groupby(['method','regularization','alpha']):
            t=g.groupby('K').success_difference.mean();ax.plot(t.index,t.values,'o-',label=f'{m}/{r}/a{a}')
        ax.set_xlabel('K at frozen snapshot');ax.set_ylabel('Paired control success difference vs Frozen');ax.legend(fontsize=6,ncol=2);save('exp1_snapshot_control.png')
    if len(episodes) and 'stage' in episodes:
        live=episodes[episodes.stage.isin(['confirmation','smoke'])]
        fig,ax=plt.subplots(figsize=(10,4))
        if len(live):live.groupby(['method','safety']).success.mean().unstack().plot.bar(ax=ax)
        ax.set_ylabel('Actual online success fraction');save('exp3_safety_adaptation.png')
        # Separate observed phenotype examples from causal claims.
        reps=[]
        for (method,failed),g in live.groupby(['method','failed']):
            row=g.sort_values('cost',ascending=False).iloc[0]
            with np.load(ctx.run/'episodes'/f'{row.key}.npz') as data:
                i=int(row.row);end=int(row.survival_steps);s=data['states'][i,:end+1];a=data['actions'][i,:end,0];p=data['predictions'][i,:end];correction=data['corrections'][i,:end]
            fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
            axes[0].plot(s[:,0],label='cart');axes[0].plot(np.arctan2(np.sin(s[:,2]),np.cos(s[:,2])),label='wrapped angle');axes[0].plot(np.arange(1,len(s)),p[:,0],':',label='predicted cart');axes[0].legend()
            axes[1].plot(a,label='action');axes[1].plot(s[:,3],label='omega');axes[1].legend()
            axes[2].plot(np.linalg.norm(correction,axis=1));axes[2].set_ylabel('Correction norm');axes[2].set_xlabel('Post-change control steps')
            name=f'case_{method}_failed{failed}.png';save(name);reps.append(dict(figure=name,key=row.key,row=int(row.row),condition=row.condition,failure_type=row.failure_category))
        save_json(ctx.run/'case_studies.json',reps)


def additional_results(ctx,ep,snap,live_coverage):
    """Direct trust inversions, equal-data Ridge ablation, fixed interactions and subsequent residuals."""
    valid=snap[~snap.censored].copy() if len(snap) else snap
    for name,df in [('experiment1_learning_curves',valid.groupby(['condition','collector','method','regularization','alpha','K']).agg(
             reached=('K','size'),prediction=('local_one_step_nmse','mean'),regret=('normalized_regret','median'),
             correction_harm=('correction_worsens_prediction_fraction','mean'),action_harm=('preferred_action_worse_than_frozen','mean')).reset_index() if len(valid) else pd.DataFrame())]:df.to_csv(ctx.run/f'{name}.csv',index=False)
    inversions=pd.DataFrame();regularization=[];fixed_interactions=[];coverage_quality=[]
    if len(valid):
        keys=['seed','episode_trial','condition','collector','K']
        ref=valid[valid.method=='frozen'][keys+['local_one_step_nmse']]
        pred=valid.merge(ref,on=keys,suffixes=('','_frozen'),validate='many_to_one')
        forks=ep[ep.stage.eq('snapshot_control')]
        if len(forks):
            fref=forks[forks.method=='frozen'][keys+['success','cost']]
            f=forks.merge(fref,on=keys,suffixes=('','_frozen'),validate='many_to_one')
            columns=keys+['method','regularization','alpha','success','cost','success_frozen','cost_frozen']
            inversions=pred.drop(columns=['cost']).merge(f[columns],on=keys+['method','regularization','alpha'],validate='one_to_one')
            inversions['prediction_improved']=inversions.local_one_step_nmse<inversions.local_one_step_nmse_frozen
            inversions['control_cost_worsened']=inversions.cost>inversions.cost_frozen
            inversions['success_lost']=inversions.success_frozen&~inversions.success
            inversions['prediction_better_control_worse']=inversions.prediction_improved&inversions.control_cost_worsened
        r=valid[valid.method.eq('ridge')&valid.alpha.eq(1)&valid.K.isin(ctx.cfg['regularization_K'])]
        j=r[r.regularization.eq('count_scaled')].merge(r[r.regularization.eq('existing')],on=keys,suffixes=('_scaled','_old'),validate='one_to_one')
        for ids,g in j.groupby(['condition','collector','K']):
            row=dict(zip(['condition','collector','K'],ids),pairs=len(g))
            for metric in ['local_one_step_nmse','long_recursive_100_nmse','normalized_regret','coefficient_norm','condition_number']:
                diff=g[metric+'_scaled']-g[metric+'_old'];est,lo,hi=bootstrap(diff.groupby(g.seed).mean(),ctx.cfg)
                row.update({metric+'_difference':est,metric+'_ci_low':lo,metric+'_ci_high':hi})
            regularization.append(row)
        # Oracle-selected alpha is query-label hindsight, explicitly not a deployable gate.
        choices=[]
        for ids,g in valid.groupby(['seed','episode_trial','condition','collector','K']):
            neutral=g[g.method=='frozen'].iloc[0]
            for method in ['ridge','meta_residual_1step','meta_residual_rollout']:
                options=g[(g.method==method)&g.regularization.eq('existing')]
                if options.empty:continue
                best=options.sort_values('preferred_true_cost').iloc[0]
                alpha=0. if neutral.preferred_true_cost<=best.preferred_true_cost else float(best.alpha)
                choices.append(dict(zip(keys,ids),method=method,oracle_selected_alpha=alpha,deployable=False,criterion='future Oracle true cost; diagnostic only'))
        pd.DataFrame(choices).to_csv(ctx.run/'oracle_alpha_diagnostic.csv',index=False)
    fixed=ep[ep.stage.eq('fixed_confirmation')]
    for (condition,method),g in fixed.groupby(['condition','method']):
        if method in ['frozen','oracle']:continue
        for interaction in ['safety_x_support','blend_x_support']:
            settings=[('original','consecutive',1.),('margin','consecutive',1.),('original','state_action_diverse',1.),('margin','state_action_diverse',1.)] if interaction=='safety_x_support' else [('original','consecutive',1.),('original','consecutive',.5),('original','state_action_diverse',1.),('original','state_action_diverse',.5)]
            frames=[]
            for label,(safe,support,alpha) in zip('ABCD',settings):
                f=g[g.safety.eq(safe)&g.support.eq(support)&g.alpha.eq(alpha)]
                frames.append(f[['seed','episode_trial','success','cost']].rename(columns={'success':'success_'+label,'cost':'cost_'+label}))
            j=frames[0]
            for f in frames[1:]:j=j.merge(f,on=['seed','episode_trial'],validate='one_to_one')
            if j.empty:continue
            row=dict(condition=condition,method=method,interaction=interaction,pairs=len(j))
            for metric in ['success','cost']:
                delta=j[metric+'_D'].astype(float)-j[metric+'_C'].astype(float)-j[metric+'_B'].astype(float)+j[metric+'_A'].astype(float)
                est,lo,hi=bootstrap(delta.groupby(j.seed).mean(),ctx.cfg);row.update({metric+'_interaction':est,metric+'_ci_low':lo,metric+'_ci_high':hi})
            fixed_interactions.append(row)
    if len(live_coverage):
        # Future data below are evaluation targets only; fitting uses the recorded prefix ending atK.
        for key,g in live_coverage.groupby('key'):
            packet=json.loads((ctx.run/'episodes'/f'{key}.json').read_text());seed=packet['rows'][0]['seed'];base,_=ctx.templates(seed)
            with np.load(ctx.run/'episodes'/f'{key}.npz') as data:
                for _,cov in g.iterrows():
                    i=int(cov.row);k=int(cov.K);row=packet['rows'][i];end=min(k+5,row['survival_steps'])
                    result=cov.to_dict();result['subsequent_steps_available']=max(0,end-k)
                    if end>k:
                        error=data['predictions'][i,k:end]-data['states'][i,k+1:end+1];error[:,2]=np.arctan2(np.sin(error[:,2]),np.cos(error[:,2]))
                        result['following5_nmse']=float(np.mean(np.square(error/base.normalizer.y_scale.numpy())))
                    result['remaining_actual_survival']=max(0,row['survival_steps']-k)
                    result['subsequent_updates_may_occur']=True
                    coverage_quality.append(result)
    for name,frame in [('prediction_control_inversions',inversions),('ridge_regularization_paired',pd.DataFrame(regularization)),
                        ('fixed_confirmation_interactions',pd.DataFrame(fixed_interactions)),('online_coverage_subsequent_quality',pd.DataFrame(coverage_quality))]:
        frame.to_csv(ctx.run/f'{name}.csv',index=False)
    return dict(regularization=pd.DataFrame(regularization),inversions=inversions,fixed_interactions=pd.DataFrame(fixed_interactions),coverage=pd.DataFrame(coverage_quality))


def generate(ctx,smoke=False):
    ep,updates,snap,profiles,timing,checkpoints,censored,live_coverage=load(ctx)
    if ep.empty:raise RuntimeError('No cached episodes to report')
    for name,df in [('episodes',ep),('online_updates',updates),('snapshot_metrics',snap),('horizon_trust_profiles',profiles),
                    ('execution_throughput',timing),('observation_checkpoints',checkpoints),('censored_forks',censored),('online_observation_coverage',live_coverage)]:df.to_csv(ctx.run/f'{name}.csv',index=False)
    se=paired_snapshot_effects(snap,ctx.cfg);fe=fork_effects(ep,ctx.cfg);inter=safety_interactions(ep,ctx.cfg);recovery=classify_recovery(ep)
    earliest=usefulness(se,fe,ctx.cfg)
    for name,df in [('snapshot_paired_effects',se),('fork_control_paired_effects',fe),('safety_adaptation_interactions',inter),('oracle_recoverability',recovery),('adaptation_usefulness',earliest)]:df.to_csv(ctx.run/f'{name}.csv',index=False)
    registry=matrix_check(ctx,ep,snap,censored)
    ep.groupby(['stage','condition','method','regularization','safety','alpha'],dropna=False).agg(episodes=('success','size'),success=('success','mean'),cost=('cost','mean'),
       penalty_free_cost=('cost_without_failure_penalty','mean'),rail_rate=('rail_failure','mean'),median_postchange_steps=('actual_postchange_transitions','median')).to_csv(ctx.run/'aggregate_results.csv')
    ep.groupby(['seed','stage','condition','method','regularization','safety','alpha'],dropna=False).agg(success=('success','mean'),cost=('cost','mean')).to_csv(ctx.run/'per_seed_results.csv')
    reach=checkpoints.groupby(['stage','condition','method','regularization','safety','K']).reached.agg(['mean','size']).reset_index() if len(checkpoints) else pd.DataFrame()
    reach.to_csv(ctx.run/'reach_K_probability.csv',index=False)
    extra=additional_results(ctx,ep,snap,live_coverage)
    plots(ctx,ep,snap,fe,inter)
    # Reuse wider-alpha evidence explicitly; never pool its outcomes into new held-out estimates.
    old=pd.read_csv(PREVIOUS/'action_ranking_results.csv');old.groupby(['split','method','alpha']).agg(states=('state_id','size'),median_regret=('normalized_regret','median')).to_csv(ctx.run/'reused_prior_alpha_evidence.csv')
    text='# Final CartPole V2 online validation\n\n'+('SMOKE ONLY; no scientific validation.\n\n' if smoke else '')
    text+=f'Matrix complete: {registry["complete"]}. Checkpoints/model weights unchanged; no neural offline training.\n\n'
    text+='All observations are actual non-failing past transitions. Snapshot counterfactuals freeze the adapter atK and share initial states/actions; actual online trajectories continue updating. These are separate estimands. CensoredK values are not imputed as model failures or successes.\n\n'
    text+='Nominal Frozen warmup is shared across methods before the change; all methods then control independently from the same state. Condition/event comparisons therefore concern post-change recovery, not differences in nominal pre-change controllers.\n\n'
    text+='##1. How many observations before useful adaptation?\n\n'+as_table(earliest)+'\n\nNo universal kick-in step is inferred. Practical usefulness requires five seed clusters and at least80% reach-K: a5% prediction/decision improvement with a favorable seed interval, or10points success/10%cost improvement at two evaluated checkpoints. Detectable changes are reported separately. A benefit atK0 is an initialization advantage, not learning from observations. These criteria are exploratory, predeclared in config_locked.json, and require nominal controls.\n\n'
    text+='##2. Does useful adaptation occur before failure?\n\n'+as_table(reach)+'\n\nCompare conditional snapshot effects with these unconditional reach-K denominators; survivor-only results overestimate operational usefulness.\n\n'
    text+='##3. Which changes are Oracle recoverable?\n\n'+as_table(recovery)+'\n\nCategoryC means no recovery demonstrated by tested controllers, never mathematical impossibility. Development gate selections and unvalidated cases are in selection_locked.json.\n\n'
    text+='##4. Does conservative control help adaptation?\n\n'+as_table(inter)+'\n\nInteraction is D−C−B+A; safety alone and the complete Meta-prior-plus-update policy are explicit controls. This is not an isolated comparison of gradient updates versus an unadapted Meta prior. Static margin is empirical, not a formal safety certificate. Feature variance/conditioning compare the excitation cost of conservative behavior.\n\n'
    text+='##5. When does an adapter hurt?\n\nSee snapshot_metrics.csv for local correction harm, action regret, normalized residuals, and horizon profiles; fork_control_paired_effects.csv for control harm. Oracle-minimum alpha in offline truth is diagnostic only and is never a deployed rule. Prior wider-alpha data are reused separately.\n\n'
    if len(extra['inversions']):
        harm=extra['inversions'].groupby(['method','regularization','alpha']).agg(pairs=('K','size'),prediction_better_control_worse=('prediction_better_control_worse','mean'),success_lost=('success_lost','mean')).reset_index()
        text+=as_table(harm)+'\n\nThese are conditional snapshot comparisons, not an automatic deployable gating rule.\n\n'
    text+='##6. Does Ridge sample-count scaling matter?\n\nExisting sum loss uses lambda0.1; scaled mean loss uses lambda0.1/20, equivalent sum penalty0.1*K/20. They match atK20. Compare same supports/query states in snapshot_metrics.csv forK5/10/20/50; do not assume shrinkage explains degradation.\n\n'
    text+=as_table(extra['regularization'])+'\n\nSame-state normalized-minus-existing differences above use independent seed-cluster intervals. Full counterfactual control differences are also in fork_control_paired_effects.csv; predictor improvements alone are not control evidence.\n\n'
    text+='##7. Is online diversity limiting?\n\nCoverage uses only the actual prefix atK: action/angle/velocity spread, feature rank, regularized conditioning, and query distance. Horizon-trust profiles separate extrapolation from support. Coverage/error correlations are diagnostic, not causal. Fixed-support confirmation costs1024 passive-pool interactions per seed/episode/condition, separately from50 fitting observations.\n\n'
    if len(extra['coverage']):
        pairs=[]
        for ids,g in extra['coverage'].groupby(['method','regularization','safety']):
            if 'following5_nmse' in g:
                matrix=g[['feature_variance','query_support_distance_median','following5_nmse']].rank().corr()
                pairs.append(dict(zip(['method','regularization','safety'],ids),variance_error_spearman=matrix.loc['feature_variance','following5_nmse'],distance_error_spearman=matrix.loc['query_support_distance_median','following5_nmse']))
        text+=as_table(pd.DataFrame(pairs))+'\n\nRepeatedK and controller visits are dependent; no causal interpretation or pooled-observation p-value is justified.\n\n'
    text+='##8. What supports moving toV3?\n\nRequire reproducible held-out benefits before failure on Oracle-recoverable cases. Distinguish missing observation time, inadequate support, harmful corrections, and controller limitations. No architecture or retraining is selected from pooled prediction loss alone.\n\n'
    text+='##9. Highest-priority architectural change\n\nUse the strongest confirmed conditional failure mechanism: coverage-sensitive trust if correction harm persists with sufficient survival; improved temporal identification if information arrives before trust improves. If Oracle still fails, resolve the controller benchmark first. No new gate or neural model was trained in this phase.\n\n'
    text+='## Limitations and reproducibility\n\nFive pretrained seeds, noiseless simulator, short episodes; paired bootstrap resamples five seed means. RepeatedK snapshots are dependent. MPC batch timing is not a standalone real-time guarantee. Shared warmup and counterfactual frozen snapshots are labeled. Cached outputs are atomic/config hashed; source/checkpoint hashes must match for resume.\n'
    for number in range(1,10):text=text.replace(f'##{number}.',f'## {number}.')
    text+='\n\n## Fixed held-out confirmation interactions\n\n'+as_table(extra['fixed_interactions'])+'\n\nInteraction uses four matched cells; benefits are not assumed additive. The complete fixed-control tables include nominal/Oracle references.\n'
    (ctx.run/'report.md').write_text(text,encoding='utf-8')
    save_json(ctx.run/'experiment_outputs.json',dict(experiments_1_to_6=True,held_out_confirmation=True,smoke=smoke))
    if not smoke and not registry['complete']:raise RuntimeError('Required records incomplete; see completion_registry.json')
    return registry

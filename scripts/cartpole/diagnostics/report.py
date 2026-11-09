"""Evidence-linked reports and plots from cached measurements; no model training."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .context import save_json
from .protocol import FAMILIES,METHODS,applicable,development_settings


def load_tables(ctx):
    control=[];replay=[];search=[];support=[];plans=[]
    for path in sorted((ctx.run/'control').glob('*.json')):
        packet=json.loads(path.read_text())
        for i,row in enumerate(packet['rows']):control.append(dict(row,group=packet['key'],row=i))
        with np.load(path.with_suffix('.npz')) as data:
            for i,row in enumerate(packet['rows']):
                for j,t in enumerate(data['plan_steps']):
                    if t>=row['survival_steps']:continue
                    costs=data['plan_costs'][i,j];clear=data['plan_clearances'][i,j]
                    truth=data['plan_true'][i,j];pred=data['plan_states'][i,j]
                    err=pred-truth;err[:,2]=np.arctan2(np.sin(err[:,2]),np.cos(err[:,2]))
                    r=dict(phase=row['phase'],seed=row['seed'],condition=row['condition'],trial=row['trial'],id=row['id'],
                           method=row['method'],family=row['family'],step=int(t),predicted_cost=float(costs[0]),true_cost=float(costs[1]),
                           predicted_clearance=float(clear[0]),actual_clearance=float(clear[1]),
                           false_safe=bool(clear[0]>=0 and clear[1]<0),
                           recursive_5_nmse=float(np.mean(np.square(err[:5]/np.array([2.4,3,np.pi,4])))),
                           recursive_15_nmse=float(np.mean(np.square(err[:15]/np.array([2.4,3,np.pi,4])))),
                           recursive_30_nmse=float(np.mean(np.square(err/np.array([2.4,3,np.pi,4])))))
                    plans.append(r)
    for path in sorted((ctx.run/'replay').glob('*.json')):
        p=json.loads(path.read_text());replay.extend(p['rows'])
        search.extend([dict(s,state_id=p['record']['id'],split=p['record']['split'],seed=p['record']['seed'],category=p['record']['category']) for s in p['search']])
    for path in sorted((ctx.run/'support_replay').glob('*.json')):
        support.extend(json.loads(path.read_text())['rows'])
    return [pd.DataFrame(v) for v in [control,replay,search,support,plans]]


def paired_effects(rows):
    if rows.empty:return pd.DataFrame()
    keys=['phase','seed','condition','trial','method']
    baseline=rows[rows.family=='baseline']
    joined=rows.merge(baseline[keys+['success','cost','cost_without_failure_penalty','rail_failure','one_step_nmse']],on=keys,suffixes=('','_baseline'),validate='many_to_one')
    for col in ['success','cost','cost_without_failure_penalty','rail_failure','one_step_nmse']:
        joined[col+'_difference']=joined[col].astype(float)-joined[col+'_baseline'].astype(float)
    rng=np.random.default_rng(4123);out=[]
    for ids,group in joined.groupby(['phase','condition','family','method','id']):
        for metric in ['success','cost','cost_without_failure_penalty','rail_failure','one_step_nmse']:
            values=group.groupby('seed')[metric+'_difference'].mean().to_numpy()
            samples=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
            out.append(dict(zip(['phase','condition','family','method','id'],ids),metric=metric,
                            paired_difference=float(values.mean()),ci_low=float(np.quantile(samples,.025)),
                            ci_high=float(np.quantile(samples,.975)),seeds=len(values),episodes=len(group)))
    return pd.DataFrame(out)


def completion(ctx,rows,replay):
    expected=[]
    for seed in ctx.cfg['seeds']:
        for c in ctx.cfg['conditions']:
            for f in FAMILIES:
                if not applicable(c,f):continue
                for s in development_settings(f):
                    for trial in ctx.cfg['development_trials']:expected.append(('development',seed,c['name'],trial,s['id']))
    lock=ctx.run/'selection_locked.json'
    if lock.exists():
        choices=json.loads(lock.read_text())['settings']
        for seed in ctx.cfg['seeds']:
            for c in ctx.cfg['conditions']:
                for options in choices[c['name']].values():
                    for s in options:
                        for trial in ctx.cfg['confirmation_trials']:expected.append(('confirmation',seed,c['name'],trial,s['id']))
    actual=set(tuple(x) for x in rows[['phase','seed','condition','trial','id']].to_numpy()) if len(rows) else set()
    if len(actual)!=len(rows):raise AssertionError('Duplicate control identities')
    manifest=ctx.run/'replay_state_manifest.json'
    states=json.loads(manifest.read_text())['states'] if manifest.exists() else []
    missing=[x for x in expected if x not in actual]
    replay_missing=[r['id'] for r in states if r['id'] not in set(replay.state_id)] if len(replay) else [r['id'] for r in states]
    coverage=all((ctx.run/'coverage'/f'{r["id"]}.json').exists() for r in states)
    support_missing=[]
    for r in states:
        applicable_state=r['source']['protocol']!='online' and any(c['environment']==r['source']['environment'] and not c['online'] and c['task']=='swingup' for c in ctx.cfg['conditions'])
        if applicable_state and not (ctx.run/'support_replay'/f'{r["id"]}.json').exists():support_missing.append(r['id'])
    audit=ctx.run/'termination_audit.json'
    audit_complete=audit.exists() and json.loads(audit.read_text())['episodes_audited']==26625
    complete=bool(lock.exists() and not missing and not replay_missing and not support_missing and coverage and states and audit_complete)
    result=dict(complete=complete,expected_control_rows=len(expected),actual_control_rows=len(rows),missing_control=missing,
                missing_replay=replay_missing,missing_support_replay=support_missing,coverage_complete=coverage,audit_complete=audit_complete)
    save_json(ctx.run/'completion_registry.json',result)
    return result


def plot_families(ctx,rows,replay,search,coverage):
    figdir=ctx.run/'figures';figdir.mkdir(exist_ok=True)
    def finish(name):
        plt.tight_layout();plt.savefig(figdir/name,dpi=125);plt.close()
    for family in ['blend','search','support','rail','cost']:
        # Balanced development curves show all screened choices. Held-out effects are paired separately.
        phase='development' if (rows.phase=='development').any() else 'smoke'
        sub=rows[rows.family.isin([family,'baseline']) & (rows.phase==phase)]
        if sub.empty:continue
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for method in METHODS:
            s=sub[sub.method==method]
            if not len(s):continue
            if family=='blend':column='alpha'
            elif family=='search':column='search'
            elif family=='support':
                s=s.copy();s['coverage_K']=s.support+'_'+s.K.astype(str);column='coverage_K'
            else:column='safety' if family=='rail' else 'cost'
            # cost is both a setting and numeric outcome: settings are named cost_variant below.
            if family=='cost':column='cost_variant'
            table=s.groupby(column).agg(success=('success','mean'),control_cost=('cost','mean'))
            x=table.index.astype(str)
            axes[0].plot(x,table.success,'o-',label=method)
            axes[1].plot(x,table.control_cost,'o-',label=method)
        axes[0].set_ylabel('Success fraction');axes[1].set_ylabel('Original V2 cost including failure penalty')
        for ax in axes:ax.set_title(family+'; '+str(sorted(sub.phase.unique())));ax.tick_params(axis='x',rotation=35);ax.legend(fontsize=7)
        finish(f'{family}_comparison.png')
    if len(replay):
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for method in METHODS:
            s=replay[(replay.method==method)&(replay.alpha.isin([0,1]))]
            axes[0].scatter(s.multi_step_nmse,s.normalized_regret,s=12,alpha=.5,label=method)
            axes[1].scatter(s.rank_correlation,s.normalized_regret,s=12,alpha=.5,label=method)
        axes[0].set_xlabel('Multi-step prediction NMSE');axes[1].set_xlabel('Cost-ranking Spearman correlation')
        for ax in axes:ax.set_ylabel('Planning regret / max(abs(best cost),1)');ax.set_yscale('symlog',linthresh=.01);ax.legend(fontsize=7)
        finish('prediction_ranking_regret.png')
        table=replay.groupby(['method','alpha']).agg(regret=('normalized_regret','median'),unsafe=('unsafe_top1','mean'))
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for method in METHODS[1:4]:
            s=table.loc[method];axes[0].plot(s.index,s.regret,'o-',label=method);axes[1].plot(s.index,s.unsafe,'o-',label=method)
        for ax in axes:ax.set_xlabel('Correction alpha');ax.legend()
        axes[0].set_ylabel('Median normalized regret');axes[1].set_ylabel('Unsafe preferred-sequence fraction');finish('offline_blend_ranking.png')
    if len(search):
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for method in METHODS:
            s=search[search.method==method].groupby('search').agg(regret=('regret_vs_shared_pool','median'),time=('runtime_batch_seconds','median')).reindex(['baseline','expanded','high'])
            axes[0].plot(s.index,s.regret,'o-',label=method);axes[1].plot(s.index,s.time,'o-',label=method)
        axes[0].set_ylabel('Median true-cost regret against shared pool');axes[1].set_ylabel('Eager replay batch seconds (five methods)')
        for ax in axes:ax.legend(fontsize=7)
        finish('search_regret_latency.png')
    if len(coverage):
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        data=coverage[coverage.method=='frozen']
        for col,ax in zip(['angle','omega','x','v','commanded_force','support_distance'],axes.flat):
            for source,sub in data.groupby('source'):
                vals=sub[col].replace([np.inf,-np.inf],np.nan).dropna()
                if len(vals):ax.hist(vals,bins=40,density=True,histtype='step',label=source)
            ax.set_xlabel(col);ax.legend(fontsize=5)
        finish('state_action_distributions.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for method in METHODS[:-1]:
            s=coverage[(coverage.method==method)&coverage.source.eq('visited')]
            axes[0].scatter(s.support_distance,s.one_step_nmse,s=4,alpha=.2,label=method)
            axes[1].scatter(s.correction_normalized_rms,s.one_step_nmse,s=4,alpha=.2,label=method)
        axes[0].set_xlabel('Joint distance from support');axes[1].set_xlabel('Normalized correction magnitude')
        for ax in axes:ax.set_ylabel('One-step NMSE');ax.set_yscale('symlog',linthresh=1e-5);ax.legend(fontsize=7)
        finish('error_localization.png')


def representative_plots(ctx,rows,replay):
    reps=[];figdir=ctx.run/'figures'
    # Save each observed mechanism, including successful controls, with actual/predicted/actions/corrections/costs.
    for (method,reason),sub in rows.groupby(['method','failure_category']):
        row=sub.sort_values('cost',ascending=False).iloc[0]
        with np.load(ctx.run/'control'/f'{row.group}.npz') as data:
            i=int(row.row);end=int(row.survival_steps);s=data['states'][i,:end+1];u=data['actions'][i,:end,0]
            pred=data['predictions'][i,:end];correction=data['corrections'][i,:end]
            from adaptive_world_model.control.evaluation_mpc import stage_cost
            import torch
            cost=stage_cost(torch.tensor(s[1:]),torch.tensor(u[:,None]),row.task).numpy()
            available=np.flatnonzero(data['plan_steps']<end)
            last=int(available[-1]) if len(available) else 0
            plan=data['plan_states'][i,last];plan_start=int(data['plan_steps'][last])+1
        fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
        axes[0].plot(s[:,0],label='actual cart');axes[0].plot(np.arctan2(np.sin(s[:,2]),np.cos(s[:,2])),label='actual wrapped angle')
        axes[0].plot(np.arange(1,len(s)),pred[:,0],':',label='one-step predicted cart');axes[0].legend(fontsize=7)
        axes[0].plot(np.arange(plan_start,plan_start+len(plan)),plan[:,0],'--',label='selected-plan predicted cart')
        axes[0].legend(fontsize=7)
        axes[1].plot(s[:,3],label='angular velocity');axes[1].plot(u,label='command');axes[1].legend(fontsize=7)
        axes[2].plot(np.linalg.norm(correction,axis=1));axes[2].set_ylabel('Physical correction norm')
        axes[3].plot(np.cumsum(cost));axes[3].set_ylabel('Cost without failure penalty');axes[3].set_xlabel('Control step')
        fig.suptitle(f'{method}: {reason}; {row.condition}; {row.id}')
        name=f'trajectory_{method}_{reason}.png';plt.tight_layout();plt.savefig(figdir/name,dpi=110);plt.close()
        reps.append(dict(method=method,observed_category=reason,group=row.group,row=int(row.row),figure=name,
                         initial_state=s[0].tolist(),condition=row.condition,setting=row.id,
                         attribution='Observed phenotype. Ranking/support/correction causal attribution requires matched replay or intervention tables.'))
    for category,sub in replay.groupby('category') if len(replay) else []:
        row=sub[sub.method!='oracle'].sort_values('normalized_regret',ascending=False).iloc[0]
        packet=json.loads((ctx.run/'replay'/f'{row.state_id}.json').read_text());idx=next(i for i,r in enumerate(packet['rows']) if r['id']==row.id)
        with np.load(ctx.run/'replay'/f'{row.state_id}.npz') as data:
            pred=data['predicted_costs'][idx];truth=data['true_costs'];chosen=int(np.argmin(pred))
            path=data['predicted_paths'][idx,chosen];actual=data['true_paths'][chosen]
        fig,axes=plt.subplots(1,2,figsize=(12,4));axes[0].scatter(truth,pred,s=4,alpha=.3)
        axes[0].scatter([truth[chosen]],[pred[chosen]],color='red',label='model preferred');axes[0].set_xscale('symlog');axes[0].set_yscale('symlog')
        axes[0].set_xlabel('Oracle cost');axes[0].set_ylabel('Predicted cost');axes[0].legend()
        axes[1].plot(path[:,0],label='predicted cart');axes[1].plot(actual[:,0],label='actual cart');axes[1].axhline(2.4,color='red');axes[1].axhline(-2.4,color='red');axes[1].legend()
        plt.tight_layout();plt.savefig(figdir/f'ranking_{category}.png',dpi=125);plt.close()
    save_json(ctx.run/'representative_episodes.json',reps)


def answers_and_mechanisms(ctx,rows,replay,search,effects):
    confirm=rows[rows.phase=='confirmation']
    ranking=replay[replay.split=='confirmation'] if len(replay) else replay
    mechanisms=[]
    for (method,condition),sub in confirm.groupby(['method','condition']):
        for phenotype,count in sub.failure_category.value_counts().items():
            if phenotype!='success':mechanisms.append(dict(method=method,condition=condition,mechanism=phenotype,count=int(count),denominator=len(sub),
                                                           rate=count/len(sub),evidence='direct observation',causal=False))
    for (method,environment),sub in ranking.groupby(['method','environment']) if len(ranking) else []:
        harmful=(sub.avoidable_unsafe_choice)&(sub.normalized_regret>.1)
        mechanisms.append(dict(method=method,condition=environment,mechanism='unsafe_preferred_action_with_positive_oracle_regret',
                               count=int(harmful.sum()),denominator=len(sub),rate=float(harmful.mean()),
                               evidence='counterfactual Oracle evaluation of identical candidate sequences; not an episode-level attribution',causal=True))
    mechanism=pd.DataFrame(mechanisms)
    if len(mechanism):mechanism.sort_values('rate',ascending=False).to_csv(ctx.run/'failure_mechanism_impact.csv',index=False)
    lines=[]
    questions=[
      'Why does Frozen outperform adapters in some control conditions?',
      'Are adapter predictions causing incorrect action rankings?',
      'Does reduced prediction error correspond to reduced planning regret?',
      'Does partial correction outperform full correction?',
      'Can larger MPC search budgets rescue Meta?',
      'Does increased search expose model exploitation?',
      'Why does Ridge get worse with more observations?',
      'Does support-data diversity improve adaptation?',
      'Are rail-safety or stabilization-cost weaknesses responsible?',
      'Are termination and success scientifically sound?',
      'Which failure mechanisms dominate for each model?',
      'Which finding best informs the next architecture/training change?']
    def evidence(family):
        if effects.empty:return 'No held-out effect measurements yet.'
        s=effects[(effects.phase=='confirmation')&(effects.family==family)&effects.metric.isin(['success','cost'])]
        if s.empty:return 'No held-out measurements yet.'
        success=s[s.metric=='success'].sort_values('paired_difference',ascending=False)
        if len(success):
            best=success.iloc[0];worst=success.iloc[-1]
            summary=(f"Largest measured success change: {best.method}, {best.condition}: {100*best.paired_difference:+.1f} percentage points "
                     f"(seed95% interval {100*best.ci_low:+.1f} to {100*best.ci_high:+.1f}). "
                     f"Smallest: {worst.method}, {worst.condition}: {100*worst.paired_difference:+.1f} points.\n")
        else:summary=''
        return summary+s[['condition','method','id','metric','paired_difference','ci_low','ci_high']].to_csv(index=False)
    base=confirm[confirm.family=='baseline'].groupby('method').agg(success=('success','mean'),cost=('cost','mean'))
    rank=ranking.groupby('method').agg(top1=('top1_agreement','mean'),regret=('normalized_regret','median'),unsafe=('unsafe_top1','mean')) if len(ranking) else pd.DataFrame()
    corr=float(ranking[['multi_step_nmse','normalized_regret']].rank().corr().iloc[0,1]) if len(ranking)>2 else None
    text=[
      'Held-out unchanged-controller performance:\n'+base.to_csv()+'\nCompare shared-state rankings below and paired correction interventions; differences in success alone do not identify a cause.',
      'Held-out identical-candidate counterfactuals:\n'+rank.to_csv()+'\nPositive regret directly measures choosing a worse evaluated sequence. An unsafe choice is avoidable only when a safe alternative exists in the saved pool.',
      f'Held-out pooled Spearman(prediction NMSE, normalized regret)={corr}. Stratified values are in ranking_correlations.csv; correlation is not causation.',
      evidence('blend'),evidence('search'),
      'Per-iteration predicted minima and Oracle-evaluated elite means are saved in search_convergence.csv. Falling predicted cost with rising true cost supports exploitation; search/control effects must agree before calling it the dominant cause.',
      evidence('support')+'\nK20 versus K50 and design conditioning/coefficient norms are in support_ranking_results.csv. Diversity benefit supports an information-coverage explanation; its absence does not prove insufficient expressiveness.',
      evidence('support'),evidence('rail')+'\n'+evidence('cost'),
      json.dumps(json.loads((ctx.run/'termination_audit.json').read_text()),indent=2),
      mechanism.to_csv(index=False)+'\nPhenotypes and counterfactual state-level causes use different denominators; they are ranked separately and are not mutually exclusive episode causes.',
      'Use the largest reproducible held-out paired intervention effect, its seed interval, and its counterfactual ranking evidence. No architecture winner is inferred automatically from pooled success. The next highest-information experiment is an independent oracle-labeled action-ranking confirmation concentrated on the dominant rescued-versus-unrescued failure stratum; no new model training is initiated.'
    ]
    result=dict(zip(questions,text));save_json(ctx.run/'diagnostic_answers.json',result)
    for q,a in result.items():lines.extend([f'### {q}',a])
    return '\n\n'.join(lines)


def generate(ctx,smoke=False):
    started=time.perf_counter()
    rows,replay,search,support,plans=load_tables(ctx)
    # Preserve setting name separately from the numeric control-cost field.
    # IDs encode the original cost variant; numeric cost intentionally remains the common scoring objective.
    if len(rows):rows['cost_variant']=rows.id.str.split('__').str[-1]
    for name,frame in [('control_results',rows),('action_ranking_results',replay),('search_convergence',search),('support_ranking_results',support),('selected_plan_diagnostics',plans)]:
        frame.to_csv(ctx.run/f'{name}.csv',index=False)
    effects=paired_effects(rows);effects.to_csv(ctx.run/'paired_seed_bootstrap.csv',index=False)
    registry=completion(ctx,rows,replay)
    groups=['phase','condition','family','method','id']
    aggregate=rows.groupby(groups).agg(episodes=('success','size'),success=('success','mean'),cost=('cost','mean'),
                                      cost_without_failure_penalty=('cost_without_failure_penalty','mean'),rail_rate=('rail_failure','mean'),
                                      numerical_rate=('numerical_failure','mean'),prediction=('one_step_nmse','mean'),correction=('correction_rms','mean'))
    aggregate.to_csv(ctx.run/'aggregate_results.csv')
    rows.groupby(groups).agg(crossing_time=('first_upright_crossing','median'),crossing_omega=('crossing_omega','median'),
                             max_overshoot=('max_overshoot','median'),stabilization=('stabilization_step','median'),
                             lost_stabilization=('settled_then_lost','mean'),recovery=('recovery_steps','median'),
                             initial_fit_latency=('initial_adaptation_seconds','median'),mpc_batch_latency=('mpc_batch_seconds','median')).to_csv(ctx.run/'stabilization_recovery_latency.csv')
    # Confirmatory intervention impact is measured separately from phenotype prevalence.
    if len(effects):
        impact=effects[(effects.phase=='confirmation')&(effects.metric=='success')].copy()
        impact['positive_interval']=impact.ci_low>0
        impact.sort_values('paired_difference',ascending=False).to_csv(ctx.run/'intervention_impact_ranking.csv',index=False)
    rows.groupby(['seed']+groups).agg(success=('success','mean'),cost=('cost','mean')).to_csv(ctx.run/'per_seed_results.csv')
    correlations=[]
    if len(replay):
        for ids,s in replay.groupby(['split','category','method']):
            for pred in ['one_step_nmse','multi_step_nmse','rank_correlation']:
                correlations.append(dict(zip(['split','category','method'],ids),metric=pred,
                                         spearman=s[[pred,'normalized_regret']].rank().corr().iloc[0,1],states=len(s)))
    pd.DataFrame(correlations).to_csv(ctx.run/'ranking_correlations.csv',index=False)
    covpath=ctx.run/'state_action_distribution.csv';coverage=pd.read_csv(covpath) if covpath.exists() else pd.DataFrame()
    plot_families(ctx,rows,replay,search,coverage);representative_plots(ctx,rows,replay)
    timings=[];update_rows=[]
    for path in (ctx.run/'control').glob('*.json'):
        p=json.loads(path.read_text())
        timings.append(dict(group=p['key'],setup_seconds=p['graph_and_setup_seconds'],loop_seconds=p['loop_seconds'],
                            wall_seconds=p['wall_seconds'],batch=p['batch'],steps=p['steps'],
                            simulated_transitions_per_second=p['batch']*p['steps']/max(p['loop_seconds'],1e-9)))
        update_rows.extend([dict(group=p['key'],**u) for u in p['updates']])
    pd.DataFrame(timings).to_csv(ctx.run/'execution_throughput.csv',index=False)
    pd.DataFrame(update_rows).to_csv(ctx.run/'online_updates.csv',index=False)
    answers=answers_and_mechanisms(ctx,rows,replay,search,effects)
    header=f'# CartPole V2 final control diagnostics\n\nMode: {"SMOKE ONLY, not scientific validation" if smoke else "full"}. Complete matrix: {registry["complete"]}.\n\n'
    header+='No neural offline training or architecture changes. Archived checkpoints/results remain immutable. Development and held-out confirmation use independent trials/support namespaces.\n\n'
    header+='All control scores use the ORIGINAL V2 objective, even when the planner objective is ablated. Failure penalties and penalty-free cost are separate. Rail/nonfinite/numerical-bound/time-limit labels preserve original thresholds.\n\n'
    header+='Candidate-ranking regret is relative to the best evaluated finite shared pool, not a global optimum. Elite-average sequences from all three budgets are explicitly included. True candidate costs include the matching terminal and soft rail terms; episode cost stops at actual failure.\n\n'
    header+='Support diversity uses a precollected1024-transition passive pool for ALL modes. K counts fitting observations;1024 counts pool interactions. Online updates use only the controller\'s own past. Oracle is a privileged reference.\n\n'
    header+='## Aggregate controls\n\n```csv\n'+aggregate.to_csv()+'```\n\n'
    footer='\n\n## Reproduction and limits\n\nRun `.\\.venv\\Scripts\\python.exe scripts\\cartpole\\run_diagnostics.py all`. Resume with `--run-dir <directory>`. Stages: audit, offline, develop, confirm, report. Source hashes must match.\n\nFive seed clusters limit uncertainty precision. Development-selected settings are confirmed without retuning. No observational correlation proves a cause. Oracle-unsolved is not physically impossible. Alpha and controller interventions identify effects only within the targeted conditions.\n'
    (ctx.run/'report.md').write_text(header+answers+footer,encoding='utf-8')
    save_json(ctx.run/'report_timing.json',dict(seconds=time.perf_counter()-started))
    if not smoke and not registry['complete']:raise RuntimeError('Report saved, but required matrix/stages incomplete; see completion_registry.json')
    return registry

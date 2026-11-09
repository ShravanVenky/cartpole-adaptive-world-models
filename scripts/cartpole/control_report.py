"""Report saved V2 control episodes; never runs an optimizer or a model."""
import json,math,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptive_world_model.utils.artifacts import write_json

def generate(run,cfg):
    packets=[json.loads(p.read_text()) for p in sorted((run/'episodes').glob('*.json'))]
    rows=pd.DataFrame([r for p in packets for r in p['rows']])
    rows['postchange_interactions']=np.where(rows.protocol.eq('online'),
        np.maximum(0,rows.survival_steps-np.where(rows.task.eq('balance'),cfg['online_shift_step'],cfg['large_angle_shift_step'])),0)
    rows['maximum_online_support_available']=np.minimum(rows.postchange_interactions,50)
    identity=['seed','trial','environment','task','horizon','protocol','method','budget']
    if rows.duplicated(identity).any():raise AssertionError('Duplicate matched episode identities')
    registry=set()
    for seed in cfg['seeds']:
        for env in cfg['environments']:
            for task in ['balance','swingup']:
                horizons=cfg['horizons']+([cfg['swingup_extra_horizon']] if task=='swingup' else [])
                for h in horizons:
                    for protocol in ['fixed','disturbance','online']:
                        if protocol=='online' and env['name']=='nominal':continue
                        registry.add(f"{protocol}_{task}_H{h}_{env['name']}_seed{seed}")
    actual={p['key'] for p in packets}
    expected=cfg['expected_episode_rows'];complete=actual==registry and len(packets)==cfg['expected_groups'] and len(rows)==expected
    write_json(run/'completion_registry.json',{'complete':complete,'missing_groups':sorted(registry-actual),'unexpected_groups':sorted(actual-registry),
        'expected_groups':len(registry),'actual_groups':len(actual),'expected_rows':expected,'actual_rows':len(rows)})
    rows.to_csv(run/'per_episode_results.csv',index=False)
    groups=['task','protocol','category','horizon','method','budget']
    aggregate=rows.groupby(groups).agg(episodes=('success','size'),success_rate=('success','mean'),
        mean_cost=('cost','mean'),median_cost=('cost','median'),cost_std=('cost','std'),
        median_survival=('survival_steps','median'),median_recovery=('recovery_steps','median'),
        median_stabilization=('stabilization_step','median'),mean_prediction_nmse=('one_step_nmse','mean'),
        median_mpc_seconds=('mpc_seconds_per_step','median')).reset_index()
    aggregate.to_csv(run/'aggregate_results.csv',index=False)
    rows.groupby(['seed']+groups).agg(success_rate=('success','mean'),mean_cost=('cost','mean')).to_csv(run/'per_seed_results.csv')
    keys=['seed','trial','environment','category','task','horizon','protocol']
    frozen=rows[rows.method.eq('frozen')][keys+['cost','success']].rename(columns={'cost':'frozen_cost','success':'frozen_success'})
    paired=rows.merge(frozen,on=keys,validate='many_to_one')
    oracle=rows[rows.method.eq('oracle')][keys+['cost','success']].rename(columns={'cost':'oracle_cost','success':'oracle_success'})
    paired=paired.merge(oracle,on=keys,validate='many_to_one')
    paired['cost_benefit_vs_frozen']=1-paired.cost/paired.frozen_cost
    paired['cost_gap_vs_oracle']=paired.cost-paired.oracle_cost
    paired['success_difference_vs_frozen']=paired.success.astype(float)-paired.frozen_success.astype(float)
    paired.to_csv(run/'paired_results.csv',index=False)
    rng=np.random.default_rng(cfg['bootstrap_seed']);comparisons=[]
    for ids,frame in paired.groupby(['task','protocol','category','horizon','method','budget']):
        byseed=frame.groupby('seed')[['cost_benefit_vs_frozen','success_difference_vs_frozen','cost_gap_vs_oracle']].mean()
        for metric in byseed:
            values=byseed[metric].to_numpy();draws=values[rng.integers(len(values),size=(cfg['bootstrap_samples'],len(values)))].mean(1)
            comparisons.append(dict(zip(['task','protocol','category','horizon','method','budget'],ids))|{'metric':metric,
                'estimate':values.mean(),'ci_low':np.quantile(draws,.025),'ci_high':np.quantile(draws,.975),'seeds':len(values)})
    pd.DataFrame(comparisons).to_csv(run/'paired_seed_bootstrap.csv',index=False)
    updates=[{'group':p['key'],**{k:v for k,v in u.items() if k!='methods'},'adapter_updates':len(u['methods']),
        'latency_seconds':sum(x['adaptation_seconds'] for x in u['methods']),
        'rejected':sum(not x['accepted'] for x in u['methods'])} for p in packets for u in p['updates']]
    pd.DataFrame(updates).to_csv(run/'online_update_interactions.csv',index=False)
    figdir=run/'figures';figdir.mkdir(exist_ok=True)
    styles=['frozen','ridge','meta_residual_1step','meta_residual_rollout','oracle']
    def save(name):plt.tight_layout();plt.savefig(figdir/name,dpi=140);plt.close()
    for task in ['balance','swingup']:
        chosen=rows[(rows.task==task)&(rows.protocol=='fixed')&((rows.budget==50)|rows.method.isin(['frozen','oracle']))]
        for metric in ['success','cost']:
            fig,axes=plt.subplots(1,2,figsize=(13,4))
            for ax,column in zip(axes,['horizon','category']):
                table=chosen.groupby([column,'method'])[metric].mean().unstack().reindex(columns=styles)
                table.plot.bar(ax=ax);ax.set_title(task+' '+metric);ax.set_ylabel('Success fraction' if metric=='success' else 'Mean cost including failure penalty')
                ax.legend(fontsize=7)
            save(f'{task}_{metric}_comparison.png')
        horizons=sorted(rows[rows.task.eq(task)].horizon.unique())
        fig,axes=plt.subplots(2,len(horizons),figsize=(6*len(horizons),8),squeeze=False)
        for col,h in enumerate(horizons):
            for method in styles[1:-1]:
                sub=rows[(rows.task==task)&(rows.protocol=='fixed')&(rows.method==method)&(rows.horizon==h)]
                for ax,metric in zip(axes[:,col],['success','cost']):
                    table=sub.groupby('budget')[metric].mean();ax.plot(table.index,table.values,'o-',label=method)
                    ax.set_xlabel('Support observations');ax.set_ylabel(metric);ax.set_title(f'{task} H={h}');ax.legend(fontsize=7)
        save(f'{task}_adaptation_budget.png')
    prediction_metrics=['one_step_nmse','recursive_5_nmse','recursive_15_nmse','recursive_100_nmse','shared_frozen_trajectory_one_step_nmse']
    corr=[]
    for metric in prediction_metrics:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for task,ax in zip(['balance','swingup'],axes):
            subset=rows[(rows.task==task)&(~rows.method.eq('oracle'))]
            finite=subset[[metric,'cost','success']].replace([np.inf,-np.inf],np.nan).dropna()
            for outcome in ['cost','success']:
                rho=finite[metric].rank().corr(finite[outcome].astype(float).rank())
                corr.append({'task':task,'metric':metric,'outcome':outcome,'spearman':rho,'episodes':len(finite)})
            for method in styles[:-1]:
                sub=subset[subset.method.eq(method)];ax.scatter(sub[metric],sub.cost,s=5,alpha=.15,label=method)
            ax.set_xscale('log');ax.set_yscale('log');ax.set_title(task);ax.set_xlabel(metric);ax.set_ylabel('Cost');ax.legend(fontsize=6)
        save(f'prediction_vs_cost_{metric}.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for task,ax in zip(['balance','swingup'],axes):
            subset=rows[(rows.task==task)&(~rows.method.eq('oracle'))]
            for method in styles[:-1]:
                sub=subset[subset.method.eq(method)]
                ax.scatter(sub[metric],sub.success.astype(float),s=5,alpha=.12,label=method)
            ax.set_xscale('log');ax.set_xlabel(metric);ax.set_ylabel('Success (0/1)');ax.set_title(task);ax.legend(fontsize=6)
        save(f'prediction_vs_success_{metric}.png')
    pd.DataFrame(corr).to_csv(run/'prediction_control_correlations.csv',index=False)
    representatives=[]
    for task in ['balance','swingup']:
        for outcome in [True,False]:
            subset=rows[(rows.task==task)&(rows.success==outcome)&(~rows.method.eq('oracle'))]
            for method in styles[:-1]:
                sub=subset[subset.method.eq(method)]
                if sub.empty:continue
                row=sub.sort_values('cost',ascending=outcome).iloc[0]
                packet=next(p for p in packets if p['key']==f"{row.protocol}_{row.task}_H{row.horizon}_{row.environment}_seed{row.seed}")
                ix=next(i for i,r in enumerate(packet['rows']) if r['method']==row.method and r['budget']==row.budget and r['trial']==row.trial)
                data=np.load(run/'episodes'/(packet['key']+'.npz'));s=data['states'][ix];a=data['actions'][ix,:,0]
                pred=data['one_step_prediction'][ix];fig,axes=plt.subplots(3,1,figsize=(11,8))
                axes[0].plot(s[:,0],label='cart position');axes[0].plot(np.arctan2(np.sin(s[:,2]),np.cos(s[:,2])),label='wrapped pole angle')
                starts=data['forecast_starts'];event=int(row.survival_steps) if row.failed else len(a)
                available=np.where(starts<event)[0]
                if len(available):
                    forecast_ix=available[-1];begin=int(starts[forecast_ix]);forecast=data['forecast_states'][ix,forecast_ix]
                    axes[0].plot(np.arange(begin+1,begin+101),forecast[:,0],'--',label='forecast cart position')
                    axes[0].plot(np.arange(begin+1,begin+101),np.arctan2(np.sin(forecast[:,2]),np.cos(forecast[:,2])),'--',label='forecast angle')
                oracle_ix=next(i for i,r in enumerate(packet['rows']) if r['method']=='oracle' and r['trial']==row.trial)
                axes[0].plot(data['states'][oracle_ix,:,0],':',label='paired oracle cart position');axes[0].legend(fontsize=7)
                axes[1].plot(a);axes[1].set_ylabel('Force command')
                residual=pred-s[1:];residual[:,2]=np.arctan2(np.sin(residual[:,2]),np.cos(residual[:,2]))
                axes[2].plot(np.linalg.norm(residual,axis=1));axes[2].set_ylabel('Wrapped one-step residual norm');axes[2].set_xlabel('Control step')
                fig.suptitle(f'{task} {method} success={outcome} {row.environment} H={row.horizon} K={row.budget}')
                name=f'trajectory_{task}_{method}_{outcome}.png';save(name)
                representatives.append({'artifact':name,'group':packet['key'],'row':ix,'initial_state':s[0].tolist(),
                    'physics':packet['physics'],'outcome':bool(outcome),'plausible_cause':'Compare paired oracle, frozen-shadow errors, visited coverage and rail event; no causal attribution from a single trajectory'})
    write_json(run/'representative_episodes.json',representatives)
    # Online recovery curves use controller-generated trajectories and the exact shift clock.
    for task in ['balance','swingup']:
        curves={}
        for p in packets:
            if not p['key'].startswith('online_'+task):continue
            data=np.load(run/'episodes'/(p['key']+'.npz'));s=data['states']
            good=(np.abs(np.arctan2(np.sin(s[:,:,2]),np.cos(s[:,:,2])))<cfg['success_angle'])&(np.abs(s[:,:,3])<cfg['success_omega'])&(np.abs(s[:,:,0])<=2.4)
            for i,r in enumerate(p['rows']):curves.setdefault((r['method'],r['budget']),[]).append(good[i])
        plt.figure(figsize=(11,4))
        for (method,budget),values in curves.items():plt.plot(np.mean(values,axis=0),label=f'{method} enabled={bool(budget)}')
        plt.axvline(cfg['online_shift_step'] if task=='balance' else cfg['large_angle_shift_step'],color='black',linestyle='--');plt.xlabel('Control step');plt.ylabel('Fraction within upright region');plt.legend(fontsize=6);save(f'online_recovery_{task}.png')
    # Original files are rehashed at completion, rather than trusting weight-freezing alone.
    original=json.loads((run/'preserved_artifact_hashes.json').read_text())
    root=Path(__file__).resolve().parents[2]
    for name,digest in original.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise AssertionError('Original artifact changed: '+name)
    write_json(run/'status.json',{'complete':complete,'phase':'complete' if complete else 'incomplete_matrix','groups':len(packets),'episode_rows':len(rows),'original_artifacts_verified':len(original)})
    from control_diagnostics import diagnostics
    diagnostics(run)
    from control_interpret import interpret
    answers,recommendation=interpret(run,cfg,rows,paired)
    lines=['# CartPole V2 closed-loop control evaluation','',f'Complete: **{complete}**. Groups {len(packets)}/{cfg["expected_groups"]}; episodes {len(rows)}/{expected}. No offline model training.','',
      'One shared CEM planner, task costs, constraints, sampling budget and matched per-trial random numbers. H5/H15 swing-up remains explicitly oracle-horizon-limited; H30 is the shortest validated horizon. Oracle is privileged and receives the current true physics, including immediately after a shift.','',
      'Five independent pretrained seeds, three evaluation episodes per condition. Fixed-budget support is independent and passive. Online support is actual controller-generated post-change data; reset/refit cumulative prefixes5/10/20/50 with the original ten Meta gradient updates. No detector or probes.','',
      'Primary cost includes a100,000 rail/numerical-failure penalty and stops accumulating stage cost after failure. Full-angle simulation continues until rail/numerical failure, then uses an explicitly failed absorbing state. One-step prediction metrics exclude post-failure absorbing transitions.','',
      'Summary (K50 fixed and disturbance; online enabled; Frozen/Oracle unchanged):','']
    chosen=aggregate[((aggregate.protocol=='online')&(aggregate.budget==1))|((aggregate.protocol!='online')&(aggregate.budget==50))|aggregate.method.isin(['frozen','oracle'])]
    lines.append(chosen.to_markdown(index=False) if __import__('importlib').util.find_spec('tabulate') else chosen.to_csv(index=False))
    lines+=['','Paired uncertainty is reported in paired_seed_bootstrap.csv using2,000 resamples of five seed-level paired means; the small number of independent seeds limits precision. Prediction correlations are observational and are in prediction_control_correlations.csv. Shared-Frozen-trajectory prediction errors compare alternative predictors on identical visited states/actions, with their protocol-specific adaptation snapshots.','',
      'All per-episode metrics, initial states, physical configurations, actions, actual states, recursive forecasts and online update counts are saved. Per-episode adaptation_seconds measures the initial fixed-support fit; online fitting latency is in online_update_interactions.csv and complete group wall time includes transfers. postchange_interactions counts all actual transitions until failure/end, separately from the at-most50 support observations. Failed absorbing-state padding is not counted as an interaction. Representative success/failure plots and metadata are in figures/ and representative_episodes.json. Oracle failures establish limitations of this controller/budget/constraints, not physical impossibility.','',
      'Answers to the requested questions:','',*['**'+question+'**\n\n'+answer+'\n' for question,answer in answers.items()],
      'Critical assessment: the test evaluates one fixed shared controller and its oracle-validated costs/budget; failure may arise from optimization under an imperfect model as well as prediction error. The oracle has privileged physics and must not be presented as a learned competitor. Five independent training seeds permit seed-block uncertainty but limit inference precision. There is no observation noise, and long-horizon or severe-shift oracle failures do not prove physical infeasibility.','',
      'Single most informative next experiment: '+recommendation,'',
      'Reproduction: `.venv/Scripts/python.exe scripts/cartpole/control_evaluate.py full --device cuda`. Completed groups are cached. Reporting alone: import control_report.generate and pass this run and config_locked.json. GPU benchmark: throughput_benchmark.json.']
    (run/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'CONTROL_REPORT complete={complete} episodes={len(rows)}',flush=True)

if __name__=='__main__':
    import sys
    run=Path(sys.argv[1]);generate(run,json.loads((run/'config_locked.json').read_text()))

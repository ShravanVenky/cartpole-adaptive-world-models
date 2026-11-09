"""Saved-output-only tables, paired seed uncertainty, plots and auditable conclusions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import save_json, save_csv
from .protocol import METAS, workload


def table(frame,limit=60):
    if frame is None or not len(frame):return 'Not available; this workload has not produced rows.'
    frame=frame.head(limit).copy()
    def cell(v):
        if isinstance(v,(dict,list)):return str(v)
        if v is None or pd.isna(v):return 'N/A'
        return f'{v:.5g}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                     '| '+' | '.join(['---']*len(frame.columns))+' |',
                     *['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def read(ctx):
    controls=[];offline=[];updates=[];forecasts=[]
    for path in sorted((ctx.run/'controls').glob('*.json')):
        p=json.loads(path.read_text());stage=path.stem.split('_')[0]
        controls += [dict(r,stage=stage) for r in p.get('rows',[])]
        updates += [dict(u,key=path.stem,stage=stage,seed=p['seed'],split=p['split']) for u in p.get('updates',[])]
        forecasts += [dict(f,key=path.stem,stage=stage,seed=p['seed'],split=p['split']) for f in p.get('forecasts',[])]
    for path in sorted((ctx.run/'offline').glob('*.json')):
        p=json.loads(path.read_text());offline += [dict(r,tag=p['tag']) for r in p['rows']]
    return pd.DataFrame(controls),pd.DataFrame(offline),pd.DataFrame(updates),pd.DataFrame(forecasts)


def paired(frame,reference='frozen'):
    rows=[]
    if not len(frame):return pd.DataFrame(rows)
    keys=['stage','split','condition','evaluation_protocol','seed','trial']
    baselines=frame[frame.policy==reference]
    rng=np.random.default_rng(704011)
    for policy,g in frame.groupby('policy'):
        if policy==reference:continue
        matched=g.merge(baselines,on=keys,suffixes=('_method','_frozen'),validate='one_to_one')
        for group,q in matched.groupby(['stage','split','condition','evaluation_protocol']):
            for metric in ['success','cost','cost_without_failure_penalty','one_step_nmse','recursive_30_nmse']:
                d=pd.to_numeric(q[metric+'_method'],errors='coerce').astype(float)-pd.to_numeric(q[metric+'_frozen'],errors='coerce').astype(float)
                clusters=pd.DataFrame(dict(seed=q.seed,difference=d)).groupby('seed').difference.mean().dropna().to_numpy()
                if not len(clusters):continue
                boot=clusters[rng.integers(0,len(clusters),(2000,len(clusters)))].mean(1)
                rows.append(dict(stage=group[0],split=group[1],condition=group[2],evaluation_protocol=group[3],
                    policy=policy,reference=reference,metric=metric,pairs=len(q),seed_clusters=len(clusters),difference=float(clusters.mean()),
                    ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    return pd.DataFrame(rows)


def plots(ctx,controls,offline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ctx.run/'figures';folder.mkdir(exist_ok=True)
    names=[]
    def save(name):
        import os
        plt.tight_layout();tmp=folder/(name+'.tmp');plt.savefig(tmp,format='png',dpi=140);plt.close();os.replace(tmp,folder/name);names.append(name)
    if len(offline):
        data=offline[offline.method.isin(METAS)]
        for metric,filename in [('one_step_nmse','optimization_prediction.png'),('planning_regret','optimization_regret.png')]:
            g=data[data.tag.isin(['A12','A34'])].groupby(['optimizer','method'])[metric].mean().unstack()
            if len(g):g.plot.bar(figsize=(12,5),logy=metric=='one_step_nmse');plt.ylabel(metric);save(filename)
        for method in METAS:
            d=data[(data.method==method)&(data.tag=='A5')]
            if len(d):
                d.groupby(['support_K','optimizer']).one_step_nmse.mean().unstack().plot(marker='o',figsize=(8,5),logy=True)
                plt.ylabel('Independent query NMSE');save(f'fixed_support_budgets_{method}.png')
        if len(data):
            plt.figure(figsize=(7,5))
            for m,g in data.groupby('method'):plt.scatter(g.parameter_delta,g.one_step_nmse,label=m,s=12,alpha=.4)
            plt.yscale('log');plt.xlabel('Parameter displacement');plt.ylabel('Independent query NMSE');plt.legend();save('movement_generalization.png')
    if len(controls):
        for stage in sorted(controls.stage.unique()):
            d=controls[controls.stage==stage]
            if len(d):
                g=d.groupby(['condition','policy']).success.mean().unstack()
                g.plot.bar(figsize=(max(10,len(g)*.65),5));plt.ylabel('Success fraction');plt.legend(fontsize=6,ncol=3);save(f'{stage}_control_success.png')
        c=controls[controls.stage=='C']
        if len(c):
            g=c.groupby(['condition','policy']).recovery_seconds.mean().unstack()
            if len(g):g.plot.bar(figsize=(14,5));plt.ylabel('Recovery seconds (successful recoveries only)');plt.legend(fontsize=6,ncol=3);save('disturbance_recovery_latency.png')
        # One matched representative success and failure per method, selected after evaluation for description only.
        for policy,g in controls.groupby('policy'):
            for outcome in [True,False]:
                subset=g[g.success==outcome]
                if not len(subset):continue
                r=subset.iloc[0]
                with np.load(ctx.run/'controls'/f'{r.key}.npz') as a:
                    n=int(r.survival_steps);s=a['states'][int(r.row),:n+1];u=a['actions'][int(r.row),:n,0];p=a['predictions'][int(r.row),:n]
                    corr=a['corrections'][int(r.row),:n]
                fig,ax=plt.subplots(4,1,figsize=(10,8),sharex=True)
                ax[0].plot(s[:,0],label='Actual x');ax[0].plot(np.arange(1,n+1),p[:,0],label='One-step prediction');ax[0].axhline(2.4,color='r');ax[0].axhline(-2.4,color='r');ax[0].legend()
                ax[1].plot(np.arctan2(np.sin(s[:,2]),np.cos(s[:,2])),label='Angle');ax[1].plot(s[:,3],label='Angular speed');ax[1].legend()
                ax[2].plot(u);ax[2].set_ylabel('Command [N]');ax[3].plot(np.sqrt(np.square(corr).mean(1)));ax[3].set_ylabel('Correction RMS');ax[3].set_xlabel('Control step')
                fig.suptitle(f'{policy}: {r.condition}, success={outcome}')
                save(f'trajectory_{policy}_{outcome}.png')
    return names


def findings(ctx,controls,offline,selection):
    """Numerical synthesis, always scoped to the saved rows actually available."""
    lines=[];criteria=[]
    if ctx.mode!='full':
        return ['These are bounded engineering workloads. No optimizer winner, recovery capability, generalization result or V2 exit decision is scientifically validated.'],pd.DataFrame()
    if len(offline):
        a=offline[(offline.tag.isin(['A12','A34','A5']))&offline.method.isin(METAS)].copy()
        if len(a):
            best=a[a.tag.isin(['A12','A34'])].groupby(['method','optimizer']).agg(query_nmse=('one_step_nmse','mean'),regret=('planning_regret','mean'),
                support_loss=('support_loss_after','mean'),movement=('parameter_delta','mean'),fit_ms=('fit_ms','mean')).reset_index()
            for method,g in best.groupby('method'):
                r=g.sort_values('query_nmse').iloc[0]
                lines.append(f'{method}: lowest descriptive mean query NMSE among screened treatments is {r.optimizer}: {r.query_nmse:.5g}, mean regret {r.regret:.5g}, support loss {r.support_loss:.5g}, movement {r.movement:.4g}, fit {r.fit_ms:.3g} ms. This descriptive minimum is not a held-out optimizer selection.')
            matched=a.merge(a[a.optimizer=='baseline'][['query_id','support_K','method','one_step_nmse','support_loss_after','planning_regret']].drop_duplicates(['query_id','support_K','method']),on=['query_id','support_K','method'],suffixes=('','_baseline'))
            if len(matched):
                overfit=(matched.support_loss_after<matched.support_loss_after_baseline)&(matched.one_step_nmse>matched.one_step_nmse_baseline)
                lines.append(f'Support loss improves while independent error worsens in {int(overfit.sum())}/{len(matched)} matched treatment/query rows. These repeated states are descriptive, not independent replications.')
    if not selection:return lines+['A development selection has not yet been frozen.'],pd.DataFrame()
    if len(offline):
        for method,config in selection['selected'].items():
            q=offline[(offline.tag=='A_confirmation')&(offline.method==method)&(offline.optimizer==config['name'])]
            f=offline[(offline.tag=='A_confirmation')&(offline.method=='frozen')]
            if len(q) and len(f):
                ratio=q.one_step_nmse.mean()/max(f.one_step_nmse.mean(),1e-12)
                regret=q.planning_regret.mean()/max(f.planning_regret.mean(),1e-12)
                lines.append(f'Independent shared-query A confirmation, {method}: selected/Frozen NMSE ratio {ratio:.4g}, regret ratio {regret:.4g}. Support remains exactly 50 independent transitions.')
                criteria.extend([dict(criterion='independent_prediction_ratio',method=method,observed=ratio,threshold=.9,passes_point_threshold=ratio<=.9),
                                 dict(criterion='independent_planning_regret_ratio',method=method,observed=regret,threshold=.9,passes_point_threshold=regret<=.9)])
    if len(controls):
        for method,config in selection['selected'].items():
            label=f'{method}__{config["name"]}'
            d=controls[(controls.stage=='A')&(controls.split=='confirmation')&(controls.evaluation_protocol=='online')]
            for condition,g in d.groupby('condition'):
                b=g[g.policy==method];q=g[g.policy==label];f=g[g.policy=='frozen']
                if not len(b) or not len(q):continue
                change=100*(q.success.mean()-b.success.mean());cost_ratio=q.cost.mean()/max(b.cost.mean(),1e-9)
                lines.append(f'A online confirmation, {condition}, {method}: selected {config["name"]} succeeds {int(q.success.sum())}/{len(q)} versus original {int(b.success.sum())}/{len(b)}; difference {change:+.1f} pp, cost ratio {cost_ratio:.4g}.')
                if condition.endswith('__combined_geometry'):
                    criteria.append(dict(criterion='difficult_control_gain',method=method,observed=change,
                        threshold=ctx.cfg['exit_criteria']['difficult_success_gain_pp'],passes_point_threshold=change>=ctx.cfg['exit_criteria']['difficult_success_gain_pp']))
                if condition.endswith('__nominal'):
                    criteria.append(dict(criterion='nominal_regression',method=method,observed=change,
                        threshold=-ctx.cfg['exit_criteria']['nominal_regression_pp'],passes_point_threshold=change>=-ctx.cfg['exit_criteria']['nominal_regression_pp']))
            dg=controls[(controls.stage=='D')&(controls.policy.isin([label,method,'frozen']))]
            if len(dg):
                q=dg[dg.policy==label];b=dg[dg.policy==method];f=dg[dg.policy=='frozen']
                lines.append(f'D aggregate for {method}: selected {int(q.success.sum())}/{len(q)}, original {int(b.success.sum())}/{len(b)}, Frozen {int(f.success.sum())}/{len(f)}. Read condition tables and paired intervals before inferring generalization; regimes differ.')
                gain=100*(q.success.mean()-b.success.mean())
                criteria.append(dict(criterion='heldout_generalization_success_gain',method=method,observed=gain,
                    threshold=ctx.cfg['exit_criteria']['generalization_success_gain_pp'],passes_point_threshold=gain>=ctx.cfg['exit_criteria']['generalization_success_gain_pp']))
        c=controls[controls.stage=='C']
        if len(c):
            grouped=c.groupby('policy').agg(eligible=('recovery_success','count'),recovered=('recovery_success',lambda v:pd.to_numeric(v,errors='coerce').sum()),
                                          never_stable=('never_stable_before_impulse','sum'),seconds=('recovery_seconds','mean')).reset_index()
            lines.append('C observed recovery (different disturbance types are detailed separately):\n\n'+table(grouped))
            transient=c[c.kind.isin(['cart_impulse','angular_velocity','force_pulse','repeated_cart'])]
            for method,g in transient.groupby('policy'):
                rate=float(pd.to_numeric(g.recovery_success,errors='coerce').mean())
                criteria.append(dict(criterion='transient_recovery',method=method,observed=rate,threshold=.8,passes_point_threshold=rate>=.8))
        latency=ctx.run/'deployment_latency.json'
        if latency.exists():
            for r in json.loads(latency.read_text())['rows']:
                criteria.append(dict(criterion=f'single_controller_fit_frame_H{r["horizon"]}_ms',method=r['label'],observed=r['fit_frame_estimate_ms'],threshold=20.,passes_point_threshold=r['fit_frame_estimate_ms']<=20.))
    lines += ['Confirmed treatment effects require paired independent control comparisons. Parameter movement itself is not isolated: learning rate, steps and optimizer also change fitted functions. A remaining dense gap does not distinguish missing information from objective/optimization limitations on its own.',
              'V2 exit is not a universal-winner decision. The point-threshold table is supplemented by paired seed uncertainty, failure rates and compute limitations. Missing confirmation stages make overall validation incomplete. A highest-information follow-up should target the largest remaining independently measured gap; no new architecture is trained or selected by this report.']
    return lines,pd.DataFrame(criteria)


QUESTIONS=[
    ('Is ten-step optimization restrictive?','Compare A12/A34 held-out errors and A development/confirmation control. A smaller support loss alone is insufficient.'),
    ('Do larger learning rates help?','A1 changes only learning rate, on identical 50 transitions. Read acceptance, query loss and control jointly.'),
    ('Do extra updates improve same-data control?','A2/A7 separate gradient updates from observations; snapshot and online protocols are separate.'),
    ('Is large movement itself beneficial?','Movement is an outcome of randomized fixed treatments, not a causal intervention on its own. Relaxed guards are isolated diagnostics.'),
    ('Does longer fitting overfit?','A5 reports support/query divergence across K and update counts; independent queries never select per-seed checkpoints.'),
    ('Best observation/update combination?','A5 curves and frozen selection list the measured compromise; no universal optimum is inferred.'),
    ('Do informative observations help?','B contrasts ordinary/excitation/privileged-diverse supports at K50 with matched optimizers and queries.'),
    ('Information or optimization limitation?','Combine A same-data treatment effects with B fixed-effort information changes; dense references alter both and remain privileged.'),
    ('Can it reject impulses?','C uses coupled horizontal impulses in N s, separately labelled angular-velocity kicks and physical external-force pulses.'),
    ('Temporary versus persistent identification?','Existing CartPole V2 has no detector. Feedback/adaptation outcomes and residual duration can be compared, but detector accuracy is unavailable.'),
    ('Does adaptation improve recovery?','C includes unadapted Meta, baseline updates and selected updates at identical starts; combined cases condition on reaching stable balance.'),
    ('Does it generalize?','D uses declared unseen interpolation/extrapolation configurations and fresh confirmation trials, never optimization selection.'),
    ('Effect of observation noise?','D sensors corrupt x/v/theta/omega, including adaptation targets. Simulator truth is evaluator-only except privileged Oracle dynamics.'),
    ('Consecutive changes?','D resets initialization and local optimizer state at each known change; 50 observations per segment maximum. This is not continual-learning/detection.'),
    ('Runtime cost?','Update, MPC, inference and loop wall times are logged separately; 0.02 s integration does not establish a 20 ms wall-clock deadline.'),
    ('Reusable V3 components?','Frozen checkpoint loader, residual interface, shared CEM, causal observation buffers and content-verified jobs are retained. Capabilities depend on confirmation results.'),
    ('What motivates future modules?','Only measured residual gaps justify follow-up hypotheses. This suite implements no history, learned probing, gating or control-aware training.'),
    ('Ready to freeze V2?','Use the frozen exit thresholds and confirmation uncertainty; completion or dense-reference success alone does not establish readiness.'),
]


def generate(ctx,only_A=False,development_only=False,selection_override=None):
    controls,offline,updates,forecasts=read(ctx)
    if only_A:
        if len(controls):controls=controls[controls.stage=='A']
        if len(offline):offline=offline[offline.tag.str.startswith('A')]
    if development_only:
        if len(controls):controls=controls[controls.split=='development']
        if len(offline):offline=offline[offline.split=='development']
    tables=ctx.run/('tables_A_development' if development_only else 'tables_A' if only_A else 'tables');tables.mkdir(exist_ok=True)
    if only_A:
        if len(updates):updates=updates[updates.stage=='A']
        if len(forecasts):forecasts=forecasts[forecasts.stage=='A']
    if development_only:
        if len(updates):updates=updates[updates.split=='development']
        if len(forecasts):forecasts=forecasts[forecasts.split=='development']
    for name,data in [('per_episode',controls),('prediction_ranking',offline),('online_updates',updates),('visited_forecasts',forecasts)]:
        save_csv(tables/f'{name}.csv',data.to_dict('records'))
    aggregate=pd.DataFrame();sens=pd.DataFrame();failures=pd.DataFrame()
    if len(controls):
        aggregate=controls.groupby(['stage','split','condition','evaluation_protocol','policy']).agg(
            episodes=('success','size'),success_rate=('success','mean'),mean_cost=('cost','mean'),
            penalty_free_cost=('cost_without_failure_penalty','mean'),rail_rate=('rail_failure','mean'),
            numerical_rate=('numerical_failure','mean'),recovery_rate=('recovery_success','mean'),
            recovery_trials=('recovery_success','count'),never_stable_before_impulse=('never_stable_before_impulse','sum'),
            recovery_seconds=('recovery_seconds','mean'),one_step_nmse=('one_step_nmse','mean'),
            recursive_30_nmse=('recursive_30_nmse','mean'),correction_rms=('correction_rms','mean')).reset_index()
        failures=controls.groupby(['stage','condition','policy','failure_category']).size().reset_index(name='count')
        save_csv(tables/'control_aggregate.csv',aggregate.to_dict('records'));save_csv(tables/'failure_breakdown.csv',failures.to_dict('records'))
        perseed=controls.groupby(['stage','split','condition','evaluation_protocol','policy','seed']).agg(success_rate=('success','mean'),cost=('cost','mean')).reset_index()
        save_csv(tables/'per_seed.csv',perseed.to_dict('records'))
    if len(offline):
        sens=offline.groupby(['tag','environment','method','optimizer','support_K']).agg(
            rows=('one_step_nmse','size'),nmse=('one_step_nmse','mean'),recursive_30=('recursive_30_nmse','mean'),
            regret=('planning_regret','mean'),true_cost=('true_plan_cost','mean'),delta=('parameter_delta','mean'),
            accepted=('accepted','mean'),fit_ms=('fit_ms','mean'),support_loss=('support_loss_after','mean')).reset_index()
        save_csv(tables/'optimization_sensitivity.csv',sens.to_dict('records'))
    effects=paired(controls);save_csv(tables/'paired_seed_intervals.csv',effects.to_dict('records'))
    comparisons=pd.concat([paired(controls,ref) for ref in ['oracle',*METAS]],ignore_index=True)
    save_csv(tables/'paired_meta_and_oracle_intervals.csv',comparisons.to_dict('records'))
    selection=selection_override or (json.loads((ctx.run/'A_selection_frozen.json').read_text()) if (ctx.run/'A_selection_frozen.json').exists() else None)
    conclusions,exit_table=findings(ctx,controls,offline,selection)
    save_csv(tables/'exit_criteria.csv',exit_table.to_dict('records'))
    eventrows=[]
    for path in (ctx.run/'controls').glob('*.json'):
        p=json.loads(path.read_text());eventrows += [dict(r,key=path.stem,seed=p.get('seed'),split=p.get('split')) for r in p.get('event_recoveries',[])]
    if only_A:eventrows=[r for r in eventrows if r['key'].startswith('A_')]
    if development_only:eventrows=[r for r in eventrows if r['split']=='development']
    save_csv(tables/'per_event_recovery.csv',eventrows)
    kind='Scientific evaluation' if ctx.mode=='full' else 'BOUNDED ENGINEERING CHECK ONLY; NO SCIENTIFIC VALIDATION'
    content=[f'# CartPole V2 final hardening {"Experiment A" if only_A else "A–D"}\n',kind,
        '\nPrior evidence (immutable November 1 run): Frozen/Original Meta/Rollout Meta each 2/10 hard-geometry successes, Ridge 0/10, both dense copies and Oracle 10/10. These are prior results, not new measurements.',
        '\n## Execution and interpretation\n',
        f'Rows present: {len(controls)} control episodes and {len(offline)} matched prediction/ranking rows. Mode: {ctx.mode}. Missing stages remain unvalidated.',
        'All prior artifacts are preserved. Frozen normalization, architecture, objective, CEM budget and physical safety checks remain fixed. Dense copies are reused condition-specific upper references. Fixed snapshots use separate support/control episodes; true online controllers collect their own past observations. No future observations or Oracle scoring enter learned control.',
        'Intervals resample paired model-seed means (2,000 draws), not individual states. Five pretrained seeds limit precision. Degenerate intervals do not imply population certainty. Development/confirmation, snapshot/online, and privileged/unprivileged comparisons remain separate.',
        '\n## Frozen configuration decision\n',str(selection or 'No scientific selection is available. Minimal smoke settings never authorize B–D scientific execution.'),
        '\n## Measured findings\n','\n\n'.join(conclusions),
        '\n## Optimization sensitivity\n',table(sens),
        '\n## Control results\n',table(aggregate),
        '\n## Paired differences relative to Frozen\n',table(effects),
        '\n## Failure mechanisms\n',table(failures),
        'The failure categories above describe observed outcomes. Ranking errors require the matched Oracle counterfactual rows; a rail crossing alone is not classified as incorrect ranking. Forecast snapshots use only the weights available at their start. Exogenous impulses can invalidate a constant-physics forecast and are labelled separately.',
        '\n## Quantitative exit criteria\n',str(ctx.cfg['exit_criteria']),
        table(exit_table),
        'These thresholds are protocol-locked before confirmation. Both capability limitations and compute limitations must be reported; no automatic success declaration follows from execution completion.',
        '\n## Scientific questions and evidence\n']
    for question,answer in QUESTIONS:content.append(f'### {question}\n\n{answer}')
    content += ['\n## Runtime and limitations\n',
        'See runtime_estimate.json for measured bounded workloads versus extrapolated stage times, hardware.json for actual hardware, and execution.json for actual completed-stage time. Cached-fit reuse is distinguished from newly executed updates. CPU/GPU utilization is sampled, not inferred from CUDA availability.',
        'No detector exists in the evaluated CartPole stack: false alarms and detector classification are unavailable. Change times are known. Safe-excitation restrictions are heuristic observed-state limits, not a guarantee of safety. Diverse support has privileged broad-pool access and cannot establish 50-interaction deployability. D uses state observation noise, not unmodelled sensor delays. Oracle-unsolved cases remain controller/physical-feasibility uncertainty rather than adapter failure proof. Main runs pause simulation while fitting. Targeted C delayed variants model a single serial fitting worker using the development-p95 delay frozen by A; fitted results use past data and are withheld until the scheduled control step. This is a delay ablation, not a measured asynchronous hardware deployment.',
        '\n## Reproduction\n',
        '```powershell\n.\\.venv\\Scripts\\python.exe scripts\\cartpole\\run_hardening_v2.py --mode full --device cuda\n.\\.venv\\Scripts\\python.exe scripts\\cartpole\\run_hardening_v2.py --mode full --stages A --device cuda\n```']
    name='report_A_development.md' if development_only else 'report_A.md' if only_A else 'report.md'
    target=ctx.run/name;temporary=target.with_suffix('.md.tmp');temporary.write_text('\n\n'.join(content)+'\n',encoding='utf-8')
    import os
    os.replace(temporary,target)
    registry=json.loads((ctx.run/'completion_registry.json').read_text()) if (ctx.run/'completion_registry.json').exists() else {}
    if not only_A:save_json(ctx.run/'report_summary.json',dict(mode=ctx.mode,control_rows=len(controls),prediction_rows=len(offline),
        scientific_validation=ctx.mode=='full' and all(registry.get(s,{}).get('complete') for s in ['A','B','C','D'])))
    return controls,offline

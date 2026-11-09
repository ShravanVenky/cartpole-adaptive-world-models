"""Auditable tables, seed-clustered intervals and evidence-graded interpretation. No fitting or simulation."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import save_json
from .protocol import METAS


def interval(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a):return None,None,None
    g=np.random.default_rng(4123);boot=a[g.integers(len(a),size=(2000,len(a)))].mean(1)
    return float(a.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))


def md(frame):
    if frame.empty:return '*No completed records in this stage.*'
    frame=frame.copy()
    for name in frame:
        if pd.api.types.is_numeric_dtype(frame[name]):frame[name]=frame[name].map(lambda x:'n/a' if pd.isna(x) else f'{x:.4g}')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in frame.to_numpy())


def collect(run,folder,key):
    rows=[]
    for path in sorted((run/folder).glob('*.json')):
        p=json.loads(path.read_text());rows+=p.get(key,[])
    frame=pd.DataFrame(rows)
    # Short smoke trajectories may never enter upright; missing event speeds remain N/A,
    # with numeric dtype so aggregation does not invoke object/empty-slice reductions.
    for name in ['mean_entry_omega','max_entry_omega','high_speed_entry_fraction','crossing_omega','max_overshoot']:
        if name in frame:frame[name]=pd.to_numeric(frame[name],errors='raise')
    return frame


def generate(ctx,bounded=False):
    run=ctx.run;(run/'tables').mkdir(exist_ok=True);(run/'figures').mkdir(exist_ok=True)
    frames={n:collect(run,'replay',n) for n in ['ranking','profiles','substitutions','regimes']}
    frames['fixed_query']=collect(run,'fixed_query','rows');frames['control']=collect(run,'episodes','rows')
    tables={}
    def put(name,frame):frame.to_csv(run/'tables'/f'{name}.csv',index=False);tables[name]=frame;return frame
    for n,f in frames.items():put('raw_'+n,f)
    ranking=frames['ranking'];control=frames['control'];profiles=frames['profiles'];sub=frames['substitutions'];fixed=frames['fixed_query']
    if not ranking.empty:
        # Base vs dense replay are distinct rows; report separate query roles rather than pooling them.
        ranking['query_role']=np.where(ranking.id.str.startswith('fixed_'),'independent_fixed_query',ranking.split)
        group=['query_role','condition','method']
        put('ranking_summary',ranking.groupby(group).agg(states=('id','size'),regret=('normalized_planning_regret','median'),
            search_coverage_regret=('oracle_source_coverage_regret','median'),rank_correlation=('rank_correlation','median'),
            teacher_error=('local_teacher_nmse','median'),recursive_error=('recursive_nmse','median'),false_safe=('false_safe_fraction','mean')).reset_index())
        put('mechanism_counts',ranking.groupby(group+['mechanism']).size().reset_index(name='cases'))
        associations=[]
        for ids,g in ranking.groupby(['split','condition','operating_phase','method']):
            if len(g)<3:continue
            for x in ['local_teacher_nmse','recursive_nmse','decision_critical_nmse','p95_candidate_nmse','oracle_source_coverage_regret','support_distance']:
                t=g[[x,'normalized_planning_regret']].dropna().rank()
                associations.append(dict(zip(['split','condition','operating_phase','method'],ids),predictor=x,states=len(t),
                    spearman=t.corr().iloc[0,1] if len(t)>2 else None))
        put('condition_phase_associations',pd.DataFrame(associations))
        observed=[]
        own=ranking[ranking.method.eq(ranking.collector_method)&~ranking.id.str.startswith('fixed_')]
        for ids,g in own.groupby(['split','condition','operating_phase','method']):
            for x in ['local_teacher_nmse','recursive_nmse','decision_critical_nmse','normalized_planning_regret','oracle_source_coverage_regret','support_distance']:
                for y in ['source_cost','source_success']:
                    a=g[[x,y]].dropna().rank();value=a.corr().iloc[0,1] if len(a)>2 else None
                    observed.append(dict(zip(['split','condition','operating_phase','method'],ids),predictor=x,outcome=y,states=len(a),spearman=value))
        put('prediction_actual_control_associations',pd.DataFrame(observed))
        paired=[]
        for (split,condition),g in ranking.groupby(['split','condition']):
            pivot=g.pivot_table(index=['id','seed'],columns='method',values='normalized_planning_regret')
            for method in [*METAS,'ridge',*['dense_'+m for m in METAS]]:
                if method not in pivot or 'frozen' not in pivot:continue
                delta=(pivot[method]-pivot['frozen']).dropna();a,lo,hi=interval(delta.groupby(level='seed').mean())
                paired.append(dict(split=split,condition=condition,method=method,metric='planning_regret_minus_frozen',states=len(delta),mean=a,ci_low=lo,ci_high=hi))
        put('paired_regret_intervals',pd.DataFrame(paired))
        inversions=[]
        for (split,condition),g in ranking.groupby(['split','condition']):
            columns=['id','seed','local_teacher_nmse','true_cost_own_cem','normalized_planning_regret']
            baseline=g[g.method.eq('frozen')][columns]
            for method in METAS:
                joined=g[g.method.eq(method)][columns].merge(baseline,on=['id','seed'],suffixes=('_model','_frozen'),validate='one_to_one')
                inversions.append(dict(split=split,condition=condition,method=method,pairs=len(joined),
                    prediction_better_fraction=float((joined.local_teacher_nmse_model<.95*joined.local_teacher_nmse_frozen).mean()),
                    prediction_better_native_plan_worse_fraction=float(((joined.local_teacher_nmse_model<.95*joined.local_teacher_nmse_frozen)&
                        (joined.true_cost_own_cem_model>joined.true_cost_own_cem_frozen+np.maximum(joined.true_cost_own_cem_frozen.abs(),1)*.1)).mean())))
        put('decision_prediction_inversions',pd.DataFrame(inversions))
    if not profiles.empty:put('rollout_horizon_summary',profiles.groupby(['split','condition','method','forecast_horizon']).agg(
        states=('id','size'),recursive_error=('recursive_nmse','median'),teacher_error=('teacher_nmse','median'),regret=('normalized_regret','median')).reset_index())
    if not sub.empty:
        put('oracle_substitution_matrix',sub.groupby(['split','condition','meta','candidate_source','scoring_model']).agg(states=('id','size'),
            mean_true_cost=('true_cost','mean'),median_regret=('normalized_regret','median'),unsafe_fraction=('unsafe','mean')).reset_index())
        effects=[]
        for (split,condition,meta),g in sub.groupby(['split','condition','meta']):
            p=g.pivot(index=['id','seed'],columns=['candidate_source','scoring_model'],values='true_cost')
            for label,on,off in [('oracle_scoring',(meta,'oracle'),(meta,meta)),('oracle_candidates',('oracle',meta),(meta,meta)),
                                 ('both_oracle',('oracle','oracle'),(meta,meta))]:
                delta=p[on]-p[off];a,lo,hi=interval(delta.groupby(level='seed').mean())
                effects.append(dict(split=split,condition=condition,meta=meta,intervention=label,pairs=len(delta),true_cost_delta=a,ci_low=lo,ci_high=hi))
            delta=p[('oracle','oracle')]-p[('oracle',meta)]-p[(meta,'oracle')]+p[(meta,meta)]
            a,lo,hi=interval(delta.groupby(level='seed').mean())
            effects.append(dict(split=split,condition=condition,meta=meta,intervention='joint_nonadditivity',pairs=len(delta),true_cost_delta=a,ci_low=lo,ci_high=hi))
        put('substitution_paired_effects',pd.DataFrame(effects))
    if not control.empty:
        put('control_summary',control.groupby(['split','evaluation_type','condition','policy']).agg(episodes=('success','size'),success=('success','mean'),
            cost=('cost','mean'),penalty_free_cost=('cost_without_failure_penalty','mean'),rail_failure=('rail_failure','mean'),
            numerical_failure=('numerical_failure','mean'),survival=('survival_steps','mean')).reset_index())
        put('control_failure_counts',control.groupby(['split','evaluation_type','condition','policy','failure_category']).size().reset_index(name='episodes'))
        put('control_per_seed',control.groupby(['split','evaluation_type','condition','policy','seed']).agg(episodes=('success','size'),success=('success','mean'),cost=('cost','mean')).reset_index())
        paired=[]
        for (split,evaluation_type,condition),g in control.groupby(['split','evaluation_type','condition']):
            p=g.pivot_table(index=['source_case','seed'],columns='policy',values='success')
            if 'frozen' not in p:continue
            for method in [*METAS,'ridge',*['dense_'+m for m in METAS],'oracle']:
                if method not in p:continue
                delta=(p[method]-p['frozen']).dropna();a,lo,hi=interval(delta.groupby(level='seed').mean())
                paired.append(dict(split=split,evaluation_type=evaluation_type,condition=condition,method=method,success_delta=a,ci_low=lo,ci_high=hi,pairs=len(delta)))
        put('control_paired_intervals',pd.DataFrame(paired))
        capacity=[]
        for (condition,evaluation_type),g in control[control.split.eq('confirmation')].groupby(['condition','evaluation_type']):
            p=g.pivot_table(index=['source_case','seed'],columns='policy',values='success')
            for method in METAS:
                if method not in p or 'dense_'+method not in p:continue
                delta=(p['dense_'+method]-p[method]).dropna();a,lo,hi=interval(delta.groupby(level='seed').mean())
                capacity.append(dict(condition=condition,evaluation_type=evaluation_type,method=method,metric='dense_minus_fewshot_success',difference=a,ci_low=lo,ci_high=hi,pairs=len(delta)))
        put('dense_vs_fewshot_control',pd.DataFrame(capacity))
        put('new_control_behavior',control.groupby(['split','evaluation_type','condition','policy']).agg(
            episodes=('success','size'),upright_visits=('upright_visits','mean'),longest_stable_streak=('longest_stable_streak','median'),
            entry_speed=('mean_entry_omega','median'),swingup_fraction=('swingup_fraction','mean'),capture_fraction=('capture_fraction','mean'),
            stable_fraction=('stable_fraction','mean'),force_saturation=('force_saturation_fraction','mean'),rail_clearance=('minimum_rail_clearance','mean')).reset_index())
    if not fixed.empty:put('fixed_query_learning',fixed.groupby(['condition','method','support_K']).agg(
        states=('id','size'),teacher_nmse=('local_teacher_nmse','median'),recursive_nmse=('recursive_nmse','median'),
        regret=('normalized_planning_regret','median'),cost_error=('planning_cost_mae','median'),correction=('correction_rms','median'),support_distance=('support_distance','median')).reset_index())
    dense=[]
    for path in sorted((run/'dense').glob('*_best.json')):
        p=json.loads(path.read_text());dense.append({k:p[k] for k in ['seed','environment','method','parameter_count','training_count','validation_count','total_updates','validation_selected_nmse','wall_seconds']})
    put('dense_fit_summary',pd.DataFrame(dense))
    if (run/'saved_behavior.csv').exists():
        behavior=pd.read_csv(run/'saved_behavior.csv')
        put('archived_behavior_summary',behavior.groupby(['condition','method','safety','regularization']).agg(episodes=('success','size'),
            success=('success','mean'),upright_visits=('upright_visits','mean'),stable_streak=('longest_stable_streak','median'),
            entry_speed=('mean_entry_omega','median'),swingup_fraction=('swingup_fraction','mean'),capture_fraction=('capture_fraction','mean'),
            stable_fraction=('stable_fraction','mean'),force_saturation=('force_saturation_fraction','mean')).reset_index())
    timing=[]
    for folder in ['replay','fixed_query','episodes']:
        for path in (run/folder).glob('*.json'):
            p=json.loads(path.read_text());timing.append(dict(stage=folder,key=path.stem,seconds=p.get('wall_seconds'),actual_interactions=p.get('actual_interactions')))
    put('execution_records',pd.DataFrame(timing))
    figures(ctx,frames)
    status='SMOKE / BENCHMARK ONLY: no scientific validation' if bounded else 'Scientific results from completed selected stages; consult completion_registry.json for scope'
    sections=[f'# CartPole V2 model/planner mismatch investigation\n\n**{status}**\n',
        'No archived checkpoints or results are overwritten. Dense copies are privileged extensive-data references, not few-shot methods.\n',
        '## Design and interpretation\n\nKnown V2 states are development diagnostics. Prospective trials900/901 are independent confirmation. '
        'CEM remains128 candidates,16 elites,3 iterations,H30 swing-up/H15 balancing. Costs and sustained success use unchanged V2 rules. '
        'Candidate replay uses a common zero warm start and all three sampled iterations plus elite means. It diagnoses a search from that state, '
        'not the exact archived search trace. Native controls execute the final elite mean. Substitution diagonals rescore source pools and are not aliases for native CEM.\n\n'
        'Ranking regret compares the true cost of the model-preferred candidate with the best evaluated true cost. Search coverage regret compares '
        'the Oracle-best candidate in a source pool with the union best. Thresholds0.10 and recognized-good tolerance0.05 are predeclared. '
        'These are within-pool finite-objective findings, not globally optimal or whole-episode causal labels.\n\n'
        'Intervals resample paired seed-level means over five pretrained seeds; candidates/checkpoints are not independent model replications. '
        'Phase/condition-specific associations are descriptive. Missing states and truncated bounded controls never become successes.\n']
    for title,name in [('Experiment1: ranking versus discovery','ranking_summary'),('Mechanism counts','mechanism_counts'),
        ('Experiment2: scoring/source substitutions','oracle_substitution_matrix'),('Matched scoring/source interventions','substitution_paired_effects'),
        ('Experiment3: teacher-forced versus recursive errors','rollout_horizon_summary'),
        ('Experiment4: counterfactual recovery and non-stabilization','control_summary'),('Control uncertainty','control_paired_intervals'),
        ('Observed phase behavior','new_control_behavior'),
        ('Experiment5: isolated dense capacity references','dense_fit_summary'),('Dense versus few-shot paired controls','dense_vs_fewshot_control'),
        ('Experiment6: identical-query prefix curves','fixed_query_learning')]:
        sections.append(f'## {title}\n\n'+md(tables.get(name,pd.DataFrame()))+'\n')
    sections.append('## Experiment7: evidence-graded synthesis\n\n'+answers(ctx,frames,bounded)+'\n')
    sections.append('## Interpretation limits\n\n'
        '- Candidate quality is relative to the evaluated pool and unchanged finite-horizon objective. Apparent discovery failure is not proof that no other search can solve the task.\n'
        '- Restoring or reconstructing a common prefix does not turn counterfactual states into real optimizer support. Fixed-query supports come from a different episode.\n'
        '- Dense data include simulator-reset observations and development Oracle trajectories. Physics is used only for labels; this is not deployable few-shot identification. Two optimization settings do not prove capacity impossibility.\n'
        '- Earlier failure replay reports tested Oracle recoverability, never mathematical physical impossibility. Time-to-rail is distinct from time-to-stabilize.\n'
        '- Fixed-query K curves remove moving-query confounding but still vary prefix information and coverage. Repeated states and seeds remain clustered.\n'
        '- H30 is held fixed. Objective versus horizon limitations remain unresolved where the tested substitutions do not isolate them. No policy/value/gate architecture is implemented.\n')
    for name in ['rollout_error.png','ranking_search.png','fixed_query.png','control_success.png','prediction_control.png']:
        if (run/'figures'/name).exists():sections.append(f'![{name}](figures/{name})\n')
    sections.append('Raw paths, terminal states, actions, teacher predictions, horizon costs, origin memberships and chosen indices are retained in replay NPZs. '
        'Control NPZs retain actual states, predictions, actions and corrections. See tables/raw_*.csv and saved_behavior.csv for complete rows, regime metrics and phase phenotypes.\n')
    (run/'report.md').write_text('\n'.join(sections),encoding='utf-8')
    save_json(run/'report_summary.json',dict(bounded=bounded,ranking_rows=len(ranking),control_rows=len(control),fixed_query_rows=len(fixed),dense_copies=len(dense),table_count=len(tables)))
    return frames


def answers(ctx,frames,bounded):
    if bounded:
        return ('Bounded fixtures exercise the seven analysis paths; consult verification.json and status.json for actual completion. Scientific mechanism rankings, capacity claims and architectural recommendations '
                'are withheld until the full stages and independent confirmation finish. Smoke outcomes are not evidence of control capability.\n\n'
                'The full report will answer all13 questions below using the saved matched data and explicitly mark unresolved mechanisms.')
    ranking=frames['ranking'];control=frames['control'];sub=frames['substitutions'];profiles=frames['profiles']
    if ranking.empty or control.empty or sub.empty:
        return 'Incomplete evidence: replay, substitution and closed-loop stages are all required for mechanism attribution. Completed numerical tables are retained; scientific and architectural conclusions remain unresolved.'
    rows=[]
    for method in METAS:
        g=ranking[(ranking.method==method)&(ranking.split=='confirmation')&ranking.condition.eq('active_swingup__combined_geometry')]
        if g.empty:continue
        for mechanism,n in g.mechanism.value_counts().items():rows.append(dict(method=method,mechanism=mechanism,cases=int(n),fraction=n/len(g),
            evidence='Directly demonstrated within tested candidate pools' if mechanism in ['ranking_failure','search_failure','mixed_failure'] else 'Unresolved episode cause'))
    table=pd.DataFrame(rows)
    if len(table):table=table.sort_values('fraction',ascending=False)
    questions=[
        ('1. Why corrections worsen decisions?','Compare matched teacher/recursive error, ranking inversions and source/scorer substitutions; the measured table above reports pool-specific errors, not a universal cause.'),
        ('2. Prediction or discovery?','Rank the confirmation mechanism fractions below; paired Oracle-scoring and Oracle-candidate interventions isolate finite-objective contributions. They need not add.'),
        ('3. Meta recognizes Oracle candidates?','Oracle-source/Meta-scoring cells and oracle_best_predicted_percentile quantify recognition on identical sequences.'),
        ('4. Oracle recognizes misjudged Meta actions?','Meta-source/Oracle-scoring versus Meta/Meta cells give paired true-cost changes; privileged offline reference only.'),
        ('5. When do predictions deteriorate?','H1/5/10/20/30 profiles separate teacher forcing and recursive error on common and selected trajectories. No universal horizon is inferred from pooled physics.'),
        ('6. Where do hard episodes fail?','saved_behavior.csv reports visits, entry speed, stable streaks, force reversals and phase occupancy; control_failure_counts separates rails and non-stabilization.'),
        ('7. How early could Oracle recover?','earlier_recovery_windows.csv reports the latest tested recoverable lead5/10/20/40; no Oracle success is labelled no recovery demonstrated, not physically impossible.'),
        ('8. Is capacity sufficient?','Dense train/validation and independent confirmation control must be assessed jointly; good train fit with poor held-out accuracy is generalization, while poor optimization is not proof of architectural insufficiency.'),
        ('9. Is few-shot adaptation limiting?','Matched dense versus few-shot prediction, ranking and control gaps test this; dense data and optimization are privileged and cannot establish genuine few-shot learning.'),
        ('10. What proportion supports each mechanism?','Confirmation within-pool fractions below use explicit labels. They are proportions of selected state patterns, not an additive percentage decomposition of all failed episodes.'),
        ('11. Would control-aware training help?','Supported as a next test only if decision-critical ranking errors persist with discovery held fixed. Correlation alone cannot establish the training intervention will help.'),
        ('12. Would policy/value MPC help?','Candidate proposals are motivated if Oracle candidates rescue Meta scoring; a value model specifically is unresolved because horizons/objectives were not changed.'),
        ('13. Single permanent improvement?','Use the leading independently confirmed substitution effect to choose one next experiment; do not permanently adopt a new architecture solely from these diagnostics.')]
    # Put measured values next to interpretations rather than emitting a generic checklist.
    summaries=[]
    hard=ranking[ranking.split.eq('confirmation')&ranking.condition.eq('active_swingup__combined_geometry')]
    for method in METAS:
        g=hard[hard.method.eq(method)]
        if len(g):summaries.append(dict(method=method,states=len(g),ranking_or_mixed_pct=100*g.mechanism.isin(['ranking_failure','mixed_failure']).mean(),
            search_or_mixed_pct=100*g.mechanism.isin(['search_failure','mixed_failure']).mean(),median_rank=g.rank_correlation.median(),
            median_regret=g.normalized_planning_regret.median(),median_native_regret=g.native_cem_regret.median(),
            recognizes_good_pct=100*g.recognizes_good_candidate.mean()))
    numeric=md(pd.DataFrame(summaries))
    if len(summaries):
        questions[0]=(questions[0][0],'Measured confirmation fractions and regrets are shown below; decision_prediction_inversions.csv tests lower teacher error against worse native-plan true cost on identical states.')
        questions[1]=(questions[1][0],'The paired intervention and mechanism tables quantify both. Ranking/search overlap is labelled mixed and cannot be added as episode-cause percentages.')
    h=profiles[profiles.split.eq('confirmation')&profiles.condition.eq('active_swingup__combined_geometry')]
    if len(h):
        growth=h.groupby(['method','forecast_horizon']).recursive_nmse.median().unstack()
        observations=[]
        for method in METAS:
            if method in growth.index:observations.append(f'{method}: median recursive NMSE H1={growth.loc[method,1]:.4g}, H30={growth.loc[method,30]:.4g}')
        questions[4]=(questions[4][0],'; '.join(observations)+'. Horizon/cost tables retain intermediate steps and selected-plan errors; concentration matters alongside mean growth.')
    native=control[control.split.eq('confirmation')&control.evaluation_type.eq('actual_online')&control.condition.eq('active_swingup__combined_geometry')]
    if len(native):
        observations=[]
        for method in [*METAS,*['dense_'+m for m in METAS],'frozen','oracle']:
            g=native[native.policy.eq(method)]
            if len(g):observations.append(f'{method}: {int(g.success.sum())}/{len(g)} success, {int(g.rail_failure.sum())} rail failures, median stable streak {g.longest_stable_streak.median():.0f}')
        questions[5]=(questions[5][0],'; '.join(observations)+'. Phase occupancy and entry-speed tables describe behavior, not causal proof of braking or momentum failure.')
        questions[7]=(questions[7][0],'Independent control evidence: '+ '; '.join(observations[2:4])+'. Compare dense validation and fixed-query errors separately. Extensive-data success supports representability; failure alone cannot distinguish optimization from capacity.')
        questions[8]=(questions[8][0],'Compare the paired dense/few-shot control intervals and identical-state error changes above. Dense gains isolate an extensive-data/optimization advantage, not sample count alone.')
    recovery=ctx.run/'earlier_recovery_windows.csv'
    if recovery.exists():
        q=pd.read_csv(recovery);yes=q.latest_tested_recoverable_lead.dropna()
        questions[6]=(questions[6][0],f'{len(yes)}/{len(q)} seed/collector/Oracle-policy groups demonstrated recovery; '+
            ('latest successful tested lead counts: '+str(yes.astype(int).value_counts().sort_index().to_dict()) if len(yes) else 'no recovery was demonstrated at the tested checkpoints')+'. This is not proof of physical impossibility.')
    out=md(pd.DataFrame(questions,columns=['Question','Measured evidence and interpretation']))+'\n\n'+numeric+'\n\n'+md(table)
    hmeta=hard[hard.method.isin(METAS)]
    rank_status='Directly demonstrated within tested pools' if hmeta.mechanism.isin(['ranking_failure','mixed_failure']).any() else 'Unresolved: no material event observed'
    search_status='Directly demonstrated within tested pools' if hmeta.mechanism.isin(['search_failure','mixed_failure']).any() else 'Unresolved: no material event observed'
    capacity_path=ctx.run/'tables/dense_vs_fewshot_control.csv'
    capacity=pd.read_csv(capacity_path) if capacity_path.exists() and capacity_path.stat().st_size>1 else pd.DataFrame()
    dense_status='Supported but not fully isolated' if len(capacity) and (capacity.ci_low>0).any() else 'Unresolved: no positive paired control interval established'
    nonmonotonic=False
    if len(frames['fixed_query']):
        for _,g in frames['fixed_query'].groupby(['condition','seed','method']):
            a=g.sort_values('support_K').local_teacher_nmse.to_numpy()
            nonmonotonic|=bool((a[1:]>1.05*a[:-1]+1e-8).any())
    assessment=pd.DataFrame([
        ('Candidate-ranking errors',rank_status,'Matched predicted preferences are replayed under exact dynamics; fractions above concern evaluated states/pools.'),
        ('Candidate discovery and final CEM selection',search_status,'Oracle-best source coverage, recognized alternatives and final-mean gaps distinguish missing candidates from scoring errors.'),
        ('Compounding recursive error','Supported but not fully isolated','Teacher/recursive and horizon tables quantify accumulation; its causal episode impact also requires substitution controls.'),
        ('Few-shot identification/data quality',dense_status,'Extensive-data versus few-shot paired gaps change both data and optimization; they do not isolate observation count alone.'),
        ('Support/query mismatch','Plausible; association measured','Condition/operating-phase distance/error associations are descriptive, with state visitation as a confound.'),
        ('Insufficient representational capacity','Unresolved','A successful dense reference supports representability; failed optimization cannot prove insufficiency.'),
        ('Short horizon or objective design','Unresolved','H30/H15 and objectives are held fixed; Oracle-unsolved or locally reasonable plans do not isolate horizon versus objective.'),
        ('Universal monotonic prediction improvement','Contradicted by observed fixed-query prefixes' if nonmonotonic else 'Unresolved: no material violation in these prefixes','A trajectory prefix changes information quality as well as count.'),
        ('Model/planner interaction','Matched contrast measured; causal episode role unresolved','Joint substitution contrast is computed on identical candidates/states; extrapolation to architecture success remains unresolved.')],
        columns=['Mechanism','Evidence status','Scope'])
    assessment.to_csv(ctx.run/'tables/mechanism_evidence_assessment.csv',index=False)
    out+='\n\n'+md(assessment)
    effects=pd.read_csv(ctx.run/'tables/substitution_paired_effects.csv') if (ctx.run/'tables/substitution_paired_effects.csv').exists() else pd.DataFrame()
    if not effects.empty:
        e=effects[effects.split.eq('confirmation')&effects.condition.eq('active_swingup__combined_geometry')&effects.intervention.isin(['oracle_scoring','oracle_candidates'])]
        if len(e):
            best=e.sort_values('true_cost_delta').iloc[0]
            out+=f'\n\nLargest observed matched finite-horizon cost reduction: **{best.intervention}**, {best.meta}, delta {best.true_cost_delta:.4g}, seed interval [{best.ci_low:.4g},{best.ci_high:.4g}]. '
            out+=('Prioritize a controlled decision-critical residual-training experiment before adding a controller component.' if best.intervention=='oracle_scoring' else 'Prioritize a candidate-proposal intervention with scoring and budgets held fixed; this does not validate a value network.')
            if best.ci_high>=0:out+=' The interval includes no benefit, so the architectural choice remains unresolved.'
    return out


def figures(ctx,frames):
    def save(name):plt.tight_layout();plt.savefig(ctx.run/'figures'/name,dpi=130,bbox_inches='tight');plt.close()
    p=frames['profiles']
    if len(p):
        g=p[p.condition.eq('active_swingup__combined_geometry')]
        g=g[g.split.eq('confirmation')] if g.split.eq('confirmation').any() else g[g.id.str.startswith('dev_')]
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for method,q in g.groupby('method'):
            t=q.groupby('forecast_horizon')[['recursive_nmse','teacher_nmse']].median()
            for ax,metric in zip(axes,['recursive_nmse','teacher_nmse']):ax.plot(t.index,t[metric],marker='o',label=method)
        for ax,title in zip(axes,['Recursive state-scale NMSE','Teacher-forced delta-scale NMSE']):ax.set_title(title);ax.set_xlabel('Forecast steps');ax.set_yscale('symlog',linthresh=1e-6);ax.legend(fontsize=6)
        save('rollout_error.png')
    r=frames['ranking']
    if len(r):
        selected=r[r.split.eq('confirmation')] if r.split.eq('confirmation').any() else r
        fig,ax=plt.subplots(figsize=(8,5))
        for method,q in selected.groupby('method'):ax.scatter(q.oracle_source_coverage_regret,q.normalized_planning_regret,s=15,alpha=.5,label=method)
        ax.set_xlabel('Oracle-best source-pool coverage regret');ax.set_ylabel('Model ranking regret');ax.set_xscale('symlog',linthresh=.01);ax.set_yscale('symlog',linthresh=.01);ax.legend(fontsize=6);save('ranking_search.png')
        own=selected[selected.method.eq(selected.collector_method)&~selected.id.str.startswith('fixed_')]
        if len(own):
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for (condition,method),q in own.groupby(['condition','method']):
                axes[0].scatter(q.decision_critical_nmse,q.source_cost,s=18,alpha=.6,label=f'{condition}: {method}')
                axes[1].scatter(q.normalized_planning_regret,q.source_cost,s=18,alpha=.6,label=f'{condition}: {method}')
            for ax,title in zip(axes,['Decision-critical prediction error','Planning regret']):
                ax.set_xlabel(title);ax.set_ylabel('Observed source episode cost');ax.set_xscale('symlog',linthresh=.001);ax.set_yscale('symlog',linthresh=1);ax.legend(fontsize=5)
            save('prediction_control.png')
    f=frames['fixed_query']
    if len(f):
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for method,q in f[f.condition.eq('active_swingup__combined_geometry')].groupby('method'):
            t=q.groupby('support_K')[['local_teacher_nmse','normalized_planning_regret']].median()
            for ax,col in zip(axes,['local_teacher_nmse','normalized_planning_regret']):ax.plot(t.index,t[col],'o-',label=method)
        for ax,title in zip(axes,['Identical-query prediction','Identical-pool regret']):ax.set_title(title);ax.set_xlabel('Support observations');ax.set_yscale('symlog',linthresh=1e-6);ax.legend(fontsize=6)
        save('fixed_query.png')
    c=frames['control']
    if len(c):
        q=c[c.condition.eq('active_swingup__combined_geometry')]
        q=q[q.evaluation_type.eq('actual_online')] if q.evaluation_type.eq('actual_online').any() else q
        t=q.groupby('policy').success.mean()
        fig,ax=plt.subplots(figsize=(max(8,len(t)*.6),4));t.plot.bar(ax=ax);ax.set_ylabel('Success fraction');ax.set_ylim(0,1.05);ax.tick_params(axis='x',labelsize=7);save('control_success.png')

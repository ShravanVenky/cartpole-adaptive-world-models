"""Task-specific interpretation of saved control metrics, without model execution."""
import json
import numpy as np
import pandas as pd


def interpret(run,cfg,rows,paired):
    selected=rows[(rows.protocol=='fixed')&((rows.budget==50)|rows.method.isin(['frozen','oracle']))]
    table=selected.groupby(['task','horizon','method']).agg(success=('success','mean'),cost=('cost','mean'),episodes=('success','size'))
    keys=['seed','trial','environment','category','task','horizon','protocol']
    comparisons=[]
    for candidate,reference in [('ridge','frozen'),('meta_residual_1step','ridge'),('meta_residual_rollout','ridge'),('meta_residual_rollout','meta_residual_1step')]:
        left=selected[selected.method.eq(candidate)];right=selected[selected.method.eq(reference)]
        joined=left.merge(right,on=keys,suffixes=('_candidate','_reference'),validate='one_to_one')
        for (task,h),sub in joined.groupby(['task','horizon']):
            pred=sub.one_step_nmse_candidate<sub.one_step_nmse_reference
            inversion=pred&(sub.cost_candidate>sub.cost_reference)
            comparisons.append({'task':task,'horizon':int(h),'candidate':candidate,'reference':reference,'episodes':len(sub),
                'success_difference':float((sub.success_candidate.astype(float)-sub.success_reference.astype(float)).mean()),
                'median_paired_cost_benefit':float((1-sub.cost_candidate/sub.cost_reference).median()),
                'mean_cost_difference':float((sub.cost_candidate-sub.cost_reference).mean()),
                'lower_prediction_error_fraction':float(pred.mean()),'lower_error_but_higher_cost_fraction':float(inversion.mean())})
    pd.DataFrame(comparisons).to_csv(run/'method_pairwise_comparisons.csv',index=False)
    def performance(task,h=None):
        s=selected[selected.task.eq(task)]
        if h is not None:s=s[s.horizon.eq(h)]
        values=s.groupby('method').agg(success=('success','mean'),cost=('cost','mean')).round(4)
        return '; '.join(f'{m}: success {v.success:.1%}, mean cost {v.cost:.1f}' for m,v in values.iterrows())
    def contrast(task,candidate,reference,h):
        x=next((x for x in comparisons if x['task']==task and x['horizon']==h and x['candidate']==candidate and x['reference']==reference),None)
        if x is None:return 'No matching comparison'
        return f"success difference {x['success_difference']:+.1%}; median paired cost reduction {x['median_paired_cost_benefit']:+.1%}; lower one-step error but higher cost in {x['lower_error_but_higher_cost_fraction']:.1%} of all matched episodes"
    online=rows[rows.protocol.eq('online')&rows.method.isin(['ridge','meta_residual_1step','meta_residual_rollout'])]
    off=online[online.budget.eq(0)];on=online[online.budget.eq(1)]
    joined=on.merge(off,on=keys+['method'],suffixes=('_on','_off'),validate='one_to_one')
    joined['adaptation_worsened_cost']=joined.cost_on>joined.cost_off
    joined['adaptation_lost_success']=joined.success_off&~joined.success_on
    recovery=joined.groupby(['task','horizon','method']).agg(success_on=('success_on','mean'),success_off=('success_off','mean'),
        cost_on=('cost_on','mean'),cost_off=('cost_off','mean'),recovery_on=('recovery_steps_on','median'),recovery_off=('recovery_steps_off','median'),
        worsened_cost_fraction=('adaptation_worsened_cost','mean'),lost_success_fraction=('adaptation_lost_success','mean'))
    recovery.to_csv(run/'online_enabled_vs_disabled.csv')
    oracle=rows[rows.method.eq('oracle')].groupby(['task','protocol','environment','horizon']).agg(success_rate=('success','mean'),mean_cost=('cost','mean'),episodes=('success','size'))
    oracle.to_csv(run/'oracle_feasibility.csv')
    failed=paired[paired.failed]
    breakdown=failed.groupby(['task','protocol','category','method']).agg(failures=('failed','size'),paired_oracle_success=('oracle_success','mean'))
    breakdown.to_csv(run/'failure_categories.csv')
    answers={
      'Does improved predictive accuracy translate into better control?':
        'The matched prediction/control inversion fractions are in method_pairwise_comparisons.csv; '+contrast('balance','meta_residual_1step','ridge',15)+'. Prediction correlations alone are not causal evidence.',
      'Is Ridge still strongest around upright balancing?':performance('balance')+'. Use task/horizon/category breakdown and paired seed intervals before asserting a winner.',
      'Does Ridge fail when nonlinear motion becomes substantial?':performance('swingup',cfg['swingup_extra_horizon'])+'. H5/H15 failures are separately controller-horizon-limited. Inspect oracle_feasibility.csv for shift-specific controllability under this optimizer.',
      'Can Original Meta outperform Ridge in swing-up?':contrast('swingup','meta_residual_1step','ridge',cfg['swingup_extra_horizon']),
      'Does Rollout Meta outperform Original Meta at longer MPC horizons?':
        'Balancing H5: '+contrast('balance','meta_residual_rollout','meta_residual_1step',5)+'; H15: '+contrast('balance','meta_residual_rollout','meta_residual_1step',15)+'; swing-up H30: '+contrast('swingup','meta_residual_rollout','meta_residual_1step',cfg['swingup_extra_horizon']),
      'How much does online adaptation improve recovery?':
        '; '.join(f'{task} H{h} {method}: success change {v.success_on-v.success_off:+.1%}, median recovery {v.recovery_off:.0f} to {v.recovery_on:.0f} steps, cost worsened in {v.worsened_cost_fraction:.1%}, successful disabled episode lost in {v.lost_success_fraction:.1%}' for (task,h,method),v in recovery.iterrows())+'. Updates use cumulative5/10/20/50 actual post-change transitions. Full enabled/disabled costs and times are in online_enabled_vs_disabled.csv.',
      'Which methods handle severe and combined dynamics shifts best?':
        '; '.join(f'{task} {category}: '+', '.join(f'{method} success {v.success:.1%}, mean cost {v.cost:.1f}' for method,v in sub.groupby('method').agg(success=('success','mean'),cost=('cost','mean')).iterrows()) for (task,category),sub in selected[selected.category.isin(['strong_OOD','combined'])].groupby(['task','category']))+'. These pool planning horizons; horizon-specific conditions remain in aggregate_results.csv.',
      'Are catastrophic failures attributable to model, adapter, controller, or physical constraints?':
        f'{len(failed)} rail/numerical failure rows. failure_categories.csv reports the fraction whose paired oracle succeeded. Oracle success with learned failure supports a model/optimizer-with-model explanation; worse adapted versus unadapted outcomes implicate adaptation descriptively. Oracle failures leave controller/horizon/constraints versus physical feasibility unresolved; they do not establish physical impossibility. Frozen shadow errors and visited coverage separate backbone inaccuracies from adapter corrections.',
      'Which existing adapter is strongest for subsequent architecture development?':
        'Choose per task from the saved success/cost and seed-interval tables. No universal architecture is selected solely from pooled cost or prediction metrics.'}
    # Report observed task winners without converting them into unsupported significance claims.
    winners={}
    for task,h in [('balance',15),('swingup',cfg['swingup_extra_horizon'])]:
        sub=selected[(selected.task==task)&(selected.horizon==h)&(~selected.method.isin(['oracle']))]
        ranking=sub.groupby('method').agg(success=('success','mean'),cost=('cost','mean')).sort_values(['success','cost'],ascending=[False,True])
        winners[task]={'observed_leader':ranking.index[0],'success_rate':float(ranking.iloc[0].success),'mean_cost':float(ranking.iloc[0].cost)}
    answers['Which existing adapter is strongest for subsequent architecture development?']='Observed leaders, ranking success first and cost second: '+json.dumps(winners)+'. These are descriptive; paired intervals and feasibility-conditioned breakdown must support any stronger claim.'
    recommendation='The single most informative next experiment is a matched trajectory-replay comparison restricted to oracle-successful nonlinear conditions, evaluating unadapted and adapted predictors on identical visited states/actions and optimizer candidate sequences. This tests whether action-ranking errors, backbone coverage or the adaptation update is the bottleneck before any architecture change.'
    (run/'control_answers.json').write_text(json.dumps({'answers':answers,'observed_task_leaders':winners,'single_next_experiment':recommendation},indent=2))
    return answers,recommendation

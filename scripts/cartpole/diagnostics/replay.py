"""Counterfactual action rankings on matched historical states and candidate pools."""
import json
import time
import numpy as np
import pandas as pd
import torch
from adaptive_world_model.control.evaluation_mpc import physics_step
from adaptive_world_model.data import collect_sequential
from adaptive_world_model.environments.cartpole import CartPolePhysics
from adaptive_world_model.utils.randomness import stable_seed
from .protocol import HISTORICAL, METHODS, ADAPTERS, ALPHAS, setting
from .context import save_json, save_npz, transition_batch
from .engine import Predictor, Planner, rollout


def select_states(ctx, smoke=False):
    """Archived outcomes define diagnostic strata, never new support selection or confirmation tuning."""
    table = pd.read_csv(HISTORICAL/'per_episode_results.csv')
    table = table[(table.task == 'swingup') & (table.horizon == 30) & ((table.budget == 50) | table.method.isin(['frozen', 'oracle']) | ((table.protocol == 'online') & (table.budget == 1)))]
    records, missing = [], []
    for seed in ([11] if smoke else ctx.cfg['seeds']):
        for split, trials in [('development', [0]), ('confirmation', [1, 2])]:
            frame = table[(table.seed == seed) & table.trial.isin(trials)].copy()
            frozen_good = set(tuple(x) for x in frame[(frame.method == 'frozen') & frame.success][['protocol','environment','trial']].to_numpy())
            for category in ctx.cfg['replay_categories']:
                f = frame.copy()
                if category == 'frozen_success': f = f[(f.method == 'frozen') & f.success & (f.protocol == 'fixed')]
                elif category == 'frozen_success_meta_failure':
                    f = f[f.method.isin(ADAPTERS[1:]) & ~f.success & (f.protocol == 'fixed')]
                    f = f[[tuple(x) in frozen_good for x in f[['protocol','environment','trial']].to_numpy()]]
                elif category == 'ridge_failure': f = f[(f.method == 'ridge') & f.failed & (f.protocol == 'fixed')]
                elif category == 'combined': f = f[(f.environment == 'combined_geometry') & (f.protocol == 'fixed') & (f.method == 'frozen')]
                elif category == 'online_shift': f = f[(f.protocol == 'online') & (f.method == 'frozen') & (f.environment == 'strong_gain')]
                else: f = f[(f.protocol == 'fixed') & (f.method == 'frozen')]
                if len(f) == 0:
                    missing.append(dict(seed=seed, split=split, category=category, reason='no archived qualifying episode'))
                    continue
                chosen = None
                for _, row in f.sort_values(['environment', 'trial']).iterrows():
                    key = f'{row.protocol}_swingup_H30_{row.environment}_seed{seed}'
                    packet = json.loads((HISTORICAL/'episodes'/f'{key}.json').read_text())
                    ix = next(i for i,r in enumerate(packet['rows']) if r['method']==row.method and r['budget']==row.budget and r['trial']==row.trial)
                    with np.load(HISTORICAL/'episodes'/f'{key}.npz') as data:
                        states = data['states'][ix]
                    end = min(int(row.survival_steps)-1, len(states)-2)
                    if end < 1: continue
                    if category == 'high_omega': t = int(np.abs(states[1:end+1,3]).argmax()+1)
                    elif category == 'near_rail':
                        t = int(np.abs(states[:end+1,0]).argmax())
                        if abs(states[t,0]) < 1.8: continue
                    elif category == 'large_angle':
                        a = np.abs(np.arctan2(np.sin(states[:,2]),np.cos(states[:,2])))
                        cand = np.flatnonzero((a > 1.57) & (np.arange(len(a)) >= 10) & (np.arange(len(a)) <= end))
                        if not len(cand): continue
                        t = int(cand[len(cand)//2])
                    elif category == 'online_shift': t = min(end, 35)
                    elif category in ['ridge_failure','frozen_success_meta_failure']: t = max(0,end-10)
                    else: t = min(end,50)
                    chosen = dict(id=f'{split}_{seed}_{category}', split=split, seed=seed, category=category,
                                  source_key=key, source_row=ix, step=t, state=states[t].tolist(), source=packet['rows'][ix])
                    break
                if chosen is None: missing.append(dict(seed=seed,split=split,category=category,reason='no state satisfied declared stratum threshold'))
                else: records.append(chosen)
    save_json(ctx.run/'replay_state_manifest.json', dict(states=records, unavailable_strata=missing,
              selection='archived diagnostic strata only; trial0 development, trials1/2 held-out; no support selection by outcomes'))
    return records


def historical_support(record):
    row, seed = record['source'], record['seed']
    if row['protocol'] == 'online':
        change = 25
        available = record['step']-change
        k = max([n for n in [5,10,20,50] if n <= available], default=0)
        if not k: return None
        with np.load(HISTORICAL/'episodes'/f'{record["source_key"]}.npz') as data:
            ix = record['source_row']
            return transition_batch(data['states'][ix,change:change+k], data['actions'][ix,change:change+k],
                                    data['states'][ix,change+1:change+k+1], f'archived_actual_past/{record["id"]}')
    cfg = json.loads((HISTORICAL/'config_locked.json').read_text())
    env = next(e for e in cfg['environments'] if e['name'] == row['environment'])
    p = CartPolePhysics(**{k:v for k,v in env.items() if k not in ['name','category']})
    return collect_sequential(p,50,stable_seed(seed,'control_support',row['environment'],'swingup',row['trial']),
                              'broad','cpu',stream='control_independent_support')


def rankings(pred_cost, true_cost, pred_path, truth, k):
    p, q = np.nan_to_num(pred_cost, nan=1e30), np.nan_to_num(true_cost,nan=1e30)
    pi, qi = np.argsort(p, kind='stable'), np.argsort(q, kind='stable')
    pr, qr = pd.Series(p).rank().to_numpy(), pd.Series(q).rank().to_numpy()
    corr = float(np.corrcoef(pr,qr)[0,1]) if np.std(pr)>0 and np.std(qr)>0 else None
    unsafe = (~np.isfinite(truth).all((1,2))) | (np.abs(truth[:,:,0]).max(1)>2.4) | (np.abs(truth[:,:,1:]).max((1,2))>100)
    pred_safe = np.isfinite(pred_path).all((1,2)) & (np.abs(pred_path[:,:,0]).max(1)<=2.4)
    err = pred_path-truth
    err[:,:,2] = np.arctan2(np.sin(err[:,:,2]), np.cos(err[:,:,2]))
    finite_err = np.nan_to_num(np.square(err/np.array([2.4,3,np.pi,4])).mean((1,2)),nan=1e20,posinf=1e20)
    best = float(q[qi[0]])
    return dict(top1_agreement=bool(np.isclose(q[pi[0]],best,rtol=1e-6,atol=1e-6)),
                topk_agreement=len(set(pi[:k])&set(qi[:k]))/k, rank_correlation=corr,
                preferred_candidate=int(pi[0]), preferred_true_cost=float(q[pi[0]]), best_evaluated_true_cost=best,
                planning_regret=float(q[pi[0]]-best), normalized_regret=float((q[pi[0]]-best)/max(abs(best),1)),
                unsafe_top1=bool(unsafe[pi[0]]), oracle_best_unsafe=bool(unsafe[qi[0]]), safe_candidate_exists=bool((~unsafe).any()),
                avoidable_unsafe_choice=bool(unsafe[pi[0]] and ((~unsafe)&(q<q[pi[0]]-1e-5)).any()), false_safe_topk=float((unsafe[pi[:k]]&pred_safe[pi[:k]]).mean()),
                one_step_nmse=float(np.square(err[:,0]/np.array([2.4,3,np.pi,4])).mean()),
                multi_step_nmse=float(np.median(finite_err)), recursive_5_nmse=float(np.nanmean(np.square(err[:,:5]/np.array([2.4,3,np.pi,4])))),
                prediction_invalid_fraction=float((~np.isfinite(pred_path).all((1,2))).mean()))


@torch.no_grad()
def run_replay(ctx, record, smoke=False):
    target = ctx.run/'replay'/f'{record["id"]}.json'
    cached = ctx.cached(target)
    if cached is not None: return cached
    start = time.perf_counter()
    device, cfg = ctx.device, ctx.cfg
    support = historical_support(record)
    settings = [setting('frozen'),setting('oracle')]+[setting(m,'blend',alpha=a) for m in ADAPTERS for a in ALPHAS]
    base, models, logs = ctx.deploy(record['seed'], settings, [support]*len(settings))
    pack = Predictor(base, models, settings, device)
    pack.physics = ctx.physics(record['source']['environment']) if record['source']['protocol'] != 'online' or record['step']>=25 else CartPolePhysics()
    state = torch.tensor(record['state'],device=device)
    baseline = [setting(m) for m in METHODS]
    _, bm, _ = ctx.deploy(record['seed'],baseline,[support]*len(baseline))
    bp = Predictor(base,bm,baseline,device);bp.physics=pack.physics
    initial = state[None].expand(len(baseline),-1)
    g = torch.Generator(device=device).manual_seed(stable_seed(record['seed'],'ranking_noise',record['id']))
    search_records, pools, pool_labels, mean_sequences = [], [], [], []
    # Budget interventions share the same state, model, objective and explicit elite fraction.
    for budget in ['baseline','expanded','high']:
        options=[dict(s,search=budget) for s in baseline]
        planner=Planner(cfg['replay_horizon'],'swingup',options,device,cfg)
        planner.record_candidates=True
        shared=torch.randn(planner.iterations,1,planner.n,(planner.h+4)//5,1,generator=g,device=device).expand(-1,len(options),-1,-1,-1)
        t0=time.perf_counter();planner.action(bp,initial,0,shared)
        if device=='cuda':torch.cuda.synchronize()
        runtime=time.perf_counter()-t0
        plans=planner.selected_plan[:,None]
        pred,pc=rollout(bp,initial,plans,planner)
        actual,tc=rollout(lambda x,u:physics_step(x,u,pack.physics),initial,plans,planner)
        for i,m in enumerate(METHODS):
            search_records.append(dict(method=m,search=budget,predicted_cost=float(pc[i,0]),true_cost=float(tc[i,0]),
                                       rail_clearance=float(2.4-actual[i,0,:,0].abs().max()),runtime_batch_seconds=runtime,
                                       convergence_predicted_best=planner.iteration_best[:,i].cpu().tolist()))
        means=planner.iteration_means.transpose(0,1).contiguous()
        # Oracle-evaluate means at every iteration to distinguish convergence from exploitation.
        _,true_iter=rollout(lambda x,u:physics_step(x,u,pack.physics),initial,means,planner)
        for i in range(len(options)):search_records[-len(options)+i]['convergence_true_mean_cost']=true_iter[i].cpu().tolist()
        mean_sequences.append(planner.selected_plan.clone())
        if budget=='baseline':
            candidate=planner.last_candidates[-1].reshape(-1,planner.h,1)
            pools.append(candidate);pool_labels.extend(['cem_final_candidates']*len(candidate))
    h=cfg['replay_horizon']
    broad=(torch.rand(cfg['replay_broad_candidates'],(h+4)//5,1,generator=g,device=device)*20-10).repeat_interleave(5,1)[:,:h]
    pools += [broad,*mean_sequences]
    pool_labels += ['broad_uniform']*len(broad)+[f'{b}_elite_average' for b in ['baseline','expanded','high'] for _ in METHODS]
    candidates=torch.cat(pools)
    planner=Planner(h,'swingup',settings,device,cfg)
    paths,costs=[],[]
    for chunk in candidates.split(cfg['replay_chunk']):
        a=chunk[None].expand(len(settings),-1,-1,-1)
        path,cost=rollout(pack,state[None].expand(len(settings),-1),a,planner)
        paths.append(path.cpu().numpy());costs.append(cost.cpu().numpy())
    paths,costs=np.concatenate(paths,1),np.concatenate(costs,1)
    truth,truecost=paths[1],costs[1]
    rows=[]
    for i,s in enumerate(settings):
        metrics=rankings(costs[i],truecost,paths[i],truth,cfg['replay_top_k'])
        rows.append(dict(s,seed=record['seed'],split=record['split'],category=record['category'],state_id=record['id'],
                         environment=record['source']['environment'],**metrics))
    # Endpoint equivalence is tested on identical candidates, including invalid-prediction masks.
    for i,s in enumerate(settings):
        if s['alpha']==0 and s['method']!='oracle':
            if not np.array_equal(paths[i],paths[0],equal_nan=True):raise AssertionError('alpha0 differs from Frozen')
    for item in search_records:
        item['regret_vs_shared_pool']=item['true_cost']-float(np.min(truecost))
    # Store all candidate paths/costs, not only the preferred plan.
    clearance=2.4-np.max(np.abs(paths[:,:,:,0]),axis=2)
    failures=(clearance<0)|(~np.isfinite(paths).all((2,3)))|(np.abs(paths[:,:,:,1:]).max((2,3))>100)
    # Episode-style cost is separate from the terminal/soft-constraint planning objective.
    from adaptive_world_model.control.evaluation_mpc import stage_cost
    actual_actions=candidates.cpu()
    actual_states=torch.from_numpy(truth)
    stage=stage_cost(actual_states,actual_actions,'swingup').numpy()
    bad=(np.abs(truth[:,:,0])>2.4)|(~np.isfinite(truth).all(2))|(np.abs(truth[:,:,1:]).max(2)>100)
    alive=np.concatenate([np.ones((len(truth),1),bool),~np.maximum.accumulate(bad,axis=1)[:,:-1]],axis=1)
    no_penalty=np.where(alive,np.nan_to_num(stage,nan=1e20,posinf=1e20),0).sum(1)
    save_npz(target.with_suffix('.npz'),candidates=candidates.cpu().numpy(),predicted_paths=paths,predicted_costs=costs,
             true_paths=truth,true_costs=truecost,candidate_sources=np.array(pool_labels),initial=state.cpu().numpy(),
             predicted_rail_clearance=clearance,true_rail_clearance=clearance[1],predicted_failure=failures,true_failure=failures[1],
             true_control_cost_without_failure_penalty=no_penalty,true_control_cost=no_penalty+cfg['failure_penalty']*failures[1])
    packet=dict(config_sha256=ctx.digest,record=record,rows=rows,search=search_records,wall_seconds=time.perf_counter()-start,
                supports_available=0 if support is None else len(support),support_hash='none' if support is None else support.support_hash(),
                support_provenance='fixed: exact archived independent support; online: archived source controller past only, shared for shadow comparison')
    save_json(target,packet)
    print(f'REPLAY {record["id"]} candidates={len(candidates)} seconds={packet["wall_seconds"]:.1f}',flush=True)
    return packet

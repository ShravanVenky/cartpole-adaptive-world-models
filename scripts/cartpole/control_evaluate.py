"""V2 control evaluation, deployment adaptation only. Run --help for stages."""
import sys, json, time, math, hashlib, argparse, copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from dataclasses import asdict
from adaptive_world_model.experiments.v1_baseline import load_baselines
from adaptive_world_model.experiments.cartpole_v2 import load_locked_config, load_new_adapter, method_spec, model_for
from adaptive_world_model.data import collect_sequential, TransitionBatch
from adaptive_world_model.environments.cartpole import CartPolePhysics,step,wrap_angle
from adaptive_world_model.control.evaluation_mpc import EvaluationCEM,GraphedPlanner,physics_step,stage_cost
from adaptive_world_model.control.batched_predictor import BatchedPredictor
from adaptive_world_model.utils.randomness import stable_seed
from adaptive_world_model.utils.artifacts import write_json

V1=ROOT/'results/cartpole/20261005_232356_821640_cartpole_v1'
V2=ROOT/'results/cartpole/20261006_140915_614361_cartpole_v2_factorial'
RUN=ROOT/'results/cartpole/20261023_v2_control'
METHODS=['frozen','ridge','meta_residual_1step','meta_residual_rollout']


def configuration():
    cfg=load_locked_config(V2)
    return {'version':1,'seeds':cfg['seeds'],'episodes_per_seed':3,'budgets':[0,5,10,20,50],
      'environments':[{'name':'nominal','category':'nominal'}]+[x for x in cfg['test_environments'] if x['category']!='ID'],
      'horizons':[5,15],'swingup_extra_horizon':None,'balance_steps':400,'swingup_steps':600,
      'candidates':128,'iterations':3,'elites':16,'rail_limit':2.4,'action_limit':10,
      'online_shift_step':150,'large_angle_shift_step':25,'disturbance_step':150,
      'disturbance_delta':[0,.8,.5,1.0],'online_updates':[5,10,20,50],
      'success_angle':.15,'success_omega':.8,'sustained_steps':100,'recovery_consecutive_steps':25,
      'oracle_required_success':.875,'oracle_validation_episodes':8,
      'bootstrap_seed':4123,'bootstrap_samples':2000,
      'noise':'none','support':'independent passive sequential; matched prefixes; separate seed namespace',
      'online_protocol':'known shift; reset shared initialization and refit cumulative post-change prefixes at5/10/20/50 actual controller transitions; no detector/probes',
      'angle':'unwrapped state, sin/cos features, wrapped error; no small-angle termination',
      'source_v2':str(V2),'methods':METHODS,'offline_training':False}


def load_models(seed):
    c=load_locked_config(V2);base,models=load_baselines(V1,seed)
    spec=method_spec('meta_residual_rollout',c['selected'])
    adapter=load_new_adapter(V2/'checkpoints'/f'meta_residual_rollout_seed{seed}.pt',base,spec,c,seed)
    models['meta_residual_rollout']=model_for(base,adapter,c,'residual')
    return base,models


def initial(task,seed,trial):
    g=torch.Generator().manual_seed(stable_seed(seed,'control_initial',task,trial))
    s=torch.randn(4,generator=g)*torch.tensor([.1,.1,.05,.1])
    if task=='swingup':s[2]+=math.pi
    return s


def batch_data(states,actions,next_states,key):
    n=len(states)
    return TransitionBatch(torch.stack(states).cpu(),torch.stack(actions).cpu(),torch.stack(next_states).cpu(),
        torch.zeros(n,dtype=torch.long),torch.arange(n),provenance=key)


def run_group(cfg,seed,env,task,h,protocol,device):
    key=f"{protocol}_{task}_H{h}_{env['name']}_seed{seed}"
    target=RUN/'episodes'/key
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if target.with_suffix('.json').exists():
        saved=json.loads(target.with_suffix('.json').read_text())
        if saved.get('config_sha256')!=digest:raise RuntimeError('Cached condition config mismatch: '+key)
        return
    base,templates=load_models(seed);nominal=CartPolePhysics();physical={**asdict(nominal),**{k:v for k,v in env.items() if k not in ('name','category')}}
    physics=CartPolePhysics(**physical)
    labels=[];models=[];supports=[];adaptlogs=[]
    support_cache={trial:collect_sequential(physics,50,stable_seed(seed,'control_support',env['name'],task,trial),
        'perturbed_balance' if task=='balance' else 'broad','cpu',stream='control_independent_support') for trial in range(cfg['episodes_per_seed'])}
    for method in METHODS:
        budgets=[0] if method=='frozen' else cfg['budgets'] if protocol=='fixed' else [0,50] if protocol=='disturbance' else [0,1]
        for budget in budgets:
            for trial in range(cfg['episodes_per_seed']):
                model=base if method=='frozen' else copy.deepcopy(templates[method]);model.reset_adaptation()
                support=support_cache[trial]
                log=None
                if budget and protocol!='online':log=model.adapt(support.prefix(budget))
                labels.append({'method':method,'budget':budget,'trial':trial});models.append(model)
                supports.append(support.prefix(budget).support_hash() if budget and protocol!='online' else 'none');adaptlogs.append(log)
    for trial in range(cfg['episodes_per_seed']):
        labels.append({'method':'oracle','budget':0,'trial':trial});models.append(base);supports.append('privileged_true_physics');adaptlogs.append(None)
    b=len(models);pack=BatchedPredictor(base,models,device)
    oracle_mask=torch.tensor([x['method']=='oracle' for x in labels],device=device)
    def predict(s,u):
        values=pack(s,u)
        n_oracle=cfg['episodes_per_seed']
        values[-n_oracle:]=physics_step(s[-n_oracle:],u[-n_oracle:],current_p)
        return values
    states=torch.stack([initial(task,seed,x['trial']) for x in labels]).to(device)
    cem=EvaluationCEM(h,task,b,cfg['candidates'],cfg['iterations'],cfg['elites'],device)
    count=cfg['balance_steps'] if task=='balance' else cfg['swingup_steps']
    change=cfg['online_shift_step'] if task=='balance' else cfg['large_angle_shift_step']
    ss=[states.clone()];aa=[];prediction=[];updates=[];elapsed=[];dead=torch.zeros(b,dtype=torch.bool,device=device)
    snapshots=[(0,{k:v.clone() for k,v in pack.weights.items()},pack.coef.clone())]
    current_p=physics
    noise_groups=torch.tensor([x['trial'] for x in labels],device=device)
    graph_target=GraphedPlanner(cem,predict,states,noise_groups) if device=='cuda' else None
    graph_nominal=None
    if device=='cuda' and protocol=='online':
        current_p=nominal;graph_nominal=GraphedPlanner(cem,predict,states,noise_groups);current_p=physics
    cem.mean.zero_()
    started=time.perf_counter();failure_step=torch.full((b,),count,device=device,dtype=torch.long)
    for t in range(count):
        current_p=nominal if protocol=='online' and t<change else physics
        if protocol=='disturbance' and t==cfg['disturbance_step']:
            states=states+torch.tensor(cfg['disturbance_delta'],device=device);ss[-1]=states.clone()
        begin=time.perf_counter()
        with torch.no_grad():
            planning_seed=stable_seed(seed,'mpc',task,env['name'],t)
            graph=graph_nominal if protocol=='online' and t<change else graph_target
            action=graph.action(states,planning_seed) if graph is not None else cem.action(predict,states,planning_seed)
            predicted=predict(states,action)
            following=physics_step(states,action,current_p)
            bad=(~torch.isfinite(following).all(-1)) | (following[:,0].abs()>cfg['rail_limit']) | (following[:,1:].abs()>100).any(-1)
            was_dead=dead.clone();failure_step=torch.where(bad&~dead,t+1,failure_step);dead |= bad
            # Absorbing failure, no reset or fabricated successful trajectory.
            following=torch.where(was_dead[:,None],states,following)
        if device=='cuda':torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-begin)
        aa.append(action.detach());prediction.append(predicted.detach());ss.append(following.detach())
        states=following
        observed=t-change+1
        if protocol=='online' and observed in cfg['online_updates']:
            entry={'step':t+1,'observations':observed,'methods':[]}
            for i,label in enumerate(labels):
                if label['budget'] and label['method'] not in ('frozen','oracle') and not dead[i].item():
                    data=batch_data([x[i] for x in ss[change:t+1]],[x[i] for x in aa[change:t+1]],
                         [x[i] for x in ss[change+1:t+2]],f'actual/{key}/row{i}')
                    models[i].reset_adaptation();log=models[i].adapt(data);entry['methods'].append({'row':i,**log})
            pack.refresh(models);updates.append(entry)
            snapshots.append((t+1,{k:v.clone() for k,v in pack.weights.items()},pack.coef.clone()))
    states=torch.stack(ss,1).cpu();actions=torch.stack(aa,1).cpu();pred=torch.stack(prediction,1).cpu()
    recursive={horizon:[] for horizon in (5,15,100)};frozen_recursive={horizon:[] for horizon in (5,15,100)}
    forecast_predictions=[];forecast_starts=[];shared_sum=torch.zeros(b,device=device);shared_count=torch.zeros(b,device=device)
    with torch.no_grad():
        for si,(start,weights,coef) in enumerate(snapshots):
            end=snapshots[si+1][0] if si+1<len(snapshots) else count
            for k,v in weights.items():pack.weights[k].copy_(v)
            pack.coef.copy_(coef)
            peers=[x['trial'] for x in labels]
            shared_state=states[peers,start:end].to(device);shared_action=actions[peers,start:end].to(device)
            shared_truth=states[peers,start+1:end+1].to(device)
            shadow=pack(shared_state,shared_action);se=shadow-shared_truth;se[...,2]=wrap_angle(se[...,2])
            mask=torch.arange(start,end,device=device)[None]<failure_step[peers,None]
            shared_sum+=((se/pack.base.normalizer.y_scale).square().mean(-1)*mask).sum(1)
            shared_count+=mask.sum(1)
            starts=list(range(start,min(end,count-100+1),25))
            if not starts:continue
            for k,v in weights.items():pack.weights[k].copy_(v)
            pack.coef.copy_(coef)
            current=states[:,starts].to(device);frozen=current.clone();path=[]
            total=torch.zeros(b,len(starts),device=device);ftotal=total.clone()
            start_tensor=torch.tensor(starts,device=device)
            for horizon in range(1,101):
                u=actions[:,[x+horizon-1 for x in starts]].to(device)
                old_oracle=current[-cfg['episodes_per_seed']:]
                current=pack(current,u)
                oracle_next=physics_step(old_oracle,u[-cfg['episodes_per_seed']:],physics)
                if protocol=='online':
                    oracle_next=torch.where((start_tensor>=change)[None,:,None],oracle_next,physics_step(old_oracle,u[-cfg['episodes_per_seed']:],nominal))
                current[-cfg['episodes_per_seed']:]=oracle_next
                x,v,theta,w=frozen.unbind(-1);z=torch.stack([x,v,theta.sin(),theta.cos(),w,u.squeeze(-1)],-1)
                norm=pack.base.normalizer
                frozen=frozen+norm.decode_y(pack.base.network(norm.encode_x(z)))
                truth=states[:,[x+horizon for x in starts]].to(device)
                err=current-truth;err[...,2]=wrap_angle(err[...,2]);fe=frozen-truth;fe[...,2]=wrap_angle(fe[...,2])
                scale=torch.tensor([2.4,3.,math.pi,4.],device=device)
                total+=(err/scale).square().mean(-1);ftotal+=(fe/scale).square().mean(-1)
                path.append(current.cpu())
                if horizon in recursive:
                    valid=start_tensor[None]+horizon<=failure_step[:,None]
                    recursive[horizon].append((total/horizon).masked_fill(~valid,float('nan')).cpu())
                    frozen_recursive[horizon].append((ftotal/horizon).masked_fill(~valid,float('nan')).cpu())
            forecast_predictions.append(torch.stack(path,2));forecast_starts.extend(starts)
    errors=pred-states[:,1:];errors[:,:,2]=wrap_angle(errors[:,:,2]);scale=base.normalizer.y_scale
    upright=(wrap_angle(states[:,:,2]).abs()<cfg['success_angle'])&(states[:,:,3].abs()<cfg['success_omega'])
    rail=(states[:,:,0].abs()<=cfg['rail_limit']);rows=[]
    for i,label in enumerate(labels):
        good=upright[i]&rail[i];runs=good.unfold(0,cfg['recovery_consecutive_steps'],1).all(1)
        reached=torch.where(runs)[0];fail=torch.where(~rail[i])[0]
        event=change if protocol=='online' else cfg['disturbance_step'] if protocol=='disturbance' else 0
        recovered=torch.where(runs[event:])[0]
        valid=int(failure_step[i]);cost=float(stage_cost(states[i,1:valid+1],actions[i,:valid],task).sum())
        failed=bool(dead[i]);cost+=1e5*failed
        rows.append({**label,'seed':seed,'environment':env['name'],'category':env['category'],'task':task,'horizon':h,'protocol':protocol,
          'success':bool(good[-cfg['sustained_steps']:].all() and not failed), 'failed':failed,'cost':cost,
          'survival_steps':int(failure_step[i]),'stabilization_step':int(reached[0]) if len(reached) else None,
          'recovery_steps':int(recovered[0])+cfg['recovery_consecutive_steps'] if len(recovered) else None,
          'one_step_nmse':float((errors[i,:valid]/scale).square().mean()),'action_energy':float(actions[i,:valid].square().sum()),
          'mpc_seconds_per_step':float(np.mean(elapsed)),'group_seconds':time.perf_counter()-started,
          'support_hash':supports[i] if protocol!='online' else 'controller_generated_own_stream',
          'adaptation_seconds':0 if adaptlogs[i] is None else adaptlogs[i]['adaptation_seconds']})
        tipped=torch.where(wrap_angle(states[i,:,2]).abs()>.5)[0]
        rows[-1]['balance_survival_steps']=int(tipped[0]) if len(tipped) else count
        rows[-1]['reached_upright']=bool(len(reached))
        rows[-1]['settled_then_lost']=bool(len(reached) and not good[-cfg['sustained_steps']:].all())
        if task=='balance' and protocol=='fixed':rows[-1]['success'] &= not len(tipped)
        for horizon in recursive:
            rows[-1][f'recursive_{horizon}_nmse']=float(torch.cat(recursive[horizon],1)[i].nanmean()) if recursive[horizon] else None
            rows[-1][f'frozen_shadow_{horizon}_nmse']=float(torch.cat(frozen_recursive[horizon],1)[i].nanmean()) if frozen_recursive[horizon] else None
        rows[-1]['shared_frozen_trajectory_one_step_nmse']=float(shared_sum[i]/shared_count[i].clamp_min(1)) if label['method']!='oracle' else None
    np.savez_compressed(target.with_suffix('.npz'),states=states.numpy(),actions=actions.numpy(),one_step_prediction=pred.numpy(),
        dead=dead.cpu().numpy(),failure_step=failure_step.cpu().numpy(),forecast_starts=np.array(forecast_starts),
        forecast_states=torch.cat(forecast_predictions,1).numpy() if forecast_predictions else np.empty((b,0,100,4)))
    temporary=target.with_suffix('.tmp')
    write_json(temporary,{'key':key,'config_sha256':digest,'rows':rows,'updates':updates,'physics':physical,'labels':labels})
    temporary.replace(target.with_suffix('.json'))
    base.assert_frozen();print(f'COMPLETE {key} {time.perf_counter()-started:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','smoke','full']);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    torch.set_num_threads(1);RUN.mkdir(exist_ok=True);(RUN/'episodes').mkdir(exist_ok=True)
    if args.stage=='prepare':write_json(RUN/'config_predeclared.json',configuration());return
    cfg=json.loads((RUN/'config_locked.json').read_text())
    if not json.loads((RUN/'oracle_gate.json').read_text())['passed']:raise RuntimeError('Oracle validation gate not passed; main sweep prohibited')
    if args.stage=='smoke':
        globals()['RUN']=ROOT/'smoke tests/cartpole_control_20261023'
        (RUN/'episodes').mkdir(parents=True,exist_ok=True)
        run_group(cfg,11,cfg['environments'][0],'balance',5,'fixed',args.device);return
    write_json(RUN/'status.json',{'complete':False,'phase':'full_evaluation'})
    from datetime import datetime,timezone
    if not (RUN/'execution_start.json').exists():
        write_json(RUN/'execution_start.json',{'utc':datetime.now(timezone.utc).isoformat(),'device':args.device,
            'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else None,'torch':torch.__version__})
    for seed in cfg['seeds']:
        for env in cfg['environments']:
            for task in ['balance','swingup']:
                horizons=cfg['horizons']+([cfg['swingup_extra_horizon']] if task=='swingup' and cfg['swingup_extra_horizon'] else [])
                for h in horizons:
                    for protocol in ['fixed','disturbance','online']:
                        if protocol=='online' and env['name']=='nominal':continue
                        run_group(cfg,seed,env,task,h,protocol,args.device)
    write_json(RUN/'status.json',{'complete':False,'phase':'evaluation_saved_analysis_pending'})
    from control_report import generate
    generate(RUN,cfg)

if __name__=='__main__':main()

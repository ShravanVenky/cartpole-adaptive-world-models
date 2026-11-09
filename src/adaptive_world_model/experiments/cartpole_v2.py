"""Controlled plasticity/objective experiment; never retrains the V1 base."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
import statistics
from pathlib import Path
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from . import cartpole_v1 as v1
from .v1_baseline import (IDENTIFIERS, ReferenceMetrics, baseline_config, final_data,
                          load_baselines, sampled_reproduction, verify_frozen)
from ..adaptation import AdaptationSettings, MetaResidualModel
from ..adaptation.lora import MetaLoRAModel
from ..adaptation.rollout_objective import neural_meta_objective
from ..data import collect_iid, collect_sequential
from ..environments import CartPolePhysics
from ..evaluation import one_step_metrics, rollout_metrics, recovery_ratio, fraction_error_removed
from ..models import ResidualAdapter
from ..models.lora import LoRAAdapter
from ..models.dynamics import PredictionSafetyError
from ..utils.artifacts import append_jsonl, new_run_directory, write_json
from ..utils.randomness import seed_everything, stable_seed
from ..utils.runtime import Deadline, hardware_info

ROOT = v1.ROOT
METHODS = ['frozen', 'ridge', 'meta_residual_1step', 'meta_residual_rollout', 'meta_lora_1step', 'meta_lora_rollout']
NEW_METHODS = METHODS[3:]


def baseline_path(config): return ROOT/config['baseline_run']


def load_config(path=None):
    path = Path(path or ROOT/'configs/cartpole/v2.json')
    descriptor = json.loads(path.read_text(encoding='utf-8'))
    old = baseline_config(baseline_path(descriptor))
    config = copy.deepcopy(old)
    config.update(descriptor)
    for section in ('train', 'development_search', 'final_test'):
        content = json.loads((ROOT/descriptor['config_files'][section]).read_text(encoding='utf-8'))
        config[section] = content
    config['test_environments'] = config['final_test']['test_environments']
    config['evaluation'] = config['final_test']['evaluation']
    validate_config(config)
    return config


def validate_config(config):
    old = baseline_config(baseline_path(config))
    for name in ('seeds', 'nominal', 'meta_family', 'development', 'ridge', 'adaptation', 'test_environments'):
        if config[name] != old[name]: raise ValueError(f'V2 may not change the V1 protocol: {name}')
    for name in ('K', 'regimes', 'support_modes', 'trials_per_seed', 'query_steps', 'query_trajectories',
                 'rollout_trajectories', 'rollout_state_scale'):
        if config['evaluation'][name] != old['evaluation'][name]: raise ValueError(f'V2 changed final evaluation: {name}')
    if config['evaluation']['methods'] != METHODS or config['evaluation']['rollout_horizons'] != [1,5,20,50,100]:
        raise ValueError('V2 requires the complete factorial matrix and legacy core horizons')
    settings = config['train']
    for key in ('steps','tasks_per_step','validation_every','workers','cpu_threads'):
        if type(settings[key]) is not int or settings[key] < 1: raise ValueError(f'Invalid train.{key}')
    if settings['first_order'] is not False: raise ValueError('Primary V2 uses full MAML')
    if not math.isfinite(settings['learning_rate']) or settings['learning_rate'] <= 0: raise ValueError('Invalid outer learning rate')
    if settings['steps'] != old['meta']['steps'] or settings['tasks_per_step'] != old['meta']['tasks_per_step']:
        raise ValueError('Final neural methods must use the same training-step/task budget as V1')
    for key in ('nominal', 'meta_family'):
        if settings[key] != config[key]: raise ValueError(f'Shadow train.{key} differs from the active protocol')
    search = config['development_search']
    if search['environments'] != config['development']: raise ValueError('Shadow development environments differ')
    if search['layer_indices'] != [0,2]: raise ValueError('Primary placement must remain the two hidden Linear layers')
    for key, target in (('inner_learning_rate','learning_rate'),('inner_steps','steps')):
        if search[key] != config['adaptation'][target]: raise ValueError(f'Shadow development.{key} differs')
    if settings['nominal_preservation_weight'] != 0: raise ValueError('Primary experiment records nominal diagnostics with preservation weight zero')
    if config['final_test']['test_environments'] != config['test_environments'] or config['final_test']['evaluation'] != config['evaluation']:
        raise ValueError('Final-test shadow configuration differs from the active protocol')
    AdaptationSettings(**config['adaptation'])


def locked_configuration(config, selection):
    locked=copy.deepcopy(config); locked['selected']=copy.deepcopy(selection)
    locked['development_selection']=copy.deepcopy(selection)
    locked['rollout_loss']={'horizon':selection['horizon'],'rollout_weight':selection['rollout_weight'],
        'one_step_weight':1.0,'temporal_weights':'uniform','state_scale':'frozen nominal delta scale',
        'inner_objective':'one_step','outer_objective':'one_step + rollout_weight * recursive_rollout'}
    locked['lora']={'rank':selection['rank'],'alpha':selection['rank'],'scale':1.0,
        'layer_indices':selection['layer_indices'],'initialization':'Kaiming A, zero B; meta-learn both factors'}
    return locked


def load_locked_config(run):
    original=json.loads((run/'config.json').read_text(encoding='utf-8')); validate_config(original)
    selection=json.loads((run/'development_selection.json').read_text(encoding='utf-8'))
    search=original['development_search']
    option={'horizon':selection['horizon'],'rollout_weight':selection['rollout_weight']}
    if selection['rank'] not in search['ranks'] or option not in search['rollout_options']:
        raise ValueError('Locked choice is outside the predeclared development search')
    if selection['layer_indices']!=search['layer_indices'] or selection['inner_learning_rate']!=original['adaptation']['learning_rate'] or selection['inner_steps']!=original['adaptation']['steps']:
        raise ValueError('Locked placement/inner adaptation differs from the predeclared protocol')
    locked=json.loads((run/'config_locked.json').read_text(encoding='utf-8')); validate_config(locked)
    if locked!=locked_configuration(original,selection): raise ValueError('Locked configuration differs from the saved original and development choice')
    return locked


def stage_status(run, stage, **details):
    path = run/'status.json'; saved = json.loads(path.read_text()) if path.exists() else {'complete':False,'stages':{}}
    saved['stages'][stage] = details
    saved['complete'] = all(saved['stages'].get(name,{}).get('complete',False) for name in ('prepare','develop','train','evaluate'))
    write_json(path, saved); return saved


def prepare(config):
    verify_frozen(baseline_path(config), ROOT/config['baseline_manifest'])
    run = new_run_directory(ROOT, config, 'cartpole_v2_factorial')
    write_json(run/'hardware.json', {**hardware_info(torch.device('cpu')), 'cpu_workers':config['train']['workers'],
                                    'cpu_threads_per_worker':config['train']['cpu_threads'], 'logical_cpu_count':os.cpu_count()})
    reference_dir = run/'checkpoints/reference'; reference_dir.mkdir()
    for seed in config['seeds']:
        for prefix in ('base','meta'):
            name = f'{prefix}_seed{seed}.pt'; shutil.copy2(baseline_path(config)/'checkpoints'/name, reference_dir/name)
    write_json(run/'baseline_checkpoint_hashes.json', {p.name:v1.fingerprint(p) for p in reference_dir.iterdir()})
    checks = sampled_reproduction(baseline_path(config))
    write_csv(run/'baseline_reproduction_sample.csv', checks)
    stage_status(run,'prepare',complete=True,baseline_checks=len(checks),all_baseline_checks_passed=True)
    print(f'RUN_DIRECTORY={run}',flush=True)
    return run


def model_for(base, adapter, config, kind):
    settings = AdaptationSettings(**config['adaptation'])
    return MetaResidualModel(base, adapter, settings) if kind=='residual' else MetaLoRAModel(base,adapter,settings)


def initialize(base, seed, spec):
    if spec['kind']=='residual': return ResidualAdapter(stable_seed(seed,'adapter'), (16,16))
    return LoRAAdapter(base, rank=spec['rank'], layer_indices=tuple(spec['layer_indices']),
                       seed=stable_seed(seed,'lora'), alpha=spec['rank'])


def development_metrics(base, adapter, config, tasks, spec):
    model = model_for(base,adapter,config,spec['kind']); scores=[]; logs=[]
    inner = AdaptationSettings(**config['adaptation'])
    for support, query in tasks:
        for k in config['meta']['support_K']:
            model.reset_adaptation(); adaptation = model.adapt(support.prefix(k))
            metric = one_step_metrics(model,query)
            try:
                _, components = neural_meta_objective(base,adapter,support.prefix(k),query,inner,
                   objective=spec['objective'],horizon=spec.get('horizon',5),rollout_weight=spec.get('rollout_weight',0),
                   kind=spec['kind'],first_order=False)
            except PredictionSafetyError as exc:
                components={'one_step_loss':None,'rollout_loss':None,'total_loss':None,'reason':str(exc)}
            logs.append({'K':k,'support_provenance':support.provenance,'adaptation':adaptation,'metrics':metric,'loss_components':components})
            if not adaptation['accepted'] or not metric['finite']: return float('inf'),logs
            scores.append(metric['nmse'])
    model.reset_adaptation(); base.assert_frozen()
    return sum(scores)/len(scores),logs


def nominal_initialization_metrics(base, adapter, config, seed):
    physics = CartPolePhysics(**config['nominal']); model = model_for(base,adapter,config,
        'lora' if isinstance(adapter,LoRAAdapter) else 'residual')
    records=[]
    for regime in config['evaluation']['regimes']:
        batch = collect_iid(physics,128,stable_seed(seed,'v2_nominal_check',regime),regime,'cpu',stream='nominal_init_diagnostic')
        records.append({'regime':regime,'frozen':one_step_metrics(base,batch),'unadapted':one_step_metrics(model,batch)})
    model.reset_adaptation(); return records


def train_one(run, config, seed, spec, pool, tasks, steps, tasks_per_step, output, deadline, pilot=False):
    base,_=load_baselines(baseline_path(config),seed)
    adapter=initialize(base,seed,spec)
    optimizer=torch.optim.Adam(adapter.parameters(),lr=config['train']['learning_rate'])
    sampler=torch.Generator().manual_seed(stable_seed(seed,'meta_batches'))
    resume=output.with_suffix('.resume.pt'); iteration=0; best=float('inf'); best_state=None; outer_updates=0; selected_step=None
    provenance={'config_hash':v1.config_hash(config),'spec':spec,'seed':seed,
                'base_sha256':v1.fingerprint(baseline_path(config)/'checkpoints'/f'base_seed{seed}.pt')}
    ordinary_nominal=nominal_initialization_metrics(base,adapter,config,seed)
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved['provenance']!=provenance: raise RuntimeError('V2 resume provenance mismatch')
        adapter.load_state_dict(saved['current']); optimizer.load_state_dict(saved['optimizer']); sampler.set_state(saved['sampler'])
        iteration=saved['iteration']; best=saved['best']; best_state=saved['best_state']; outer_updates=saved['outer_updates']
        selected_step=saved.get('selected_step')
    def save_resume():
        v1.atomic_save(resume,{'current':adapter.state_dict(),'optimizer':optimizer.state_dict(),'sampler':sampler.get_state(),
            'iteration':iteration,'best':best,'best_state':best_state,'selected_step':selected_step,
            'outer_updates':outer_updates,'provenance':provenance})
    if not resume.exists(): save_resume()
    settings=AdaptationSettings(**config['adaptation']); started=time.perf_counter()
    log_path=run/'logs'/f"{'pilot_' if pilot else ''}{spec['method']}_seed{seed}.jsonl"
    validation_every=min(config['train']['validation_every'],steps)
    for iteration in range(iteration+1,steps+1):
        deadline.check(); optimizer.zero_grad(set_to_none=True)
        selected=torch.randint(len(pool),(tasks_per_step,),generator=sampler).tolist()
        losses=[]; logs=[]
        for slot,task in enumerate(selected):
            support,query=pool[task]; k=config['meta']['support_K'][(iteration+slot)%len(config['meta']['support_K'])]
            try:
                loss,log=neural_meta_objective(base,adapter,support.prefix(k),query,settings,
                     objective=spec['objective'],horizon=spec.get('horizon',5),rollout_weight=spec.get('rollout_weight',0),
                     first_order=False,kind=spec['kind'])
                if log['accepted']: losses.append(loss)
                logs.append({'task':task,'K':k,**log})
            except PredictionSafetyError as exc:
                logs.append({'task':task,'K':k,'accepted':False,'reason':str(exc)})
        applied=False; gradient=None; outer_rejection=None; outer_update_norm=None
        if losses:
            objective=torch.stack(losses).mean(); objective.backward()
            if all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in adapter.parameters()):
                try:
                    gradient=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0,error_if_nonfinite=True))
                except RuntimeError:
                    outer_rejection='nonfinite aggregate gradient norm'
                if outer_rejection is None:
                    previous={k:v.detach().clone() for k,v in adapter.state_dict().items()}; old_optimizer=copy.deepcopy(optimizer.state_dict())
                    optimizer.step(); applied=all(bool(torch.isfinite(p).all()) for p in adapter.parameters())
                    if not applied:
                        adapter.load_state_dict(previous); optimizer.load_state_dict(old_optimizer)
                        outer_rejection='nonfinite optimizer candidate; rolled back parameters and Adam state'
                    else:
                        outer_update_norm=float(torch.sqrt(sum((value.detach()-previous[key]).square().sum() for key,value in adapter.state_dict().items())))
            else: outer_rejection='missing or nonfinite adapter gradient'
            optimizer.zero_grad(set_to_none=True)
        else: outer_rejection='no accepted meta tasks'
        if applied: outer_updates+=1
        component_keys=('one_step_loss','rollout_loss','total_loss')
        component_values={key:[log[key] for log in logs if isinstance(log.get(key),(int,float)) and math.isfinite(log[key])] for key in component_keys}
        mean_components={key:statistics.fmean(values) if values else None for key,values in component_values.items()}
        component_std={key:statistics.pstdev(values) if values else None for key,values in component_values.items()}
        append_jsonl(log_path,{'seed':seed,'method':spec['method'],'step':iteration,'outer_update_applied':applied,
                       'outer_gradient_norm':gradient,'outer_update_norm':outer_update_norm,
                       'outer_rejection':outer_rejection,'components':mean_components,'component_std':component_std,
                       'component_counts':{key:len(values) for key,values in component_values.items()},
                       'tasks':logs,'elapsed_seconds':time.perf_counter()-started})
        if iteration%validation_every==0 or iteration==steps:
            score,validation=development_metrics(base,adapter,config,tasks,spec)
            append_jsonl(run/'logs'/f"validation_{spec['method']}_seed{seed}{'_pilot' if pilot else ''}.jsonl",
                          {'seed':seed,'step':iteration,'one_step_selection_score':score,'tasks':validation})
            if math.isfinite(score) and score<best:
                best=score; best_state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}; selected_step=iteration
            save_resume(); base.assert_frozen()
            write_json(run/'logs'/f"progress_{spec['method']}_seed{seed}{'_pilot' if pilot else ''}.json",
                       {'step':iteration,'steps':steps,'method':spec['method'],'seed':seed,'elapsed_seconds':time.perf_counter()-started,'complete':False})
            print(f"{spec['method']} seed={seed} step={iteration}/{steps} dev_one_step={score:.6g}",flush=True)
    if best_state is None or outer_updates==0: raise RuntimeError(f'No valid trained checkpoint: {spec["method"]}, seed {seed}')
    adapter.load_state_dict(best_state); adapter.meta_trained=True
    adapter.save_checkpoint(output,metadata={**provenance,'training_seed':seed,'steps_completed':iteration,
                 'outer_updates':outer_updates,'selected_step':selected_step,'development_nmse':best,'checkpoint_selection':'mean one-step development NMSE as V1',
                 'ordinary_nominal_diagnostic':ordinary_nominal,'meta_nominal_diagnostic':nominal_initialization_metrics(base,adapter,config,seed)})
    base.assert_frozen()
    return adapter


def candidate_score(base,adapter,config,seed,spec,tasks):
    model=model_for(base,adapter,config,spec['kind']); one=[]; long=[]; logs=[]
    index=0
    for environment in config['development']:
        for regime in config['evaluation']['regimes']:
            support,query=tasks[index]; index+=1
            physics=v1.physics_for(config,environment)
            raw=collect_sequential(physics,100,stable_seed(seed,'v2_development_rollout',environment['name'],regime),regime,'cpu',4,
                                   stream='development/selection_rollout')
            support.assert_disjoint(raw); query.assert_disjoint(raw)
            inputs=v1.rollout_inputs(raw,100,4); frozen_one=one_step_metrics(base,query)['nmse']
            frozen_long=rollout_metrics(base,*inputs,(100,),config['evaluation']['rollout_state_scale'])[100]['nmse']
            for k in config['meta']['support_K']:
                model.reset_adaptation(); adaptation=model.adapt(support.prefix(k)); metric=one_step_metrics(model,query)
                rollout=rollout_metrics(model,*inputs,(100,),config['evaluation']['rollout_state_scale'])[100]
                if not adaptation['accepted'] or not metric['finite'] or not rollout['finite']: return float('inf'),logs
                one.append(metric['nmse']/max(frozen_one,1e-12)); long.append(rollout['nmse']/max(frozen_long,1e-12))
                logs.append({'environment':environment['name'],'regime':regime,'K':k,'one_step_relative_frozen':one[-1],
                             'rollout_100_relative_frozen':long[-1]})
    return .5*(sum(one)/len(one)+sum(long)/len(long)),logs


def development(run,config):
    if (run/'development_selection.json').exists():
        selection=json.loads((run/'development_selection.json').read_text())
        if not (run/'config_locked.json').exists():
            locked=locked_configuration(config,selection)
            write_json(run/'config_locked.json',locked); write_json(run/'config_locked.yaml',locked)
        load_locked_config(run); stage_status(run,'develop',complete=True,selected=selection)
        return selection
    torch.set_num_threads(config['train']['cpu_threads']); stage_status(run,'develop',complete=False)
    settings=config['development_search']; seed=settings['pilot_seed']; deadline=Deadline(settings['max_minutes'])
    base,_=load_baselines(baseline_path(config),seed); old=baseline_config(baseline_path(config))
    pool=v1.make_meta_pool(old,seed,torch.device('cpu'),deadline); tasks=v1.development_tasks(old,seed,torch.device('cpu'))
    directory=run/'checkpoints/development'; directory.mkdir(exist_ok=True)
    candidates=[]
    for rank in settings['ranks']:
        candidates.append({'method':f'dev_lora_1step_r{rank}','kind':'lora','objective':'one_step','rank':rank,'layer_indices':[0,2]})
    for option in settings['rollout_options']:
        candidates.append({'method':f"dev_residual_rollout_h{option['horizon']}",'kind':'residual','objective':'rollout',**option})
        for rank in settings['ranks']:
            candidates.append({'method':f"dev_lora_rollout_r{rank}_h{option['horizon']}",'kind':'lora','objective':'rollout',
                               'rank':rank,'layer_indices':[0,2],**option})
    records=[]
    for spec in candidates:
        path=directory/(spec['method']+'.pt')
        if path.exists():
            adapter=load_new_adapter(path,base,spec,config,seed)
        else:
            adapter=train_one(run,config,seed,spec,pool,tasks,settings['pilot_steps'],settings['pilot_tasks_per_step'],path,deadline,pilot=True)
        score,logs=candidate_score(base,adapter,config,seed,spec,tasks)
        records.append({'spec':spec,'score':score,'evaluations':logs}); write_json(run/'development_scores.json',records)
    valid=[record for record in records if math.isfinite(record['score'])]
    if len(valid)!=len(records): raise RuntimeError('One or more development candidates failed; inspect logs before locking')
    # Joint development score chooses ONE shared rank and rollout objective across factorial cells.
    combinations=[]
    for rank in settings['ranks']:
        for option in settings['rollout_options']:
            matched=[r for r in records if (r['spec']['kind']=='lora' and r['spec']['objective']=='one_step' and r['spec']['rank']==rank)
                     or (r['spec']['kind']=='residual' and r['spec']['horizon']==option['horizon'])
                     or (r['spec']['kind']=='lora' and r['spec']['objective']=='rollout' and r['spec']['rank']==rank and r['spec']['horizon']==option['horizon'])]
            combinations.append({'rank':rank,**option,'score':sum(r['score'] for r in matched)/len(matched)})
    best=min(combinations,key=lambda x:x['score'])
    selection={**best,'layer_indices':[0,2],'inner_learning_rate':config['adaptation']['learning_rate'],
               'inner_steps':config['adaptation']['steps'],'selection_data':'development only; no final-test metrics read',
               'pilot_seed':seed,'pilot_steps':settings['pilot_steps'],'candidate_count':len(records),'joint_candidates':combinations}
    write_json(run/'development_selection.json',selection)
    locked=locked_configuration(config,selection)
    write_json(run/'config_locked.json',locked); write_json(run/'config_locked.yaml',locked)
    stage_status(run,'develop',complete=True,selected=selection)
    print(f'LOCKED_DEVELOPMENT={json.dumps(selection)}',flush=True)
    return selection


def method_spec(method,selected):
    return {'method':method,'kind':'residual' if method=='meta_residual_rollout' else 'lora',
            'objective':'one_step' if method=='meta_lora_1step' else 'rollout',
            'horizon':selected['horizon'],'rollout_weight':selected['rollout_weight'],
            'rank':selected['rank'],'layer_indices':selected['layer_indices']}


def load_new_adapter(path,base,spec,config,seed):
    adapter=ResidualAdapter.load_checkpoint(path) if spec['kind']=='residual' else LoRAAdapter.load_checkpoint(path,base)
    metadata=adapter.checkpoint_metadata
    if not adapter.meta_trained or metadata.get('seed')!=seed or metadata.get('config_hash')!=v1.config_hash(config) or metadata.get('spec')!=spec:
        raise RuntimeError('V2 adapter provenance mismatch')
    expected=v1.fingerprint(baseline_path(config)/'checkpoints'/f'base_seed{seed}.pt')
    if metadata.get('base_sha256')!=expected: raise RuntimeError('V2 adapter paired with wrong frozen base')
    return adapter


def train_seed(run_string,seed):
    run=Path(run_string); config=load_locked_config(run); torch.set_num_threads(config['train']['cpu_threads'])
    seed_everything(stable_seed(seed,'v2_training')); deadline=Deadline(config['train']['max_minutes_per_seed'])
    old=baseline_config(baseline_path(config)); pool=v1.make_meta_pool(old,seed,torch.device('cpu'),deadline)
    tasks=v1.development_tasks(old,seed,torch.device('cpu')); base,_=load_baselines(baseline_path(config),seed)
    try:
        for method in NEW_METHODS:
            spec=method_spec(method,config['selected']); path=run/'checkpoints'/f'{method}_seed{seed}.pt'
            if path.exists(): load_new_adapter(path,base,spec,config,seed); continue
            train_one(run,config,seed,spec,pool,tasks,config['train']['steps'],config['train']['tasks_per_step'],path,deadline)
        write_json(run/'logs'/f'train_seed{seed}_status.json',{'complete':True,'seed':seed})
    except BaseException as exc:
        write_json(run/'logs'/f'train_seed{seed}_status.json',{'complete':False,'seed':seed,'error':str(exc)})
        (run/'logs'/f'train_seed{seed}_failure.log').write_text(traceback.format_exc(),encoding='utf-8'); raise
    return seed


def size_fields(model,method):
    base=model if method=='frozen' else model.base
    adaptive=0 if method=='frozen' else model.coefficient_count if method=='ridge' else model.adapter.parameter_count
    return {'base_parameters':base.parameter_count,'adaptive_parameters':adaptive,'adaptive_percentage':100*adaptive/base.parameter_count}


def evaluate_seed(run_string,seed):
    run=Path(run_string); config=load_locked_config(run); old=baseline_config(baseline_path(config))
    torch.set_num_threads(config['train']['cpu_threads']); deadline=Deadline(config['evaluation']['max_minutes_per_seed'])
    base,models=load_baselines(baseline_path(config),seed); reference=ReferenceMetrics(baseline_path(config))
    hashes={'base':v1.fingerprint(baseline_path(config)/'checkpoints'/f'base_seed{seed}.pt'),
            'legacy_meta':v1.fingerprint(baseline_path(config)/'checkpoints'/f'meta_seed{seed}.pt')}
    for method in NEW_METHODS:
        spec=method_spec(method,config['selected']); path=run/'checkpoints'/f'{method}_seed{seed}.pt'
        adapter=load_new_adapter(path,base,spec,config,seed); models[method]=model_for(base,adapter,config,spec['kind']); hashes[method]=v1.fingerprint(path)
    directory=run/'logs/evaluation_units'/f'seed{seed}'; directory.mkdir(parents=True,exist_ok=True); completed=0
    for environment in config['test_environments']:
        for regime in config['evaluation']['regimes']:
            for trial in range(config['evaluation']['trials_per_seed']):
                data=None; baseline_metric=None; frozen_horizons=None; id_error=None
                for mode in config['evaluation']['support_modes']:
                    for k in config['evaluation']['K']:
                        deadline.check(); key=v1.unit_key(seed,environment,regime,trial,mode,k); path=directory/(key+'.json')
                        if path.exists():
                            saved=json.loads(path.read_text())
                            if saved['checkpoint_hashes']!=hashes or saved['config_hash']!=v1.config_hash(config):
                                raise RuntimeError('Evaluation resume fingerprint mismatch')
                            completed+=1; continue
                        if data is None:
                            data=final_data(old,seed,environment,regime,trial); baseline_metric=one_step_metrics(base,data['query'])
                            id_error=one_step_metrics(base,v1.nominal_counterfactual(data['query'],old))['nmse']
                            frozen_horizons=rollout_metrics(base,*data['rollout'],config['evaluation']['rollout_horizons'],config['evaluation']['rollout_state_scale'])
                        support=data[mode].prefix(k); support_hash=support.support_hash(); query_hash=data['query'].support_hash()
                        identifiers=dict(seed=seed,trial=trial,environment=environment['name'],category=environment['category'],
                                         regime=regime,support_mode=mode,K=k)
                        rows=[]; rollouts=[]; checks=[]; logs=[]
                        for method,model in models.items():
                            model.reset_adaptation(); log=model.adapt(support); metric=baseline_metric if method=='frozen' else one_step_metrics(model,data['query'])
                            horizons=frozen_horizons if method=='frozen' else rollout_metrics(model,*data['rollout'],config['evaluation']['rollout_horizons'],config['evaluation']['rollout_state_scale'])
                            row={**identifiers,'method':method,**v1.metric_row(metric),**size_fields(model,method),
                                 'E_before':baseline_metric['nmse'],'E_ID':id_error,'error_reduction':fraction_error_removed(metric['nmse'],baseline_metric['nmse']),
                                 'recovery_ratio':recovery_ratio(metric['nmse'],baseline_metric['nmse'],id_error),
                                 'adaptation_seconds':log['adaptation_seconds'],'steps':log['gradient_steps'],'support_count':k,
                                 'support_hash':support_hash,'query_hash':query_hash,'stable':bool(log['accepted'] and metric['finite']),
                                 'update_accepted':bool(log['accepted'])}
                            rows.append(row); logs.append({**log,'method':method})
                            for horizon,values in horizons.items():
                                rollouts.append({**identifiers,'method':method,'horizon':horizon,**v1.metric_row(values),**size_fields(model,method),
                                    'endpoint_nmse':values['endpoint_nmse'],'valid_steps':values['valid_steps'],'support_hash':support_hash,'query_hash':query_hash,
                                    'stable':bool(log['accepted'] and values['finite']),'update_accepted':bool(log['accepted'])})
                            if method in ('frozen','ridge','meta_residual_1step'):
                                checks.extend(reference.checks(identifiers,method,metric,horizons,support_hash,query_hash))
                            model.reset_adaptation(); base.assert_frozen()
                        if not all(check['passed'] for check in checks): raise AssertionError('Full final replay no longer agrees with V1')
                        temporary=path.with_suffix('.tmp'); write_json(temporary,{'config_hash':v1.config_hash(config),'checkpoint_hashes':hashes,
                            'trials':rows,'rollouts':rollouts,'baseline_checks':checks,'adaptation_logs':logs}); temporary.replace(path)
                        completed+=1
                        if completed%100==0:
                            write_json(run/'logs'/f'evaluate_seed{seed}_status.json',{'complete':False,'seed':seed,'completed_units':completed})
                            print(f'Evaluation seed={seed} completed_units={completed}',flush=True)
    write_json(run/'logs'/f'evaluate_seed{seed}_status.json',{'complete':True,'seed':seed,'completed_units':completed})
    return seed


def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def export_results(run):
    paths=sorted((run/'logs/evaluation_units').rglob('*.json'))
    # Stream artifacts to keep memory bounded on the 16 GB laptop.
    specifications=[('per_trial_metrics.csv','trials'),('rollout_metrics.csv','rollouts'),('baseline_reproduction.csv','baseline_checks')]
    for filename,key in specifications:
        writer=None
        with (run/filename).open('w',newline='',encoding='utf-8') as stream:
            for path in paths:
                rows=json.loads(path.read_text())[key]
                if not rows: continue
                if writer is None: writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader()
                writer.writerows(rows)
    return len(paths)


def run_parallel(run,config,stage,function):
    completed=[]; stage_status(run,stage,complete=False)
    try:
        with ProcessPoolExecutor(max_workers=config['train']['workers']) as executor:
            futures={executor.submit(function,str(run),seed):seed for seed in config['seeds']}
            for future in as_completed(futures):
                completed.append(future.result()); stage_status(run,stage,complete=False,completed_seeds=sorted(completed))
        if stage=='evaluate':
            count=export_results(run); expected=len(list(v1.evaluation_units(config)))
            if count!=expected: raise RuntimeError(f'Incomplete final units: {count}/{expected}')
            verify_frozen(baseline_path(config),ROOT/config['baseline_manifest'])
            stage_status(run,stage,complete=True,completed_seeds=sorted(completed),completed_units=count,
                         expected_trial_rows=count*6,expected_rollout_rows=count*6*len(config['evaluation']['rollout_horizons']),immutable_v1_verified=True)
        else: stage_status(run,stage,complete=True,completed_seeds=sorted(completed))
    except BaseException as exc:
        stage_status(run,stage,complete=False,completed_seeds=sorted(completed),error=str(exc))
        if stage=='evaluate': export_results(run)
        raise


def plot(run):
    from ..evaluation.factorial_report import generate_factorial_report
    status=json.loads((run/'status.json').read_text())
    return generate_factorial_report(run,status)


def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    parser=argparse.ArgumentParser(description='CartPole V2 factorial experiment using immutable V1 pretrained bases')
    parser.add_argument('stage',choices=['prepare','develop','train','evaluate','plot','all']); parser.add_argument('--execute',action='store_true')
    parser.add_argument('--run-dir',type=Path); parser.add_argument('--config',type=Path); args=parser.parse_args(argv)
    if args.stage!='plot' and not args.execute: parser.error('Execution requires --execute')
    if args.run_dir:
        run=args.run_dir.resolve()
        if (run/'FROZEN_RESULT.md').exists(): parser.error('Immutable V1 directory cannot receive V2 outputs')
        config=json.loads((run/'config.json').read_text()); validate_config(config)
        v1.verify_source_snapshot(run)
    else:
        if args.stage not in ('prepare','all'): parser.error('This stage requires --run-dir from prepare')
        config=load_config(args.config); run=prepare(config)
    torch.set_num_threads(config['train']['cpu_threads'])
    print(f'RUN_DIRECTORY={run}',flush=True)
    try:
        if args.stage in ('develop','all'): development(run,config)
        if args.stage in ('train','evaluate','all'):
            if not (run/'config_locked.json').exists(): raise RuntimeError('Development configuration must be locked before training/final test')
            locked=load_locked_config(run)
            if args.stage in ('train','all'): run_parallel(run,locked,'train',train_seed)
            if args.stage in ('evaluate','all'): run_parallel(run,locked,'evaluate',evaluate_seed); plot(run)
        if args.stage=='plot': plot(run)
    except (Exception,KeyboardInterrupt) as exc:
        (run/'logs/failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        print(f'INCOMPLETE: {type(exc).__name__}: {exc}; artifacts={run}',flush=True)
        if (run/'config_locked.json').exists(): export_results(run); plot(run)
        return 1
    return 0


if __name__=='__main__': raise SystemExit(main())

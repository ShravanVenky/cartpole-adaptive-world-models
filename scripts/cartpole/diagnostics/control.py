"""Matched closed-loop ablations with unchanged episode scoring."""
import time
import numpy as np
import torch
from adaptive_world_model.control.evaluation_mpc import GraphedPlanner, physics_step
from adaptive_world_model.environments.cartpole import CartPolePhysics, wrap_angle
from adaptive_world_model.utils.randomness import stable_seed
from .engine import Predictor, Planner, rollout, episode_metrics
from .context import save_json, save_npz, transition_batch


def run_control(ctx, phase, seed, condition, trials, settings, steps=None, cache_name=None):
    cfg, device = ctx.cfg, ctx.device
    key = cache_name or f'{phase}_{seed}_{condition["name"]}_{settings[0]["family"]}_{settings[0]["search"]}'
    target = ctx.run/'control'/f'{key}.json'
    cached = ctx.cached(target)
    if cached is not None:
        return cached
    started = time.perf_counter()
    labels = [dict(s, trial=trial) for s in settings for trial in trials]
    supports = [None if s['K'] == 0 or condition['online'] else ctx.support(seed, condition, s['trial'], s['support'], s['K']) for s in labels]
    base, models, fitlogs = ctx.deploy(seed, labels, supports)
    pack = Predictor(base, models, labels, device)
    physical, nominal = ctx.physics(condition['environment']), CartPolePhysics()
    pack.physics = physical
    state_list = []
    for s in labels:
        g = torch.Generator().manual_seed(stable_seed(seed, 'diagnostic_initial', condition['task'], s['trial']))
        state = torch.randn(4, generator=g)*torch.tensor([.1, .1, .05, .1])
        if condition['task'] == 'swingup':
            state[2] += np.pi
        state_list.append(state)
    state = torch.stack(state_list).to(device)
    planner = Planner(condition['horizon'], condition['task'], labels, device, cfg)
    groups = torch.tensor([trials.index(s['trial']) for s in labels], device=device)
    graph = GraphedPlanner(planner, pack, state, groups) if device == 'cuda' else None
    nominal_graph = None
    if condition['online'] and device == 'cuda':
        pack.physics = nominal
        nominal_graph = GraphedPlanner(planner, pack, state, groups)
        pack.physics = physical
    planner.mean.zero_()
    graph_seconds = time.perf_counter()-started
    count = steps or cfg[condition['task']+'_steps']
    shift = cfg['shift_'+condition['task']] if condition['online'] else 0
    ss, aa, pp, cc, latencies, updates = [state.detach().clone()], [], [], [], [], []
    plan_states, plan_actions, plan_steps, plan_true, plan_pred, plan_clearance = [], [], [], [], [], []
    dead = torch.zeros(len(labels), dtype=torch.bool, device=device)
    snapshots = []
    def snapshot(step):
        snapshots.append((step, pack.coef.detach().cpu().numpy().copy(),
                          {k:v.detach().cpu().numpy().copy() for k,v in pack.weights.items()}))
    snapshot(0)
    if device == 'cuda':
        torch.cuda.synchronize()
    loop_start = time.perf_counter()
    for t in range(count):
        p = nominal if condition['online'] and t < shift else physical
        pack.physics = p
        begin = time.perf_counter()
        seed_plan = stable_seed(seed, 'diagnostic_cem', condition['name'], t)
        with torch.no_grad():
            if graph is not None:
                action = (nominal_graph if condition['online'] and t < shift else graph).action(state, seed_plan)
            else:
                # Identical noise across methods and trials, even when candidate counts differ.
                g = torch.Generator(device=device).manual_seed(seed_plan)
                noise = torch.randn(planner.iterations, len(trials), planner.n, (planner.h+4)//5, 1, generator=g, device=device)
                action = planner.action(pack, state, seed_plan, noise.index_select(1, groups))
            pack.store_frozen = True
            pred = pack(state, action)
            pack.store_frozen = False
            correction = pred-pack.last_frozen[:,0]
            following = physics_step(state, action, p)
            bad = ~torch.isfinite(following).all(-1) | (following[:, 0].abs() > 2.4) | (following[:, 1:].abs() > 100).any(-1)
            following = torch.where(dead[:, None], state, following)
            dead |= bad
        if device == 'cuda':
            torch.cuda.synchronize()
        latencies.append(time.perf_counter()-begin)
        # A small fixed logging cadence, shared by all ablations. Evaluate selected mean plans in shadow mode.
        if t % 50 == 0 or (condition['online'] and t in [shift, shift+5, shift+10, shift+20, shift+50]):
            with torch.no_grad():
                plans = planner.selected_plan[:, None].clone()
                predicted_path, predicted_cost = rollout(pack, state, plans, planner)
                future_step = [t]
                def actual_next(x,u):
                    actual_p = nominal if condition['online'] and future_step[0] < shift else physical
                    future_step[0] += 1
                    return physics_step(x,u,actual_p)
                actual_path, actual_cost = rollout(actual_next, state, plans, planner)
            plan_states.append(predicted_path[:, 0].cpu().numpy())
            plan_true.append(actual_path[:, 0].cpu().numpy())
            plan_actions.append(plans[:, 0].cpu().numpy())
            plan_steps.append(t)
            plan_pred.append(np.stack([predicted_cost[:, 0].cpu().numpy(), actual_cost[:, 0].cpu().numpy()], -1))
            plan_clearance.append(np.stack([2.4-predicted_path[:, 0, :, 0].abs().amax(-1).cpu().numpy(),
                                            2.4-actual_path[:, 0, :, 0].abs().amax(-1).cpu().numpy()], -1))
        aa.append(action.detach()); pp.append(pred.detach())
        cc.append(correction.detach()); ss.append(following.detach())
        state = following
        observed = t-shift+1
        if condition['online'] and observed in cfg['online_updates']:
            for i, s in enumerate(labels):
                if s['method'] in ['frozen', 'oracle'] or dead[i].item():
                    continue
                data = transition_batch(torch.stack(ss[shift:t+1])[:, i], torch.stack(aa[shift:t+1])[:,i],
                                        torch.stack(ss[shift+1:t+2])[:, i], f'own_past/{key}/{i}')
                models[i].reset_adaptation()
                log = models[i].adapt(data)
                updates.append(dict(row=i, step=t+1, observations=observed, **log))
            pack.refresh(models)
            snapshot(t+1)
    if device == 'cuda':
        torch.cuda.synchronize()
    loop_seconds = time.perf_counter()-loop_start
    states = torch.stack(ss).permute(1, 0, 2).cpu().numpy()
    actions, predictions, corrections = [torch.stack(v,1).cpu().numpy() for v in [aa, pp, cc]]
    rows = []
    for i, s in enumerate(labels):
        metrics = episode_metrics(states[i], actions[i], cfg, condition['task'], shift)
        end = metrics['survival_steps']
        err = predictions[i, :end]-states[i, 1:end+1]
        err[:, 2] = np.arctan2(np.sin(err[:, 2]), np.cos(err[:, 2]))
        rows.append(dict(s, phase=phase, seed=seed, condition=condition['name'], environment=condition['environment'],
                         task=condition['task'], horizon=condition['horizon'], online=condition['online'], **metrics,
                         one_step_nmse=float(np.square(err/base.normalizer.y_scale.numpy()).mean()),
                         correction_rms=float(np.sqrt(np.nanmean(np.square(corrections[i, :end])))),
                         support_hash=supports[i].support_hash() if supports[i] is not None else 'own_past' if condition['online'] else 'none',
                         initial_adaptation_seconds=fitlogs[i]['adaptation_seconds'] if fitlogs[i] else 0.,
                         mpc_batch_seconds=float(np.mean(latencies)), postchange_interactions=max(0,end-shift) if condition['online'] else 0))
    extra = {}
    for j, (_, coef, weights) in enumerate(snapshots):
        extra[f'coef_{j}'] = coef
        for k, v in weights.items():
            extra[f'weights_{j}_{k}'] = v
    save_npz(target.with_suffix('.npz'), states=states, actions=actions, predictions=predictions, corrections=corrections,
             plan_steps=np.array(plan_steps), plan_states=np.stack(plan_states, 1), plan_true=np.stack(plan_true, 1),
             plan_actions=np.stack(plan_actions, 1), plan_costs=np.stack(plan_pred, 1), plan_clearances=np.stack(plan_clearance, 1),
             snapshot_steps=np.array([s[0] for s in snapshots]), iteration_best=planner.iteration_best.cpu().numpy(), **extra)
    packet = dict(key=key, config_sha256=ctx.digest, condition=condition, rows=rows, updates=updates,
                  initial_fit_logs=fitlogs,
                  graph_and_setup_seconds=graph_seconds, loop_seconds=loop_seconds, wall_seconds=time.perf_counter()-started,
                  steps=count, batch=len(labels), complete=True)
    save_json(target, packet)
    print(f'CONTROL {key} rows={len(rows)} seconds={packet["wall_seconds"]:.1f}', flush=True)
    return packet

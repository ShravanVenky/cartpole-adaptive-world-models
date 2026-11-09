"""Instrumented shared CEM; original source and trained parameters stay untouched."""
import math
import numpy as np
import torch
from adaptive_world_model.control.batched_predictor import BatchedPredictor
from adaptive_world_model.control.evaluation_mpc import EvaluationCEM, physics_step, stage_cost
from adaptive_world_model.environments.cartpole import CartPolePhysics, wrap_angle


class Predictor(BatchedPredictor):
    def __init__(self, base, models, settings, device):
        super().__init__(base, models, device)
        self.alpha = torch.tensor([s['alpha'] for s in settings], device=device)
        self.oracle = torch.tensor([s['method'] == 'oracle' for s in settings], device=device)
        self.oracle_indices = torch.tensor([i for i,s in enumerate(settings) if s['method']=='oracle'],device=device,dtype=torch.long)
        self.has_oracle = any(s['method']=='oracle' for s in settings)
        self.store_frozen = False
        self.physics = CartPolePhysics()

    def __call__(self, s, u):
        squeeze = s.ndim == 2
        if squeeze:
            s, u = s[:, None], u[:, None]
        x, v, t, w = s.unbind(-1)
        norm = self.base.normalizer
        z = norm.encode_x(torch.stack([x, v, t.sin(), t.cos(), w, u.squeeze(-1)], -1))
        frozen = self.base.network(z)
        if self.store_frozen:
            self.last_frozen = s + norm.decode_y(frozen)
        linear = torch.bmm(torch.cat([z, torch.ones_like(z[..., :1])], -1), self.coef)
        residual = z
        for layer in range(3):
            residual = torch.bmm(residual, self.weights[f'network.{layer*2}.weight'].transpose(1, 2)) + self.weights[f'network.{layer*2}.bias'][:, None]
            if layer < 2:
                residual = residual.tanh()
        correction = torch.where(self.types[:, None, None] == 1, linear,
                                 torch.where(self.types[:, None, None] == 2, residual, 0)).clamp(-8, 8)
        delta = norm.decode_y(frozen + self.alpha[:, None, None] * correction)
        bad = (delta.abs() > 50).any(-1) & (self.types[:, None] != 0) & (self.alpha[:, None] > 0)
        result = (s + delta).masked_fill(bad[..., None], float('nan'))
        if self.has_oracle:
            exact = physics_step(s.index_select(0,self.oracle_indices),u.index_select(0,self.oracle_indices),self.physics)
            result = result.index_copy(0,self.oracle_indices,exact)
        return result[:, 0] if squeeze else result

    def frozen(self, s, u):
        x, v, t, w = s.unbind(-1)
        z = torch.stack([x, v, t.sin(), t.cos(), w, u.squeeze(-1)], -1)
        return s + self.base.normalizer.decode_y(self.base.network(self.base.normalizer.encode_x(z)))


class Planner(EvaluationCEM):
    def __init__(self, horizon, task, settings, device, cfg):
        search = {s['search'] for s in settings}
        if len(search) != 1:
            raise ValueError('Batch must share CEM budget')
        n, iterations = cfg['budgets'][next(iter(search))]
        super().__init__(horizon, task, len(settings), n, iterations, n // 8, device)
        self.rail = torch.tensor([2.1 if s['safety'] == 'margin' else 2.4 for s in settings], device=device)[:, None]
        self.penalty = torch.tensor([1e6 if s['safety'] == 'stronger' else 1e5 for s in settings], device=device)[:, None]
        self.velocity = torch.tensor([cfg['near_upright_speed_penalty'] if s['cost'] == 'velocity' else 0 for s in settings], device=device)[:, None]
        self.terminal = torch.tensor([cfg['terminal_stabilization_extra'] if s['cost'] == 'terminal' else 0 for s in settings], device=device)[:, None]
        self.has_velocity = any(s['cost']=='velocity' for s in settings)
        self.has_terminal = any(s['cost']=='terminal' for s in settings)
        self.selected_plan = self.mean.clone()
        self.iteration_means = torch.empty(iterations, len(settings), horizon, 1, device=device)
        self.iteration_best = torch.empty(iterations, len(settings), device=device)
        self.last_candidates = None
        self.record_candidates = False

    def objective(self, s, u, terminal=False):
        cost = stage_cost(s, u, self.task)
        if not self.has_velocity and not self.has_terminal:
            return 5*cost if terminal else cost+self.penalty*torch.relu(s[...,0].abs()-self.rail).square()
        a = wrap_angle(s[..., 2])
        # Excess speed only near upright; ordinary crossings below4rad/s unpenalized.
        capture = torch.exp(-a.square() / .25)
        extra = self.velocity * capture * torch.relu(s[..., 3].abs() - 4.).square()
        if terminal:
            return 5 * (cost + extra) + self.terminal * capture * (a.square() + .25*s[..., 3].square())
        return cost + extra + self.penalty * torch.relu(s[..., 0].abs() - self.rail).square()

    @torch.no_grad()
    def action(self, predict, state, seed, noise=None):
        b, n, h = self.batch, self.n, self.h
        gen = torch.Generator(device=self.device).manual_seed(seed) if noise is None else None
        mean, std = self.mean, torch.full_like(self.mean, 6.)
        pools = []
        for iteration in range(self.iterations):
            z = torch.randn(b, n, math.ceil(h/self.block), 1, generator=gen, device=self.device) if noise is None else noise[iteration]
            z = z.repeat_interleave(self.block, dim=2)[:, :, :h]
            actions = (mean[:, None] + std[:, None]*z).clamp(-10, 10)
            actions[:, 0] = mean
            current = state[:, None].expand(b, n, 4)
            cost = torch.zeros(b, n, device=self.device)
            for t in range(h):
                current = predict(current, actions[:, :, t])
                cost += self.objective(current, actions[:, :, t])
            cost += self.objective(current, torch.zeros_like(actions[:, :, 0]), terminal=True)
            cost = torch.nan_to_num(cost, nan=1e30, posinf=1e30, neginf=1e30)
            ix = cost.topk(self.elites, dim=1, largest=False).indices
            elite = actions.gather(1, ix[:, :, None, None].expand(b, self.elites, h, 1))
            mean, std = elite.mean(1), elite.std(1, unbiased=False).clamp_min(.3)
            self.iteration_means[iteration].copy_(mean)
            self.iteration_best[iteration].copy_(cost.min(1).values)
            if self.record_candidates:
                pools.append(actions.detach().clone())
        if self.record_candidates:
            self.last_candidates = torch.stack(pools)
        self.selected_plan.copy_(mean)
        self.mean.copy_(torch.cat([mean[:, 1:], mean[:, -1:]], 1))
        return mean[:, 0]


@torch.no_grad()
def rollout(predict, initial, actions, planner):
    """Shared candidates [B,N,H,1]; raw forecasts retained past predicted rail crossings."""
    s = initial[:, None].expand(-1, actions.shape[1], -1)
    path, cost = [], torch.zeros(s.shape[:2], device=s.device)
    for t in range(actions.shape[2]):
        s = predict(s, actions[:, :, t])
        path.append(s)
        cost += planner.objective(s, actions[:, :, t])
    cost += planner.objective(s, torch.zeros_like(actions[:, :, 0]), terminal=True)
    return torch.stack(path, 2), torch.nan_to_num(cost, nan=1e30, posinf=1e30, neginf=1e30)


def episode_metrics(states, actions, cfg, task, event=0):
    """Original success/termination rules; cause-specific labels added without redefining success."""
    finite = np.isfinite(states).all(1)
    rail = np.abs(states[:, 0]) > cfg['rail_boundary']
    bound = (np.abs(states[:, 1:]) > cfg['numerical_state_bound']).any(1)
    bad = rail | ~finite | bound
    hit = np.flatnonzero(bad[1:])
    end = int(hit[0]+1) if len(hit) else len(actions)
    failed = bool(len(hit))
    s = states[:end+1]
    a = np.arctan2(np.sin(s[:, 2]), np.cos(s[:, 2]))
    good = (np.abs(a) < cfg['success_angle']) & (np.abs(s[:, 3]) < cfg['success_omega']) & (np.abs(s[:, 0]) <= cfg['rail_boundary'])
    streak = cfg['recovery_window']
    windows = np.convolve(good.astype(int), np.ones(streak, dtype=int), 'valid') == streak if len(good) >= streak else np.zeros(0, bool)
    reached = np.flatnonzero(windows)
    recovered = np.flatnonzero(windows[event:]) if event < len(windows) else np.array([])
    success = not failed and len(good) >= cfg['sustained_steps'] and bool(good[-cfg['sustained_steps']:].all())
    if task == 'balance' and event == 0:
        success &= not bool((np.abs(a) > .5).any())
    crossing = np.flatnonzero((a[:-1]*a[1:] <= 0) & (np.abs(a[:-1]) < .8) & (np.abs(a[1:]) < .8)) + 1
    near = np.flatnonzero(np.abs(a) < .15)
    cross = int(crossing[0]) if len(crossing) else int(near[0]) if len(near) else None
    stage = stage_cost(torch.from_numpy(s[1:]), torch.from_numpy(actions[:end]), task).numpy()
    stage_sum = float(np.nan_to_num(stage, nan=1e20, posinf=1e20).sum())
    numerical = failed and (not finite[end] or bound[end])
    rail_failure = failed and bool(rail[end]) and not numerical
    lost = bool(len(reached) and not success)
    # These are observed phenotypes. Causal labels require paired interventions/replay.
    reason = 'numerical_failure' if numerical else 'rail_crossing' if rail_failure else 'success' if success else 'failure_to_maintain_stabilization' if lost else 'high_velocity_upright_overshoot' if cross is not None and abs(s[cross, 3]) > 4 else 'never_reached_upright' if cross is None else 'time_limit_without_stabilization'
    return dict(success=bool(success), failed=failed, rail_failure=rail_failure, numerical_failure=bool(numerical),
                numerical_bound_failure=bool(failed and bound[end]), time_limit_failure=bool(not failed and not success),
                survival_steps=end, cost=stage_sum + cfg['failure_penalty']*failed, cost_without_failure_penalty=stage_sum,
                first_upright_crossing=cross, crossing_omega=float(abs(s[cross, 3])) if cross is not None else None,
                max_overshoot=float(np.max(np.abs(a[cross:]))) if cross is not None else None,
                stabilization_step=int(reached[0]+streak) if len(reached) else None,
                recovery_steps=int(recovered[0]+streak) if len(recovered) else None,
                settled_then_lost=lost, failure_category=reason,
                closest_upright_angle=float(np.min(np.abs(a))),peak_angular_velocity=float(np.max(np.abs(s[:,3]))),
                high_velocity_crossing=bool(cross is not None and abs(s[cross,3])>cfg['excessive_crossing_speed']),
                possible_insufficient_momentum=bool(cross is None and np.min(np.abs(a))>.5 and np.max(np.abs(s[:,3]))<4),
                possible_failure_to_brake=bool(cross is not None and abs(s[cross,3])>4 and not success),
                min_actual_rail_clearance=float(cfg['rail_boundary']-np.max(np.abs(s[:, 0]))),
                action_energy=float(np.square(actions[:end]).sum()))

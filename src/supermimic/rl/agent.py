"""Policy improvement by BPTT through the frozen world model (paper §3.3).

One update imagines H=32 steps from replay-buffer states, optimises the
survival-weighted return of Eq. 2 with analytic gradients, then fits the critic
to the fixed GAE targets of Eq. 3.  The dynamics weights omega and the critic
weights psi are held fixed during the actor's backward pass.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from supermimic.rl.networks import Critic, MotionEncoder, PolicyDecoder, Scales
from supermimic.rl.world_model import WorldModel


@dataclass(kw_only=True)
class AgentCfg:
  horizon: int = 32
  gamma: float = 0.99
  lam: float = 0.95
  lr: float = 2e-4
  betas: tuple[float, float] = (0.9, 0.95)
  grad_clip: float = 1.0
  element_clip: float = 0.1
  critic_epochs: int = 5
  critic_minibatches: int = 4
  critic_grad_clip: float = 1.0
  rollouts: int = 4096
  sensor_noise: bool = True
  recompute: bool = True


@contextmanager
def frozen(*modules: torch.nn.Module):
  params = [p for m in modules for p in m.parameters()]
  flags = [p.requires_grad for p in params]
  for p in params:
    p.requires_grad_(False)
  try:
    yield
  finally:
    for p, f in zip(params, flags, strict=True):
      p.requires_grad_(f)


class SuperMimicAgent:
  def __init__(
    self,
    wm: WorldModel,
    encoder: MotionEncoder,
    decoder: PolicyDecoder,
    critic: Critic,
    scales: Scales,
    history: int,
    cfg: AgentCfg,
  ) -> None:
    self.wm = wm
    self.encoder = encoder
    self.decoder = decoder
    self.critic = critic
    self.scales = scales
    self.history = history
    self.cfg = cfg
    self.device = scales.state.device
    self.theta = list(encoder.parameters()) + list(decoder.parameters())
    self.opt_theta = torch.optim.Adam(self.theta, lr=cfg.lr, betas=cfg.betas)
    self.opt_psi = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    self.metrics: dict[str, float] = {}

  def set_lr(self, lr: float) -> None:
    for opt in (self.opt_theta, self.opt_psi):
      for g in opt.param_groups:
        g["lr"] = lr

  def act(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Deterministic policy for simulator collection and evaluation."""
    z = self.encoder(goal / self.scales.goal)
    return self.decoder(z, obs / self.scales.obs)

  # --- imagined rollout -------------------------------------------------

  def _imagine_step(
    self,
    s: torch.Tensor,
    o_hist: torch.Tensor,
    prev_a: torch.Tensor,
    prev2_a: torch.Tensor,
    xi: torch.Tensor,
    clip: torch.Tensor,
    frame: torch.Tensor,
    last: bool,
  ):
    wm = self.wm
    robot, ref = wm.robot_and_ref(s, clip, frame)
    obs = wm.obs_from_state(s, xi, prev_a)
    obs_n = wm.corrupt(obs) if self.cfg.sensor_noise else obs
    hist = torch.cat([o_hist[:, 1:], obs_n.unsqueeze(1)], dim=1)
    goal = wm.goal(robot, s, clip, frame)
    # Reward of the transition *into* s: R(s_t, ref_t, a_{t-2}, a_{t-1}).
    r = wm.reward(robot, ref, prev2_a, prev_a)
    if last:
      return s, obs_n, prev_a, r, goal, frame
    a = self.decoder(
      self.encoder(goal / self.scales.goal), hist.flatten(1) / self.scales.obs
    )
    adv = wm.library.advance_at(clip, frame)
    s2, logit = wm.step(s, a, xi, adv)
    d = torch.sigmoid(logit)
    # A clip end is a truncation, not a failure; either way the rollout stops.
    d = torch.where(frame >= wm.library.lengths[clip] - 1, torch.ones_like(d), d)
    return s2, obs_n, a, r, goal, frame + 1, d

  def imagine(
    self,
    s0: torch.Tensor,
    o_hist0: torch.Tensor,
    xi: torch.Tensor,
    clip: torch.Tensor,
    frame: torch.Tensor,
    prev_a: torch.Tensor,
  ):
    H = self.cfg.horizon
    B = s0.shape[0]
    zero = torch.zeros(B, device=self.device)
    ones = torch.ones(B, device=self.device)
    s, c, d, f = s0, ones, zero, frame
    pa, p2a = prev_a, torch.zeros_like(prev_a)
    hist = o_hist0
    s_buf, g_buf, r_buf, c_buf, d_buf = [], [], [], [], []
    discounts = torch.tensor(
      [self.cfg.gamma**t for t in range(H + 1)], device=self.device
    )
    for t in range(H + 1):
      s_buf.append(s.detach())
      c_buf.append(c)
      d_buf.append(d)
      last = t == H
      args = (s, hist, pa, p2a, xi, clip, f, last)
      out = (
        checkpoint(self._imagine_step, *args, use_reentrant=False)
        if self.cfg.recompute and not last
        else self._imagine_step(*args)
      )
      r, goal = out[3], out[4]
      r_buf.append(r)
      g_buf.append(goal.detach())
      if last:
        sH, gH = s, goal
        break
      s, obs_n, a, _, _, f, d_next = out
      c = c * (1.0 - d_next)
      d = d_next
      p2a, pa = pa, a
      hist = torch.cat([hist[:, 1:], obs_n.unsqueeze(1)], dim=1)

    with frozen(self.critic):
      vH = self.critic(
        sH / self.scales.state, gH / self.scales.goal, xi / self.scales.xi
      )
    return {
      "s": torch.stack(s_buf),
      "g": torch.stack(g_buf),
      "r": torch.stack(r_buf),
      "c": torch.stack(c_buf),
      "d": torch.stack(d_buf),
      "vH": vH,
      "discounts": discounts,
      "xi": xi,
    }

  # --- Eq. 2 and Eq. 3 --------------------------------------------------

  def update(self, replay, xi_all: torch.Tensor) -> dict[str, float]:
    cfg = self.cfg
    H = cfg.horizon
    t, e = replay.sample_states(cfg.rollouts)
    xi = xi_all[e]
    s0 = replay.s[t, e]
    clip = replay.clip[t, e].long()
    frame = replay.frame[t, e].long()
    prev_a = replay.prev_action(t, e)

    def obs_fn(s, a_prev, env):
      return self.wm.obs_from_state(s, xi_all[env], a_prev)

    hist = replay.obs_history(t, e, self.history, obs_fn)
    if cfg.sensor_noise:
      hist = self.wm.corrupt(hist)

    self.encoder.train()
    self.decoder.train()
    with frozen(self.wm.dynamics, self.critic):
      roll = self.imagine(s0, hist, xi, clip, frame, prev_a)
      gamma_t = roll["discounts"][:H, None]
      survival = roll["c"][:H]
      rewards = roll["r"][1 : H + 1]
      objective = (gamma_t * survival * rewards).sum(0)
      objective = objective + roll["discounts"][H] * roll["c"][H] * roll["vH"]
      actor_loss = -objective.mean()
      self.opt_theta.zero_grad(set_to_none=True)
      actor_loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(self.theta, cfg.grad_clip)
    for p in self.theta:
      if p.grad is not None:
        p.grad.clamp_(-cfg.element_clip, cfg.element_clip)
    self.opt_theta.step()

    critic_stats = self.fit_critic(roll)
    self.metrics = {
      "agent/objective": float(objective.detach().mean()),
      "agent/actor_loss": float(actor_loss.detach()),
      "agent/grad_norm": float(gn),
      "agent/survival_H": float(roll["c"][H].detach().mean()),
      "agent/reward_mean": float(rewards.detach().mean()),
      **critic_stats,
    }
    return self.metrics

  def fit_critic(self, roll: dict[str, torch.Tensor]) -> dict[str, float]:
    cfg = self.cfg
    H = cfg.horizon
    s, g, xi = roll["s"], roll["g"], roll["xi"]
    r, d = roll["r"], roll["d"]
    with torch.no_grad():
      v = self.critic(
        (s / self.scales.state).flatten(0, 1),
        (g / self.scales.goal).flatten(0, 1),
        (xi / self.scales.xi).unsqueeze(0).expand(H + 1, -1, -1).flatten(0, 1),
      ).reshape(H + 1, -1)
      returns = [torch.empty_like(v[0]) for _ in range(H + 1)]
      returns[H] = v[H]
      for t in range(H - 1, -1, -1):
        bootstrap = (1.0 - cfg.lam) * v[t + 1] + cfg.lam * returns[t + 1]
        returns[t] = r[t + 1] + cfg.gamma * (1.0 - d[t + 1]) * bootstrap
      target = torch.stack(returns[:H]).flatten()

    cs = (s[:H] / self.scales.state).flatten(0, 1)
    cg = (g[:H] / self.scales.goal).flatten(0, 1)
    cx = (xi / self.scales.xi).unsqueeze(0).expand(H, -1, -1).flatten(0, 1)
    n = target.numel()
    mb = n // cfg.critic_minibatches
    total = 0.0
    steps = 0
    self.critic.train()
    for _ in range(cfg.critic_epochs):
      perm = torch.randperm(n, device=self.device)
      for j in range(cfg.critic_minibatches):
        idx = perm[j * mb : (j + 1) * mb]
        pred = self.critic(cs[idx], cg[idx], cx[idx])
        loss = (pred - target[idx]).square().mean() * H
        self.opt_psi.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
        self.opt_psi.step()
        total += float(loss.detach())
        steps += 1
    return {
      "critic/loss": total / max(steps, 1),
      "critic/target_mean": float(target.mean()),
      "critic/v_mean": float(v[:H].mean()),
    }

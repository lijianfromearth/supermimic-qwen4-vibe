"""Neural network architectures for SuperMimic (Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.state import (
  CANONICAL_STATE_DIM,
  OBS_FRAME_DIM,
  RobotState,
  SIM_CONTEXT_DIM,
  apply_canonical_delta,
  compute_termination_errors,
  state_to_canonical,
)


class RunningNormalizer(nn.Module):
  """Online running mean and std normalizer (Welford's algorithm)."""

  def __init__(self, shape: tuple[int, ...] | int, eps: float = 1e-5, clip: float = 10.0) -> None:
    super().__init__()
    if isinstance(shape, int):
      shape = (shape,)
    self.shape = shape
    self.eps = eps
    self.clip = clip
    self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
    self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
    self.register_buffer("std", torch.ones(shape, dtype=torch.float32))
    self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    flat = x.detach().reshape(-1, *self.shape)
    batch_count = flat.shape[0]
    if batch_count == 0:
      return
    batch_mean = flat.mean(dim=0)
    batch_var = flat.var(dim=0, unbiased=False)
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count

    new_mean = self.mean + delta * (batch_count / tot_count).float()
    m_a = self.var * self.count.float()
    m_b = batch_var * batch_count
    m2 = m_a + m_b + torch.square(delta) * (self.count * batch_count / tot_count).float()
    new_var = m2 / tot_count.float()

    self.mean.copy_(new_mean)
    self.var.copy_(new_var)
    self.std.copy_(torch.sqrt(new_var.clamp(min=self.eps * self.eps)))
    self.count.copy_(tot_count)

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    normed = (x - self.mean) / self.std
    return torch.clamp(normed, -self.clip, self.clip)

  def denormalize(self, y: torch.Tensor) -> torch.Tensor:
    return y * self.std + self.mean


def build_mlp(
  in_dim: int,
  hidden_dims: tuple[int, ...],
  out_dim: int,
  use_layer_norm: bool = True,
  out_init_scale: float | None = None,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  prev_dim = in_dim
  for h_dim in hidden_dims:
    linear = nn.Linear(prev_dim, h_dim)
    nn.init.orthogonal_(linear.weight, gain=1.0)
    nn.init.zeros_(linear.bias)
    layers.append(linear)
    if use_layer_norm:
      layers.append(nn.LayerNorm(h_dim))
    layers.append(nn.ELU())
    prev_dim = h_dim

  out_layer = nn.Linear(prev_dim, out_dim)
  if out_init_scale is not None:
    nn.init.orthogonal_(out_layer.weight, gain=out_init_scale)
  else:
    nn.init.orthogonal_(out_layer.weight, gain=1.0)
  nn.init.zeros_(out_layer.bias)
  layers.append(out_layer)
  return nn.Sequential(*layers)


class WorldModel(nn.Module):
  """Differentiable world model f_omega (MLP 2048, 2048, 2048) and termination head d_omega."""

  def __init__(
    self,
    state_dim: int = CANONICAL_STATE_DIM,
    action_dim: int = 29,
    context_dim: int = SIM_CONTEXT_DIM,
    out_delta_dim: int = 268,
  ) -> None:
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.context_dim = context_dim
    self.in_dim = state_dim + action_dim + context_dim
    self.out_delta_dim = out_delta_dim

    self.in_norm = RunningNormalizer(self.in_dim)
    self.delta_norm = RunningNormalizer(out_delta_dim, clip=20.0)

    # Dynamics f_omega: MLP (2048, 2048, 2048) (Table A.4)
    self.dynamics_mlp = build_mlp(
      in_dim=self.in_dim,
      hidden_dims=(2048, 2048, 2048),
      out_dim=out_delta_dim,
      use_layer_norm=True,
      out_init_scale=0.01,
    )

    # Learned termination head d_omega(s_hat_t, g_t) trained with BCE (Eq. 1)
    # Combines smooth analytic margin (|e_{p,z}| > 0.25 or e_tilt > 0.8) with learned state residual
    self.term_mlp = build_mlp(
      in_dim=state_dim + 4,
      hidden_dims=(128, 64),
      out_dim=1,
      use_layer_norm=True,
      out_init_scale=0.01,
    )
    self.log_sharpness_z = nn.Parameter(torch.tensor(3.5))     # exp(3.5) ~ 33.1
    self.log_sharpness_tilt = nn.Parameter(torch.tensor(2.8))  # exp(2.8) ~ 16.4

  def predict_next_state(
    self,
    state: RobotState,
    action: torch.Tensor,
    sim_context: torch.Tensor,
    default_joint_pos: torch.Tensor,
    s_tilde: torch.Tensor | None = None,
    cos_yaw: torch.Tensor | None = None,
    sin_yaw: torch.Tensor | None = None,
  ) -> tuple[RobotState, torch.Tensor]:
    """Predict next RobotState s_hat_{t+1} = f_omega(s_hat_t, a_t, xi)."""
    if s_tilde is None or cos_yaw is None or sin_yaw is None:
      s_tilde, cos_yaw, sin_yaw = state_to_canonical(state, default_joint_pos)

    raw_in = torch.cat((s_tilde, action, sim_context), dim=-1)
    norm_in = self.in_norm.normalize(raw_in)
    norm_delta = 10.0 * torch.tanh(self.dynamics_mlp(norm_in) / 10.0)
    delta = self.delta_norm.denormalize(norm_delta)

    next_state = apply_canonical_delta(
      state, s_tilde, cos_yaw, sin_yaw, delta, default_joint_pos
    )
    return next_state, delta

  def predict_termination_logit(
    self,
    state: RobotState,
    ref_root_pos_w: torch.Tensor,
    ref_root_rot_w: torch.Tensor,
    default_joint_pos: torch.Tensor,
    s_tilde: torch.Tensor | None = None,
    safety_margin_z: float = 0.0,
    safety_margin_tilt: float = 0.0,
  ) -> torch.Tensor:
    """Predict termination logit for d_hat_t = sigmoid(logit) in (0, 1).
    Optional safety_margin_z / safety_margin_tilt account for H=32 step world-model
    rollout epistemic error (~2.5 cm) during policy BPTT without altering simulator termination.
    """
    if s_tilde is None:
      s_tilde, _, _ = state_to_canonical(state, default_joint_pos)

    err_z, err_tilt, _ = compute_termination_errors(state, ref_root_pos_w, ref_root_rot_w)
    margin_z = err_z - 0.25
    margin_tilt = err_tilt - 0.8
    eff_margin_z = margin_z + safety_margin_z
    eff_margin_tilt = margin_tilt + safety_margin_tilt

    kz = torch.exp(self.log_sharpness_z).clamp(min=5.0, max=15.0)
    kt = torch.exp(self.log_sharpness_tilt).clamp(min=5.0, max=15.0)
    # Smooth maximum of the two termination criteria
    analytic_logit = torch.logaddexp(kz * eff_margin_z, kt * eff_margin_tilt) - 0.693147

    s_norm = (s_tilde - self.in_norm.mean[: self.state_dim]) / self.in_norm.std[: self.state_dim]
    s_norm = torch.clamp(s_norm, -10.0, 10.0)
    term_feats = torch.stack(
      (margin_z, margin_tilt, err_z / 0.25, err_tilt / 0.8), dim=-1
    )
    res_logit = self.term_mlp(torch.cat((s_norm, term_feats), dim=-1)).squeeze(-1)
    return analytic_logit + res_logit


class MotionPolicy(nn.Module):
  """Goal-conditioned policy pi_theta = dec_theta(enc_theta,motion(g_t), o_{t-9:t}) (Table A.4)."""

  def __init__(
    self,
    goal_dim: int = 730,
    obs_hist_dim: int = 10 * OBS_FRAME_DIM,
    latent_dim: int = 32,
    action_dim: int = 29,
  ) -> None:
    super().__init__()
    self.goal_dim = goal_dim
    self.obs_hist_dim = obs_hist_dim
    self.latent_dim = latent_dim
    self.action_dim = action_dim

    self.goal_norm = RunningNormalizer(goal_dim)
    self.obs_norm = RunningNormalizer(OBS_FRAME_DIM)

    # Motion encoder enc_theta,motion: MLP (2048, 1024, 512, 256) -> tanh in R^32
    self.motion_encoder = build_mlp(
      in_dim=goal_dim,
      hidden_dims=(2048, 1024, 512, 256),
      out_dim=latent_dim,
      use_layer_norm=True,
      out_init_scale=0.5,
    )

    # Shared decoder dec_theta: MLP (2048, 1024, 512, 256) -> action in R^29
    self.decoder = build_mlp(
      in_dim=latent_dim + obs_hist_dim,
      hidden_dims=(2048, 1024, 512, 256),
      out_dim=action_dim,
      use_layer_norm=True,
      out_init_scale=0.01,
    )

  def encode_motion(self, goal_seq: torch.Tensor) -> torch.Tensor:
    g_norm = self.goal_norm.normalize(goal_seq)
    return torch.tanh(self.motion_encoder(g_norm))

  def forward(self, obs_hist_flat: torch.Tensor, goal_seq: torch.Tensor) -> torch.Tensor:
    B = obs_hist_flat.shape[0]
    obs_frames = obs_hist_flat.view(B, -1, OBS_FRAME_DIM)
    obs_norm_flat = self.obs_norm.normalize(obs_frames).view(B, self.obs_hist_dim)
    z = self.encode_motion(goal_seq)
    dec_in = torch.cat((z, obs_norm_flat), dim=-1)
    return self.decoder(dec_in)


class Critic(nn.Module):
  """Value critic V_psi(s_hat_t, g_t, xi) -> scalar (MLP 256, 128) (Table A.4)."""

  def __init__(
    self,
    state_dim: int = CANONICAL_STATE_DIM,
    goal_dim: int = 730,
    context_dim: int = SIM_CONTEXT_DIM,
  ) -> None:
    super().__init__()
    self.in_dim = state_dim + goal_dim + context_dim
    self.in_norm = RunningNormalizer(self.in_dim)
    self.net = build_mlp(
      in_dim=self.in_dim,
      hidden_dims=(256, 128),
      out_dim=1,
      use_layer_norm=True,
      out_init_scale=1.0,
    )

  def forward(
    self, s_tilde: torch.Tensor, goal_seq: torch.Tensor, sim_context: torch.Tensor
  ) -> torch.Tensor:
    raw_in = torch.cat((s_tilde, goal_seq, sim_context), dim=-1)
    norm_in = self.in_norm.normalize(raw_in)
    return self.net(norm_in).squeeze(-1)

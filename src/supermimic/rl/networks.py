"""Networks of the SuperMimic agent (paper Table A.4)."""

from __future__ import annotations

import torch
from torch import nn

from supermimic.state import (
  ACTION_SCALE,
  GOAL_BLOCK_SCALE,
  NUM_JOINTS,
  OBS_SCALE,
  REF_ADV_DIM,
  REF_ADV_SCALE,
  STATE_DIM,
  STATE_SCALE,
  XI_DIM,
  XI_SCALE,
)

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}


class MLP(nn.Module):
  def __init__(
    self,
    in_dim: int,
    hidden: tuple[int, ...],
    out_dim: int,
    activation: str = "elu",
    out_gain: float = 1.0,
  ) -> None:
    super().__init__()
    act = ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
      layers += [nn.Linear(prev, width), act()]
      prev = width
    head = nn.Linear(prev, out_dim)
    if out_gain != 1.0:
      nn.init.orthogonal_(head.weight, gain=out_gain)
      nn.init.zeros_(head.bias)
    layers.append(head)
    self.net = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class Scales(nn.Module):
  """Fixed input normalisation, kept as buffers so checkpoints are portable."""

  def __init__(self, num_future_frames: int, history: int) -> None:
    super().__init__()
    self.register_buffer("state", STATE_SCALE.clone())
    self.register_buffer("xi", XI_SCALE.clone())
    self.register_buffer("adv", REF_ADV_SCALE.clone())
    self.register_buffer("goal", GOAL_BLOCK_SCALE.repeat(num_future_frames))
    self.register_buffer("obs", OBS_SCALE.repeat(history))
    self.action = ACTION_SCALE

  def to_device(self, device: str) -> "Scales":
    return self.to(device)


class Dynamics(nn.Module):
  """f_omega: (s, a, xi, ref_adv) -> (state delta, termination logit).

  The termination logit shares the dynamics trunk (fix1 baseline).  A separate
  state-only head learned only the base rate and reported "alive" on drifted
  imagined states, so the survival weight stayed ~1 and the critic bootstrap
  diverged; the shared trunk is the stable configuration.
  """

  def __init__(self, hidden: tuple[int, ...] = (2048, 2048, 2048)) -> None:
    super().__init__()
    in_dim = STATE_DIM + NUM_JOINTS + XI_DIM + REF_ADV_DIM
    self.net = MLP(in_dim, hidden, STATE_DIM + 1, out_gain=0.1)

  def forward(
    self,
    s_norm: torch.Tensor,
    a: torch.Tensor,
    xi_norm: torch.Tensor,
    adv_norm: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    out = self.net(torch.cat([s_norm, a / ACTION_SCALE, xi_norm, adv_norm], dim=-1))
    return out[..., :STATE_DIM], out[..., STATE_DIM]


class TerminationHead(nn.Module):
  """d_omega: normalized state -> termination logit (paper Eq. 1, d_hat(s, g)).

  Kept separate from the dynamics trunk so its BCE gradient is not swamped by the
  ~3600x larger state-prediction loss; Adam then adapts to the BCE alone and the
  head learns to recognize fall states.  The canonical state already carries the
  reference-relative height and tilt errors that define termination, so the head
  needs only the state.  Its logit keeps the survival weight differentiable.
  """

  def __init__(self, hidden: tuple[int, ...] = (256, 128)) -> None:
    super().__init__()
    self.net = MLP(STATE_DIM, hidden, 1)

  def forward(self, s_norm: torch.Tensor) -> torch.Tensor:
    return self.net(s_norm).squeeze(-1)


class MotionEncoder(nn.Module):
  """Goal features -> 32-dim tanh latent."""

  def __init__(
    self,
    goal_dim: int,
    latent_dim: int = 32,
    hidden: tuple[int, ...] = (2048, 1024, 512, 256),
  ) -> None:
    super().__init__()
    self.net = MLP(goal_dim, hidden, latent_dim)

  def forward(self, g_norm: torch.Tensor) -> torch.Tensor:
    return torch.tanh(self.net(g_norm))


class PolicyDecoder(nn.Module):
  """(z, last `history` proprioception frames) -> action mean.

  Imagined rollouts act on the mean so BPTT differentiates the return with
  respect to each action (paper §3.3); the N(0, 0.1^2) exploration noise of
  Table A.5 is added only when collecting in the simulator.
  """

  def __init__(
    self,
    latent_dim: int,
    obs_dim: int,
    history: int,
    hidden: tuple[int, ...] = (2048, 1024, 512, 256),
  ) -> None:
    super().__init__()
    self.net = MLP(latent_dim + obs_dim * history, hidden, NUM_JOINTS, out_gain=0.01)

  def forward(self, z: torch.Tensor, o_hist_norm: torch.Tensor) -> torch.Tensor:
    return self.net(torch.cat([z, o_hist_norm], dim=-1))


class Critic(nn.Module):
  """V_psi(s, g, xi), one per interface (simulator and imagination share it)."""

  def __init__(self, goal_dim: int, hidden: tuple[int, ...] = (256, 128)) -> None:
    super().__init__()
    self.net = MLP(STATE_DIM + goal_dim + XI_DIM, hidden, 1)

  def forward(
    self, s_norm: torch.Tensor, g_norm: torch.Tensor, xi_norm: torch.Tensor
  ) -> torch.Tensor:
    return self.net(torch.cat([s_norm, g_norm, xi_norm], dim=-1)).squeeze(-1)

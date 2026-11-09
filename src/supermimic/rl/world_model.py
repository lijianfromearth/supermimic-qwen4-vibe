"""Learned dynamics f_omega and the analytic MDP heads (paper §3.2, Eq. 1).

Only the dynamics network carries parameters.  The observation and reward heads
are exact analytic functions of the canonical state through the differentiable
FK, and Eq. 1 assigns no loss to an analytic head, so the fit reduces to the
state MSE plus the BCE of the learned termination logit against the simulator's
hard label.  That logit is what makes the survival weight c_t differentiable
inside imagined rollouts.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from supermimic.kinematics import Kinematics
from supermimic.mdp.motion_library import MotionLibrary
from supermimic.mdp.tracking_math import RefFrame, RobotFrame, TrackingMath
from supermimic.quat import normalize, quat_apply_inverse
from supermimic.rl.networks import Dynamics, Scales, TerminationHead
from supermimic.state import (
  NUM_JOINTS,
  PHYS_DIM,
  goal_from_future,
  reconstruct_reference,
  robot_frame,
)

QUAT_BLOCKS = ((1, 5), (72, 76))


def renorm_quats(s: torch.Tensor) -> torch.Tensor:
  """Keep the two quaternion blocks on the unit sphere after a delta update."""
  parts = []
  prev = 0
  for lo, hi in QUAT_BLOCKS:
    parts.append(s[:, prev:lo])
    parts.append(normalize(s[:, lo:hi]))
    prev = hi
  parts.append(s[:, prev:])
  return torch.cat(parts, dim=-1)


class WorldModel:
  def __init__(
    self,
    dynamics: Dynamics,
    term: TerminationHead,
    scales: Scales,
    kin: Kinematics,
    math: TrackingMath,
    library: MotionLibrary,
    lr: float = 2e-4,
    grad_clip: float = 1.0,
    window: int = 8,
    chunks: int = 10,
    chunk_size: int = 2048,
  ) -> None:
    self.dynamics = dynamics
    self.term = term
    self.scales = scales
    self.kin = kin
    self.math = math
    self.library = library
    self.device = scales.state.device
    self.state_scale = scales.state
    self.xi_scale = scales.xi
    self.adv_scale = scales.adv
    self.window = window
    self.chunks = chunks
    self.chunk_size = chunk_size
    self.grad_clip = grad_clip
    self.params = list(dynamics.parameters()) + list(term.parameters())
    self.opt = torch.optim.Adam(self.params, lr=lr)
    self.metrics: dict[str, float] = {}

  def set_lr(self, lr: float) -> None:
    for g in self.opt.param_groups:
      g["lr"] = lr

  # --- model interface --------------------------------------------------

  def step(
    self, s: torch.Tensor, a: torch.Tensor, xi: torch.Tensor, adv: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """One deterministic transition; returns (next state, termination logit).

    The logit is read from the separate termination head on the predicted next
    state, matching the paper's d_hat_{t+1} = d_omega(s_{t+1}, g_{t+1}).
    """
    delta = self.dynamics(
      s / self.state_scale, a, xi / self.xi_scale, adv / self.adv_scale
    )
    s2 = renorm_quats(s + delta * self.state_scale)
    return s2, self.term(s2 / self.state_scale)

  def robot_and_ref(
    self, s: torch.Tensor, clip: torch.Tensor, frame: torch.Tensor
  ) -> tuple[RobotFrame, RefFrame]:
    robot = robot_frame(s, self.kin)
    ref = reconstruct_reference(self.library.canon_at(clip, frame), s)
    return robot, ref

  def obs_from_state(
    self, s: torch.Tensor, xi: torch.Tensor, prev_a: torch.Tensor
  ) -> torch.Tensor:
    """Same head without the FK: every proprioceptive field is a state slice."""
    q_rp = normalize(s[:, 1:5])
    gravity_b = quat_apply_inverse(q_rp, self.math.gravity_w)
    return torch.cat(
      [
        s[:, 11 : 11 + NUM_JOINTS] + xi[:, 1 : 1 + NUM_JOINTS] - self.math.q0,
        s[:, 11 + NUM_JOINTS : PHYS_DIM],
        s[:, 8:11],
        gravity_b,
        prev_a,
      ],
      dim=-1,
    )

  def corrupt(self, obs: torch.Tensor) -> torch.Tensor:
    lo, hi = self.math.obs_noise_lo, self.math.obs_noise_hi
    return obs + torch.rand_like(obs) * (hi - lo) + lo

  def goal(
    self, robot: RobotFrame, s: torch.Tensor, clip: torch.Tensor, frame: torch.Tensor
  ) -> torch.Tensor:
    joint_pos, joint_vel, anchor_z = self.library.future_gather(
      clip, frame, self.offsets
    )
    return goal_from_future(
      robot.anchor_pos_w,
      robot.anchor_quat_w,
      s[:, 69:72],
      s[:, 72:76],
      joint_pos,
      joint_vel,
      anchor_z,
    )

  @property
  def offsets(self) -> tuple[int, ...]:
    return tuple(range(self.math.K))

  def reward(
    self,
    robot: RobotFrame,
    ref: RefFrame,
    prev_a: torch.Tensor,
    a: torch.Tensor,
  ) -> torch.Tensor:
    return self.math.reward(robot, ref, prev_a, a)

  # --- Eq. 1 ------------------------------------------------------------

  def window_loss(
    self, batch: dict[str, torch.Tensor], xi: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """k-step autoregressive loss on logged actions, seeded at the logged s_0."""
    k = self.window
    clip, frame = batch["clip"], batch["frame"]
    s_hat = batch["s"][:, 0]
    target = batch["s"][:, 1 : k + 1]
    labels = batch["d"][:, :k].float()
    # Only the final transition may be terminal; its successor is a post-reset
    # state, so it carries a valid termination label but no valid state target.
    boundary = (batch["d"] | batch["trunc"])[:, k - 1]
    state_loss = s_hat.new_zeros(())
    term_loss = s_hat.new_zeros(())
    for j in range(k):
      adv = self.library.advance_at(clip[:, j], frame[:, j])
      s_hat, logit = self.step(s_hat, batch["a"][:, j], xi, adv)
      err = (s_hat - target[:, j]) / self.state_scale
      sq = err.square().sum(-1)
      if j == k - 1:
        sq = torch.where(boundary, torch.zeros_like(sq), sq)
      state_loss = state_loss + sq.mean()
      term_loss = term_loss + F.binary_cross_entropy_with_logits(logit, labels[:, j])
    return state_loss, term_loss

  def fit(self, replay, xi_all: torch.Tensor) -> dict[str, float]:
    """One WM update: `chunks` gradient-accumulated minibatches, one Adam step."""
    self.dynamics.train()
    self.term.train()
    self.opt.zero_grad(set_to_none=True)
    total_state = 0.0
    total_term = 0.0
    for _ in range(self.chunks):
      t, e = replay.sample_windows(self.chunk_size, self.window)
      batch = replay.window(t, e, self.window)
      xi = xi_all[batch["env"]]
      state_loss, term_loss = self.window_loss(batch, xi)
      (state_loss + term_loss).backward(retain_graph=False)
      total_state += float(state_loss.detach())
      total_term += float(term_loss.detach())
    gn = torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
    self.opt.step()
    self.metrics = {
      "wm/state_loss": total_state / self.chunks,
      "wm/term_loss": total_term / self.chunks,
      "wm/grad_norm": float(gn),
    }
    return self.metrics

  # --- online accuracy metric -------------------------------------------

  @torch.no_grad()
  def prediction_error(
    self,
    s0: torch.Tensor,
    actions: torch.Tensor,
    xi: torch.Tensor,
    clip: torch.Tensor,
    frame: torch.Tensor,
    s_target: torch.Tensor,
  ) -> dict[str, float]:
    """Figure 3(c): state-only k-step error, normalised by the nominal error.

    ``actions`` is (B, k, NUM_JOINTS) of logged actions and ``s_target`` the
    logged state k steps after ``s0``.  Measured online on policy rollouts.
    """
    k = actions.shape[1]
    s_hat = s0
    for j in range(k):
      adv = self.library.advance_at(clip[:, j], frame[:, j])
      s_hat, _ = self.step(s_hat, actions[:, j], xi, adv)
    err = (s_hat - s_target) / self.state_scale
    return {
      "wm/pred_err_abs": float(err.abs().mean()),
      "wm/pred_err_rmse": float(err.square().mean().sqrt()),
    }

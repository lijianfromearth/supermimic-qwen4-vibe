"""Analytic observation / reward / termination heads of the SuperMimic MDP.

One implementation is used on both sides of the world model: the simulator
side feeds it fresh qpos/qvel (through the differentiable FK), the imagination
side feeds it the predicted state.  Reward follows BeyondMimic as instantiated
in mjlab, with the paper's two changes (no self-collision term, joint-limit
weight -2) and every term multiplied by dt = 0.02 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.quat import (
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  yaw_quat,
)

GRAVITY_W = (0.0, 0.0, -1.0)

# Table A.2: BeyondMimic's tracking reward with the self-collision penalty
# removed and the joint-limit weight lowered from -10 to -2.
REWARD_WEIGHTS = {
  "root_pos": 0.5,
  "root_ori": 0.5,
  "body_pos": 1.0,
  "body_ori": 1.0,
  "body_lin_vel": 1.0,
  "body_ang_vel": 1.0,
  "action_rate": -0.1,
  "joint_limit": -2.0,
}


@dataclass
class RefFrame:
  """Reference motion quantities for one frame, world frame (env origin added)."""

  anchor_pos_w: torch.Tensor  # (B, 3)
  anchor_quat_w: torch.Tensor  # (B, 4)
  body_pos_w: torch.Tensor  # (B, nt, 3)
  body_quat_w: torch.Tensor  # (B, nt, 4)
  body_lin_vel_w: torch.Tensor  # (B, nt, 3)
  body_ang_vel_w: torch.Tensor  # (B, nt, 3)
  joint_pos: torch.Tensor  # (B, nj)
  joint_vel: torch.Tensor  # (B, nj)


@dataclass
class RobotFrame:
  anchor_pos_w: torch.Tensor
  anchor_quat_w: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor
  root_quat_w: torch.Tensor
  root_ang_vel_w: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor


class TrackingMath:
  def __init__(
    self,
    default_joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    anchor_index: int,
    dt: float = 0.02,
    num_future_frames: int = 10,
    joint_limit_weight: float = -2.0,
    action_rate_weight: float = -0.1,
    height_threshold: float = 0.25,
    tilt_threshold: float = 0.8,
    soft_height_tau: float = 0.02,
    soft_tilt_tau: float = 0.02,
  ) -> None:
    self.q0 = default_joint_pos
    self.soft_limits = soft_joint_pos_limits
    self.anchor = anchor_index
    self.dt = dt
    self.K = num_future_frames
    self.joint_limit_weight = joint_limit_weight
    self.action_rate_weight = action_rate_weight
    self.weights = dict(REWARD_WEIGHTS)
    self.weights["joint_limit"] = joint_limit_weight
    self.weights["action_rate"] = action_rate_weight
    self.height_threshold = height_threshold
    self.tilt_threshold = tilt_threshold
    self.soft_height_tau = soft_height_tau
    self.soft_tilt_tau = soft_tilt_tau
    self.register_defaults()

  def register_defaults(self) -> None:
    dev = self.q0.device
    self.gravity_w = torch.tensor(GRAVITY_W, device=dev)
    # (joint_pos_rel, joint_vel_rel, imu_ang_vel, projected_gravity, prev_action)
    self.obs_noise_lo = torch.cat(
      [
        torch.full((self.q0.numel(),), -0.01, device=dev),
        torch.full((self.q0.numel(),), -0.5, device=dev),
        torch.full((3,), -0.2, device=dev),
        torch.full((3,), -0.05, device=dev),
        torch.zeros(self.q0.numel(), device=dev),
      ]
    )
    self.obs_noise_hi = -self.obs_noise_lo
    self.obs_dim = int(self.obs_noise_lo.numel())
    self.goal_dim = self.K * (2 * self.q0.numel() + 3 + 6)
    self.max_reward_per_step = (
      sum(w for w in self.weights.values() if w > 0.0) * self.dt
    )

  def to(self, device: str) -> "TrackingMath":
    self.q0 = self.q0.to(device)
    self.soft_limits = self.soft_limits.to(device)
    self.gravity_w = self.gravity_w.to(device)
    self.obs_noise_lo = self.obs_noise_lo.to(device)
    self.obs_noise_hi = self.obs_noise_hi.to(device)
    return self

  # --- observation head -------------------------------------------------

  def proprio(
    self,
    robot: RobotFrame,
    encoder_bias: torch.Tensor,
    prev_action: torch.Tensor,
  ) -> torch.Tensor:
    ang_vel_b = quat_apply_inverse(robot.root_quat_w, robot.root_ang_vel_w)
    gravity_b = quat_apply_inverse(robot.root_quat_w, self.gravity_w)
    return torch.cat(
      [
        robot.joint_pos + encoder_bias - self.q0,
        robot.joint_vel,
        ang_vel_b,
        gravity_b,
        prev_action,
      ],
      dim=-1,
    )

  def corrupt(self, obs: torch.Tensor) -> torch.Tensor:
    return (
      obs
      + torch.rand_like(obs) * (self.obs_noise_hi - self.obs_noise_lo)
      + self.obs_noise_lo
    )

  # --- reward head ------------------------------------------------------

  def align_reference(
    self, robot: RobotFrame, ref: RefFrame
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """BeyondMimic's relative-body alignment: robot xy+yaw, reference height."""
    delta_pos = robot.anchor_pos_w.clone()
    delta_pos[..., 2] = ref.anchor_pos_w[..., 2]
    delta_ori = yaw_quat(quat_mul(robot.anchor_quat_w, quat_inv(ref.anchor_quat_w)))
    return delta_pos, delta_ori

  def reward_components(
    self,
    robot: RobotFrame,
    ref: RefFrame,
    prev_action: torch.Tensor,
    action: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    delta_pos, delta_ori = self.align_reference(robot, ref)
    ref_anchor_rep = ref.anchor_pos_w[:, None, :].expand_as(ref.body_pos_w)
    rel_pos = delta_pos[:, None, :] + quat_apply(
      delta_ori[:, None, :], ref.body_pos_w - ref_anchor_rep
    )
    rel_quat = quat_mul(delta_ori[:, None, :], ref.body_quat_w)

    c: dict[str, torch.Tensor] = {}
    c["root_pos"] = torch.exp(
      -((ref.anchor_pos_w - robot.anchor_pos_w) ** 2).sum(-1) / 0.3**2
    )
    c["root_ori"] = torch.exp(
      -(quat_error_magnitude(ref.anchor_quat_w, robot.anchor_quat_w) ** 2) / 0.4**2
    )
    c["body_pos"] = torch.exp(
      -((rel_pos - robot.body_pos_w) ** 2).sum(-1).mean(-1) / 0.3**2
    )
    c["body_ori"] = torch.exp(
      -quat_error_magnitude(rel_quat, robot.body_quat_w).pow(2).mean(-1) / 0.4**2
    )
    c["body_lin_vel"] = torch.exp(
      -((ref.body_lin_vel_w - robot.body_lin_vel_w) ** 2).sum(-1).mean(-1) / 1.0**2
    )
    c["body_ang_vel"] = torch.exp(
      -((ref.body_ang_vel_w - robot.body_ang_vel_w) ** 2).sum(-1).mean(-1) / 3.14**2
    )
    c["action_rate"] = ((action - prev_action) ** 2).sum(-1)
    lo = self.soft_limits[..., 0]
    hi = self.soft_limits[..., 1]
    c["joint_limit"] = (
      (robot.joint_pos - hi).clip(min=0.0) + (lo - robot.joint_pos).clip(min=0.0)
    ).sum(-1)
    return c

  def reward(
    self,
    robot: RobotFrame,
    ref: RefFrame,
    prev_action: torch.Tensor,
    action: torch.Tensor,
  ) -> torch.Tensor:
    c = self.reward_components(robot, ref, prev_action, action)
    return sum(self.weights[k] * v for k, v in c.items()) * self.dt

  # --- termination head -------------------------------------------------

  def termination_errors(
    self, robot: RobotFrame, ref: RefFrame
  ) -> tuple[torch.Tensor, torch.Tensor]:
    height_err = (ref.anchor_pos_w[..., 2] - robot.anchor_pos_w[..., 2]).abs()
    g_ref = quat_apply_inverse(ref.anchor_quat_w, self.gravity_w)[..., 2]
    g_robot = quat_apply_inverse(robot.anchor_quat_w, self.gravity_w)[..., 2]
    tilt_err = (g_ref - g_robot).abs()
    return height_err, tilt_err

  def terminated(self, robot: RobotFrame, ref: RefFrame) -> torch.Tensor:
    h, t = self.termination_errors(robot, ref)
    return (h > self.height_threshold) | (t > self.tilt_threshold)

  def termination_prob(self, robot: RobotFrame, ref: RefFrame) -> torch.Tensor:
    """Differentiable survival weight used inside imagined rollouts."""
    h, t = self.termination_errors(robot, ref)
    p_h = torch.sigmoid((h - self.height_threshold) / self.soft_height_tau)
    p_t = torch.sigmoid((t - self.tilt_threshold) / self.soft_tilt_tau)
    return 1.0 - (1.0 - p_h) * (1.0 - p_t)

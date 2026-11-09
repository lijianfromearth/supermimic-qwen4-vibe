"""Canonical state representation shared by the simulator and the world model.

MuJoCo's flat-ground dynamics is invariant to horizontal translation and to
rotation about gravity, so the learned state is expressed in a frame whose
origin is the robot root's xy and whose yaw is the robot root's yaw:

  s[0]      root height z
  s[1:5]    root quaternion with yaw removed (wxyz)
  s[5:8]    root linear velocity, robot-yaw frame
  s[8:11]   root angular velocity, root body frame
  s[11:40]  hinge positions
  s[40:69]  hinge velocities
  s[69:72]  reference anchor position minus root position, robot-yaw frame
  s[72:76]  reference anchor quaternion, robot-yaw frame

The last seven components carry the tracking error, which is *not* invariant;
they are predicted like the rest and are supervised from the simulator.  The
exogenous reference advance (``ref_adv``) is fed to the dynamics network so the
map is closed over a multi-clip library.

The motion library is stored in a *reference-canonical* frame: at every frame
the reference anchor sits at (0, 0, z) with zero yaw.  ``reconstruct_reference``
maps it into the robot-canonical frame using s[69:76].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.kinematics import Kinematics
from supermimic.mdp.tracking_math import RefFrame, RobotFrame
from supermimic.quat import (
  matrix_from_quat,
  normalize,
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  yaw_quat,
)

NUM_JOINTS = 29
STATE_DIM = 11 + 2 * NUM_JOINTS + 7  # 76
PHYS_DIM = 11 + 2 * NUM_JOINTS  # 69
AUX_SLICE = slice(69, 76)
REF_ADV_DIM = 9
XI_DIM = 1 + NUM_JOINTS + 3  # friction, encoder bias, torso com offset

# Nominal per-component magnitudes.  These normalise the network inputs and
# double as the paper's "predefined nominal error" for the online 8-step
# prediction-error metric.
STATE_SCALE = torch.cat(
  [
    torch.full((1,), 0.1),
    torch.full((4,), 0.2),
    torch.full((3,), 1.0),
    torch.full((3,), 2.0),
    torch.full((NUM_JOINTS,), 0.3),
    torch.full((NUM_JOINTS,), 2.0),
    torch.full((3,), 0.1),
    torch.full((4,), 0.2),
  ]
)
XI_SCALE = torch.cat(
  [
    torch.full((1,), 0.5),
    torch.full((NUM_JOINTS,), 0.01),
    torch.full((3,), 0.05),
  ]
)
REF_ADV_SCALE = torch.tensor([0.05, 0.05, 0.02, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2])
ACTION_SCALE = 1.0

# Per-frame goal block: joint_pos, joint_vel, anchor-relative position, 6d rotation.
GOAL_BLOCK_SCALE = torch.cat(
  [
    torch.full((NUM_JOINTS,), 1.0),
    torch.full((NUM_JOINTS,), 2.0),
    torch.full((3,), 0.5),
    torch.full((6,), 1.0),
  ]
)
# Proprioception block: q-q0, qd, base angular velocity, projected gravity, a_{t-1}.
OBS_SCALE = torch.cat(
  [
    torch.full((NUM_JOINTS,), 0.3),
    torch.full((NUM_JOINTS,), 2.0),
    torch.full((3,), 2.0),
    torch.full((3,), 1.0),
    torch.full((NUM_JOINTS,), ACTION_SCALE),
  ]
)


def quat_yaw_angle(q: torch.Tensor) -> torch.Tensor:
  """Yaw angle in [-pi, pi] of a wxyz quaternion."""
  qw, qx, qy, qz = q.unbind(-1)
  return torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def pack_state(
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  root_lin_vel: torch.Tensor,
  root_ang_vel: torch.Tensor,
  q: torch.Tensor,
  qd: torch.Tensor,
  ref_anchor_pos: torch.Tensor,
  ref_anchor_quat: torch.Tensor,
) -> torch.Tensor:
  """World-frame simulator quantities -> canonical state (B, 76)."""
  y = yaw_quat(root_quat)
  yi = quat_inv(y)
  q_rp = normalize(quat_mul(yi, root_quat))
  v_yaw = quat_apply_inverse(y, root_lin_vel)
  w_b = quat_apply_inverse(root_quat, root_ang_vel)
  e = quat_apply_inverse(y, ref_anchor_pos - root_pos)
  q_rel = normalize(quat_mul(yi, ref_anchor_quat))
  return torch.cat([root_pos[:, 2:3], q_rp, v_yaw, w_b, q, qd, e, q_rel], dim=-1)


def unpack_state(s: torch.Tensor) -> dict[str, torch.Tensor]:
  """Canonical state -> robot-canonical world frame (root xy = 0, yaw = 0)."""
  q_rp = normalize(s[:, 1:5])
  return {
    "root_pos": torch.cat([torch.zeros_like(s[:, 0:2]), s[:, 0:1]], dim=-1),
    "root_quat": q_rp,
    "root_lin_vel": s[:, 5:8],
    "root_ang_vel": quat_apply(q_rp, s[:, 8:11]),
    "q": s[:, 11 : 11 + NUM_JOINTS],
    "qd": s[:, 11 + NUM_JOINTS : PHYS_DIM],
  }


def robot_frame(s: torch.Tensor, kin: Kinematics) -> RobotFrame:
  """Analytic robot head: FK on the canonical state."""
  u = unpack_state(s)
  pos, quat, lin, ang = kin.fk(
    u["root_pos"], u["root_quat"], u["root_lin_vel"], u["root_ang_vel"], u["q"], u["qd"]
  )
  a = kin.tracked_anchor
  return RobotFrame(
    anchor_pos_w=pos[:, a],
    anchor_quat_w=quat[:, a],
    body_pos_w=pos,
    body_quat_w=quat,
    body_lin_vel_w=lin,
    body_ang_vel_w=ang,
    root_quat_w=u["root_quat"],
    root_ang_vel_w=u["root_ang_vel"],
    joint_pos=u["q"],
    joint_vel=u["qd"],
  )


@dataclass
class CanonRef:
  """One library frame in the reference-canonical frame (anchor at xy 0, yaw 0)."""

  anchor_pos: torch.Tensor  # (B, 3) == (0, 0, z)
  anchor_quat: torch.Tensor  # (B, 4) yaw-free
  body_pos: torch.Tensor  # (B, nt, 3)
  body_quat: torch.Tensor  # (B, nt, 4)
  body_lin_vel: torch.Tensor  # (B, nt, 3)
  body_ang_vel: torch.Tensor  # (B, nt, 3)
  joint_pos: torch.Tensor  # (B, nj)
  joint_vel: torch.Tensor  # (B, nj)


def canonicalize_reference(
  anchor_pos: torch.Tensor,
  anchor_quat: torch.Tensor,
  body_pos: torch.Tensor,
  body_quat: torch.Tensor,
  body_lin_vel: torch.Tensor,
  body_ang_vel: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> CanonRef:
  """Translate by the anchor xy and de-yaw by the anchor yaw."""
  y = yaw_quat(anchor_quat)
  yi = quat_inv(y)
  origin = torch.cat([anchor_pos[:, :2], torch.zeros_like(anchor_pos[:, 2:3])], dim=-1)
  yb = y[:, None, :].expand_as(body_quat)
  yib = yi[:, None, :].expand_as(body_quat)
  return CanonRef(
    anchor_pos=torch.cat(
      [torch.zeros_like(anchor_pos[:, :2]), anchor_pos[:, 2:3]], dim=-1
    ),
    anchor_quat=normalize(quat_mul(yi, anchor_quat)),
    body_pos=quat_apply_inverse(yb, body_pos - origin[:, None, :]),
    body_quat=normalize(quat_mul(yib, body_quat)),
    body_lin_vel=quat_apply_inverse(yb, body_lin_vel),
    body_ang_vel=quat_apply_inverse(yb, body_ang_vel),
    joint_pos=joint_pos,
    joint_vel=joint_vel,
  )


def reconstruct_reference(c: CanonRef, s: torch.Tensor) -> RefFrame:
  """Reference-canonical library frame -> the robot-canonical frame of ``s``."""
  e = s[:, 69:72]
  q_rel = normalize(s[:, 72:76])
  yaw_rel = yaw_quat(q_rel)
  shift = torch.cat([e[:, :2], torch.zeros_like(e[:, :1])], dim=-1)
  yb = yaw_rel[:, None, :].expand_as(c.body_quat)
  anchor_pos = torch.cat([e[:, :2], c.anchor_pos[:, 2:3]], dim=-1)
  return RefFrame(
    anchor_pos_w=anchor_pos,
    anchor_quat_w=q_rel,
    body_pos_w=quat_apply(yb, c.body_pos) + shift[:, None, :],
    body_quat_w=normalize(quat_mul(yb, c.body_quat)),
    body_lin_vel_w=quat_apply(yb, c.body_lin_vel),
    body_ang_vel_w=quat_apply(yb, c.body_ang_vel),
    joint_pos=c.joint_pos,
    joint_vel=c.joint_vel,
  )


def goal_from_future(
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  ref_err: torch.Tensor,
  ref_quat: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  anchor_z: torch.Tensor,
) -> torch.Tensor:
  """K future reference frames against the robot anchor, batched over K.

  ``reconstruct_reference`` gives every future frame the same anchor orientation
  (``ref_quat``) and the same horizontal offset (``ref_err[:2]``); only the anchor
  height varies.  The relative rotation and its 6d encoding are therefore shared
  across K, and the body arrays the full reconstruction builds are never read by
  the goal head.  Exploiting both keeps the imagined rollout to a handful of
  kernels per step instead of ten reconstructions.

  ``joint_pos``/``joint_vel`` are (B, K, nj), ``anchor_z`` is (B, K).
  """
  B, K = joint_pos.shape[:2]
  q_rel = normalize(ref_quat)
  q10 = quat_inv(anchor_quat_w)
  q12 = quat_mul(q10, q_rel)
  d_xy = (ref_err[:, :2] - anchor_pos_w[:, :2])[:, None, :].expand(B, K, 2)
  d_z = (anchor_z - anchor_pos_w[:, 2:3])[..., None]
  pos_b = quat_apply(q10[:, None, :], torch.cat([d_xy, d_z], dim=-1))
  ori_6d = matrix_from_quat(q12)[..., :2].reshape(B, 6)
  return torch.cat(
    [joint_pos, joint_vel, pos_b, ori_6d[:, None, :].expand(B, K, 6)], dim=-1
  ).flatten(1)


def reference_advance(
  anchor_pos: torch.Tensor,
  anchor_quat: torch.Tensor,
  next_anchor_pos: torch.Tensor,
  next_anchor_quat: torch.Tensor,
) -> torch.Tensor:
  """Exogenous per-step reference motion, in the current reference frame.

  (da_x, da_y, da_z, sin dpsi, cos dpsi, next anchor quat), all expressed in the
  frame whose yaw is the *current* reference anchor yaw.  This is exactly what
  the command term integrates to recover the reference's world pose, and what
  the dynamics network needs to propagate the tracking error.
  """
  yi = quat_inv(yaw_quat(anchor_quat))
  da = quat_apply_inverse(yaw_quat(anchor_quat), next_anchor_pos - anchor_pos)
  dpsi = quat_yaw_angle(next_anchor_quat) - quat_yaw_angle(anchor_quat)
  dpsi = torch.atan2(torch.sin(dpsi), torch.cos(dpsi))
  nq = normalize(quat_mul(yi, next_anchor_quat))
  return torch.cat(
    [da, torch.sin(dpsi).unsqueeze(-1), torch.cos(dpsi).unsqueeze(-1), nq], dim=-1
  )


def sim_context(
  friction: torch.Tensor, encoder_bias: torch.Tensor, com_offset: torch.Tensor
) -> torch.Tensor:
  return torch.cat([friction.reshape(-1, 1), encoder_bias, com_offset], dim=-1)

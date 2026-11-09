"""Broadcasting quaternion helpers.

mjlab's ``lab_api.math`` routines are TorchScript and require exactly matching
shapes; the analytic heads routinely multiply a (B, 1, 4) rotation against a
(B, nb, 4) one.  These versions broadcast, and are otherwise identical.

They are also plain eager PyTorch on purpose: the JIT executor re-optimises a
scripted function between calls, which changes the saved-tensor sequence and
breaks activation checkpointing of the imagined rollout.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (  # noqa: F401
  quat_from_euler_xyz,
  sample_uniform,
)


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  """Log map of a wxyz quaternion, with mjlab's small-angle Taylor branch.

  The unused branch's denominator is made safe so ``torch.where`` does not
  backpropagate a 0/0 NaN into the state when two rotations coincide exactly,
  which happens at every reset.
  """
  quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
  mag = torch.linalg.norm(quat[..., 1:], dim=-1)
  half_angle = torch.atan2(mag, quat[..., 0])
  angle = 2.0 * half_angle
  large = angle.abs() > eps
  angle_safe = torch.where(large, angle, torch.ones_like(angle))
  sin_half_over_angle = torch.where(
    large, torch.sin(half_angle) / angle_safe, 0.5 - angle * angle / 48.0
  )
  return quat[..., 1:4] / sin_half_over_angle.unsqueeze(-1)


def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  return axis_angle_from_quat(quat_mul(q1, quat_conjugate(q2)))


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  return torch.linalg.norm(quat_box_minus(q1, q2), dim=-1)


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
  return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
  return quat_conjugate(q) / q.square().sum(-1, keepdim=True).clamp_min(eps)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  w1, x1, y1, z1 = q1.unbind(-1)
  w2, x2, y2, z2 = q2.unbind(-1)
  return torch.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dim=-1,
  )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  xyz = q[..., 1:]
  shape = torch.broadcast_shapes(xyz.shape, v.shape)
  xyz, v, w = xyz.expand(shape), v.expand(shape), q[..., 0:1].expand(shape)
  t = torch.linalg.cross(xyz, v) * 2.0
  return v + w * t + torch.linalg.cross(xyz, t)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  xyz = q[..., 1:]
  shape = torch.broadcast_shapes(xyz.shape, v.shape)
  xyz, v, w = xyz.expand(shape), v.expand(shape), q[..., 0:1].expand(shape)
  t = torch.linalg.cross(xyz, v) * 2.0
  return v - w * t + torch.linalg.cross(xyz, t)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  half = yaw * 0.5
  zero = torch.zeros_like(half)
  return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
  r, i, j, k = q.unbind(-1)
  two_s = 2.0 / (q * q).sum(-1)
  o = torch.stack(
    [
      1 - two_s * (j * j + k * k),
      two_s * (i * j - k * r),
      two_s * (i * k + j * r),
      two_s * (i * j + k * r),
      1 - two_s * (i * i + k * k),
      two_s * (j * k - i * r),
      two_s * (i * k - j * r),
      two_s * (j * k + i * r),
      1 - two_s * (i * i + j * j),
    ],
    dim=-1,
  )
  return o.reshape(q.shape[:-1] + (3, 3))


def subtract_frame_transforms(
  t01: torch.Tensor,
  q01: torch.Tensor,
  t02: torch.Tensor | None = None,
  q02: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  q10 = quat_inv(q01)
  q12 = q10 if q02 is None else quat_mul(q10, q02)
  d = -t01 if t02 is None else t02 - t01
  return quat_apply(q10, d), q12

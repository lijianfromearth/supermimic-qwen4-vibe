"""Differentiable SE(3) / SO(3) math utilities for SuperMimic."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.jit.script
def quat_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
  """Convert rotations given as quaternions (w, x, y, z) to 3x3 rotation matrices."""
  r, i, j, k = torch.unbind(quaternions, -1)
  two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp(min=1e-12)

  o = torch.stack(
    (
      1 - two_s * (j * j + k * k),
      two_s * (i * j - k * r),
      two_s * (i * k + j * r),
      two_s * (i * j + k * r),
      1 - two_s * (i * i + k * k),
      two_s * (j * k - i * r),
      two_s * (i * k - j * r),
      two_s * (j * k + i * r),
      1 - two_s * (i * i + j * j),
    ),
    -1,
  )
  return o.reshape(quaternions.shape[:-1] + (3, 3))


@torch.jit.script
def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
  """Extract 6D rotation representation (first two columns) from 3x3 rotation matrix.
  Matches mjlab's matrix[..., :2].reshape(..., 6).
  """
  return matrix[..., :2].reshape(matrix.shape[:-2] + (6,))


@torch.jit.script
def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
  """Differentiable Gram-Schmidt orthonormalization from 6D representation to SO(3).
  rot6d has shape (..., 6) representing the first two columns reshaped from (..., 3, 2).
  """
  mat32 = rot6d.reshape(rot6d.shape[:-1] + (3, 2))
  a1 = mat32[..., 0]
  a2 = mat32[..., 1]
  b1 = F.normalize(a1, dim=-1, eps=1e-8)
  proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
  b2 = F.normalize(a2 - proj, dim=-1, eps=1e-8)
  b3 = torch.linalg.cross(b1, b2, dim=-1)
  return torch.stack((b1, b2, b3), dim=-1)


@torch.jit.script
def geodesic_rot_error_sq(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
  """Compute ||log(R1 @ R2^T)||^2 smoothly and differentiably for R1, R2 in SO(3).
  Uses u = 1 - cos(theta) = 0.25 * ||R1 - R2||_F^2 in [0, 2] to guarantee
  non-negativity and exact, NaN-free derivatives at theta = 0.
  """
  u = (0.25 * torch.sum(torch.square(R1 - R2), dim=(-2, -1))).clamp(
    min=0.0, max=2.0 - 1e-6
  )
  small_mask = u < 1e-3
  # Taylor expansion for arccos(1 - u)^2 = 2u * (1 + u/6 + 3*u^2/40 + 5*u^3/112)
  taylor_sq = 2.0 * u * (1.0 + u * (1.0 / 6.0 + u * (3.0 / 40.0 + u * (5.0 / 112.0))))
  cos_theta = (1.0 - u).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
  exact_sq = torch.square(torch.acos(cos_theta))
  return torch.where(small_mask, taylor_sq, exact_sq)


@torch.jit.script
def extract_yaw_cos_sin(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Extract (cos(yaw), sin(yaw)) from rotation matrix R (..., 3, 3) matching yaw_quat."""
  rx = R[..., 0, 0]
  ry = R[..., 1, 0]
  norm = torch.sqrt(rx * rx + ry * ry + 1e-12)
  return rx / norm, ry / norm


@torch.jit.script
def make_yaw_matrix(cos_yaw: torch.Tensor, sin_yaw: torch.Tensor) -> torch.Tensor:
  """Construct 3x3 rotation matrix R_z(yaw) from cos(yaw), sin(yaw)."""
  zero = torch.zeros_like(cos_yaw)
  one = torch.ones_like(cos_yaw)
  row0 = torch.stack((cos_yaw, -sin_yaw, zero), dim=-1)
  row1 = torch.stack((sin_yaw, cos_yaw, zero), dim=-1)
  row2 = torch.stack((zero, zero, one), dim=-1)
  return torch.stack((row0, row1, row2), dim=-2)


@torch.jit.script
def rotate_vec_z(vec: torch.Tensor, cos_yaw: torch.Tensor, sin_yaw: torch.Tensor) -> torch.Tensor:
  """Rotate 3D vector(s) around Z-axis by angle with given cos_yaw, sin_yaw."""
  vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]
  nx = cos_yaw * vx - sin_yaw * vy
  ny = sin_yaw * vx + cos_yaw * vy
  return torch.stack((nx, ny, vz), dim=-1)


@torch.jit.script
def rotate_mat_z(mat: torch.Tensor, cos_yaw: torch.Tensor, sin_yaw: torch.Tensor) -> torch.Tensor:
  """Compute R_z(yaw) @ mat for 3x3 matrix mat (..., 3, 3)."""
  r0 = mat[..., 0, :]
  r1 = mat[..., 1, :]
  r2 = mat[..., 2, :]
  c = cos_yaw.unsqueeze(-1)
  s = sin_yaw.unsqueeze(-1)
  nr0 = c * r0 - s * r1
  nr1 = s * r0 + c * r1
  return torch.stack((nr0, nr1, r2), dim=-2)


@torch.jit.script
def compute_relative_body_poses_rot(
  ref_root_pos: torch.Tensor,
  ref_root_rot: torch.Tensor,
  ref_body_pos: torch.Tensor,
  ref_body_rot: torch.Tensor,
  robot_root_pos: torch.Tensor,
  robot_root_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Differentiable equivalent of mjlab's update_relative_body_poses using 3x3 rotation matrices.
  Matches yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w))) without gimbal lock
  during prone/supine poses (e.g. fallAndGetUp and floor rolls).
  ref_root_pos, robot_root_pos: (B, 3)
  ref_root_rot, robot_root_rot: (B, 3, 3)
  ref_body_pos: (B, N, 3)
  ref_body_rot: (B, N, 3, 3)
  """
  rel_root_rot = torch.matmul(robot_root_rot, ref_root_rot.transpose(-1, -2))
  cos_d, sin_d = extract_yaw_cos_sin(rel_root_rot)

  rel_pos = ref_body_pos - ref_root_pos.unsqueeze(1)
  rot_rel_pos = rotate_vec_z(rel_pos, cos_d.unsqueeze(1), sin_d.unsqueeze(1))
  anchor_xy_zref = torch.stack(
    (robot_root_pos[:, 0], robot_root_pos[:, 1], ref_root_pos[:, 2]), dim=-1
  )
  body_pos_rel_w = anchor_xy_zref.unsqueeze(1) + rot_rel_pos
  body_rot_rel_w = rotate_mat_z(ref_body_rot, cos_d.unsqueeze(1), sin_d.unsqueeze(1))
  return body_pos_rel_w, body_rot_rel_w

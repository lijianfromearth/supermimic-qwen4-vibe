"""Preprocess LAFAN1 Unitree G1 CSV motion files (30 FPS) to 50 FPS with exact FK."""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  matrix_from_quat,
  quat_conjugate,
  quat_from_matrix,
  quat_mul,
)

TRACKED_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


def build_g1_fk_constants(device: str = "cuda"):
  spec = get_g1_robot_cfg().spec_fn()
  mj_model = spec.compile()

  body_names = [mj_model.body(i).name for i in range(1, mj_model.nbody)]
  tracked_indices = [body_names.index(name) for name in TRACKED_BODY_NAMES]

  body_parent = [int(mj_model.body_parentid[i]) - 1 for i in range(1, mj_model.nbody)]
  body_pos_local = torch.tensor(
    mj_model.body_pos[1:], dtype=torch.float32, device=device
  )
  body_quat_local = torch.tensor(
    mj_model.body_quat[1:], dtype=torch.float32, device=device
  )
  body_rot_local = matrix_from_quat(body_quat_local)
  jnt_axis = torch.tensor(mj_model.jnt_axis[1:], dtype=torch.float32, device=device)

  K = torch.zeros(29, 3, 3, device=device)
  K[:, 0, 1] = -jnt_axis[:, 2]
  K[:, 0, 2] = jnt_axis[:, 1]
  K[:, 1, 0] = jnt_axis[:, 2]
  K[:, 1, 2] = -jnt_axis[:, 0]
  K[:, 2, 0] = -jnt_axis[:, 1]
  K[:, 2, 1] = jnt_axis[:, 0]
  K2 = torch.bmm(K, K)
  R0 = body_rot_local[1:]
  R0_K = torch.bmm(R0, K)
  R0_K2 = torch.bmm(R0, K2)

  return {
    "body_names": body_names,
    "tracked_names": TRACKED_BODY_NAMES,
    "tracked_indices": tracked_indices,
    "body_parent": body_parent,
    "body_pos_local": body_pos_local,
    "jnt_axis": jnt_axis,
    "R0": R0,
    "R0_K": R0_K,
    "R0_K2": R0_K2,
  }


@torch.no_grad()
def compute_fk_batched(
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  root_lin_vel: torch.Tensor,
  root_ang_vel: torch.Tensor,
  dof_pos: torch.Tensor,
  dof_vel: torch.Tensor,
  fk_consts: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute exact FK for all 30 bodies and return the 14 tracked bodies."""
  R0 = fk_consts["R0"]
  R0_K = fk_consts["R0_K"]
  R0_K2 = fk_consts["R0_K2"]
  body_parent = fk_consts["body_parent"]
  body_pos_local = fk_consts["body_pos_local"]
  jnt_axis = fk_consts["jnt_axis"]
  tracked_indices = fk_consts["tracked_indices"]

  sin_q = torch.sin(dof_pos).unsqueeze(-1).unsqueeze(-1)
  cos_q = (1.0 - torch.cos(dof_pos)).unsqueeze(-1).unsqueeze(-1)
  R_rel = R0.unsqueeze(0) + sin_q * R0_K.unsqueeze(0) + cos_q * R0_K2.unsqueeze(0)

  pos_list = [None] * 30
  rot_list = [None] * 30
  lvel_list = [None] * 30
  avel_list = [None] * 30

  pos_list[0] = root_pos
  rot_list[0] = matrix_from_quat(root_quat)
  lvel_list[0] = root_lin_vel
  avel_list[0] = root_ang_vel

  for i in range(1, 30):
    p = body_parent[i]
    j = i - 1
    p_pos = pos_list[p]
    p_rot = rot_list[p]
    p_lvel = lvel_list[p]
    p_avel = avel_list[p]

    offset_w = torch.matmul(p_rot, body_pos_local[i])
    c_pos = p_pos + offset_w
    c_rot = torch.matmul(p_rot, R_rel[:, j])
    c_lvel = p_lvel + torch.linalg.cross(p_avel, offset_w)
    axis_w = torch.matmul(c_rot, jnt_axis[j])
    c_avel = p_avel + axis_w * dof_vel[:, j : j + 1]

    pos_list[i] = c_pos
    rot_list[i] = c_rot
    lvel_list[i] = c_lvel
    avel_list[i] = c_avel

  t_pos = torch.stack([pos_list[i] for i in tracked_indices], dim=1)
  t_rot = torch.stack([rot_list[i] for i in tracked_indices], dim=1)
  t_quat = quat_from_matrix(t_rot)
  t_lvel = torch.stack([lvel_list[i] for i in tracked_indices], dim=1)
  t_avel = torch.stack([avel_list[i] for i in tracked_indices], dim=1)
  return t_pos, t_quat, t_lvel, t_avel


def batched_slerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
  """Vectorized spherical linear interpolation for quaternions (w, x, y, z).
  q0, q1: (T, 4), blend: (T, 1)
  """
  cos_half_theta = torch.sum(q0 * q1, dim=-1, keepdim=True)
  neg_mask = cos_half_theta < 0.0
  q1 = torch.where(neg_mask, -q1, q1)
  cos_half_theta = torch.abs(cos_half_theta)

  close_mask = cos_half_theta >= 1.0 - 1e-7
  half_theta = torch.acos(torch.clamp(cos_half_theta, -1.0, 1.0))
  sin_half_theta = torch.sqrt(torch.clamp(1.0 - cos_half_theta * cos_half_theta, min=1e-12))

  ratio_a = torch.sin((1.0 - blend) * half_theta) / sin_half_theta
  ratio_b = torch.sin(blend * half_theta) / sin_half_theta

  lerp_q = (1.0 - blend) * q0 + blend * q1
  slerp_q = ratio_a * q0 + ratio_b * q1
  out = torch.where(close_mask, lerp_q, slerp_q)
  return out / torch.linalg.norm(out, dim=-1, keepdim=True)


def so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
  """Compute angular velocities from quaternion trajectory (T, 4) at rate dt."""
  q_prev, q_next = rotations[:-2], rotations[2:]
  q_rel = quat_mul(q_next, quat_conjugate(q_prev))
  omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
  omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
  return omega


def process_csv_clip(
  csv_path: str, fk_consts: dict, input_fps: float = 30.0, output_fps: float = 50.0, device: str = "cuda"
) -> dict[str, torch.Tensor]:
  raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
  data = torch.from_numpy(raw).to(device)

  base_pos_in = data[:, :3]
  # Convert xyzw in CSV to wxyz
  base_quat_in = data[:, 3:7][:, [3, 0, 1, 2]]
  base_quat_in = base_quat_in / torch.linalg.norm(base_quat_in, dim=-1, keepdim=True)
  dof_pos_in = data[:, 7:]

  n_in = data.shape[0]
  input_dt = 1.0 / input_fps
  output_dt = 1.0 / output_fps
  duration = (n_in - 1) * input_dt

  times = torch.arange(0, duration, output_dt, device=device, dtype=torch.float32)
  phase = times / duration
  index_0 = (phase * (n_in - 1)).floor().long()
  index_1 = torch.minimum(index_0 + 1, torch.tensor(n_in - 1, device=device))
  blend = ((phase * (n_in - 1)) - index_0.float()).unsqueeze(-1)

  base_pos = (1.0 - blend) * base_pos_in[index_0] + blend * base_pos_in[index_1]
  dof_pos = (1.0 - blend) * dof_pos_in[index_0] + blend * dof_pos_in[index_1]
  base_quat = batched_slerp(base_quat_in[index_0], base_quat_in[index_1], blend)

  # Finite difference velocities matching np.gradient(..., output_dt, axis=0)
  base_lin_vel = torch.gradient(base_pos, spacing=output_dt, dim=0)[0]
  dof_vel = torch.gradient(dof_pos, spacing=output_dt, dim=0)[0]
  base_ang_vel = so3_derivative(base_quat, output_dt)

  # Compute exact FK for the 14 tracked bodies
  t_pos, t_quat, t_lvel, t_avel = compute_fk_batched(
    base_pos, base_quat, base_lin_vel, base_ang_vel, dof_pos, dof_vel, fk_consts
  )

  return {
    "joint_pos": dof_pos.cpu(),
    "joint_vel": dof_vel.cpu(),
    "body_pos_w": t_pos.cpu(),
    "body_quat_w": t_quat.cpu(),
    "body_lin_vel_w": t_lvel.cpu(),
    "body_ang_vel_w": t_avel.cpu(),
  }


def main():
  t0 = time.perf_counter()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  fk_consts = build_g1_fk_constants(device=device)

  csv_files = sorted(glob.glob("/home/sunteng/carbon-test/lafan1_g1/*.csv"))
  assert len(csv_files) == 40, f"Expected 40 CSV files, found {len(csv_files)}"

  clip_names = []
  clip_lengths = []
  all_joint_pos = []
  all_joint_vel = []
  all_body_pos_w = []
  all_body_quat_w = []
  all_body_lin_vel_w = []
  all_body_ang_vel_w = []

  for f in csv_files:
    name = Path(f).stem
    clip = process_csv_clip(f, fk_consts, device=device)
    clip_names.append(name)
    clip_lengths.append(clip["joint_pos"].shape[0])
    all_joint_pos.append(clip["joint_pos"])
    all_joint_vel.append(clip["joint_vel"])
    all_body_pos_w.append(clip["body_pos_w"])
    all_body_quat_w.append(clip["body_quat_w"])
    all_body_lin_vel_w.append(clip["body_lin_vel_w"])
    all_body_ang_vel_w.append(clip["body_ang_vel_w"])

  clip_lengths_t = torch.tensor(clip_lengths, dtype=torch.long)
  clip_starts_t = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(clip_lengths_t[:-1], dim=0)])

  dataset = {
    "clip_names": clip_names,
    "clip_lengths": clip_lengths_t,
    "clip_starts": clip_starts_t,
    "tracked_body_names": TRACKED_BODY_NAMES,
    "joint_pos": torch.cat(all_joint_pos, dim=0),
    "joint_vel": torch.cat(all_joint_vel, dim=0),
    "body_pos_w": torch.cat(all_body_pos_w, dim=0),
    "body_quat_w": torch.cat(all_body_quat_w, dim=0),
    "body_lin_vel_w": torch.cat(all_body_lin_vel_w, dim=0),
    "body_ang_vel_w": torch.cat(all_body_ang_vel_w, dim=0),
  }

  out_dir = Path("/home/sunteng/carbon-test/data")
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / "lafan1_g1_50fps.pt"
  torch.save(dataset, out_path)
  dt = time.perf_counter() - t0
  total_frames = int(clip_lengths_t.sum().item())
  print(
    f"Processed {len(csv_files)} clips ({total_frames} frames, {total_frames / 50 / 3600:.3f} h) "
    f"in {dt:.2f}s -> {out_path}"
  )


if __name__ == "__main__":
  main()

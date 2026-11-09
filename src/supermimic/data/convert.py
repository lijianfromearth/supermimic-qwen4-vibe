"""Convert the LAFAN1-G1 retargeted CSV library into mjlab motion npz files.

Adapted from mjlab/src/mjlab/scripts/csv_to_npz.py (batched over frames, no wandb).
CSV layout: root pos (3), root quat xyzw (4), 29 joint positions, at 30 Hz.
Output: mjlab MotionLoader npz at 50 Hz, one file per clip.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul

from supermimic.data import LAFAN1_CSV_DIR, MOTIONS_DIR


def quat_slerp_batch(
  q1: torch.Tensor, q2: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
  """Batched SLERP for (w, x, y, z) quaternions; tau is shape (N,)."""
  d = (q1 * q2).sum(-1, keepdim=True)
  q2 = torch.where(d < 0.0, -q2, q2)
  d = d.abs()
  angle = torch.acos(d.clamp(-1.0, 1.0))
  small = angle.abs() < 1e-6
  isin = 1.0 / torch.sin(angle).clamp_min(1e-8)
  t = tau[:, None]
  out = q1 * torch.sin((1.0 - t) * angle) * isin + q2 * torch.sin(t * angle) * isin
  return torch.where(small, q1, out)


JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]


def resample(csv_path: Path, input_fps: float, output_fps: float, device: str):
  """Interpolate a CSV clip onto the output frame rate and differentiate it."""
  motion = torch.from_numpy(np.loadtxt(csv_path, delimiter=",")).to(torch.float32)
  motion = motion.to(device)
  base_pos = motion[:, :3]
  base_quat = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
  dof_pos = motion[:, 7:]

  input_frames = motion.shape[0]
  duration = (input_frames - 1) / input_fps
  output_dt = 1.0 / output_fps
  num_frames = math.floor(duration / output_dt + 1e-6) + 1
  times = torch.arange(num_frames, device=device, dtype=torch.float32) * output_dt

  phase = times / duration
  idx0 = (phase * (input_frames - 1)).floor().long()
  idx1 = torch.minimum(idx0 + 1, torch.tensor(input_frames - 1, device=device))
  blend = phase * (input_frames - 1) - idx0

  out_pos = base_pos[idx0] * (1 - blend[:, None]) + base_pos[idx1] * blend[:, None]
  out_dof = dof_pos[idx0] * (1 - blend[:, None]) + dof_pos[idx1] * blend[:, None]
  # Batched slerp (mjlab's helper is per-frame; vectorise it here).
  out_quat = quat_slerp_batch(base_quat[idx0], base_quat[idx1], blend)

  lin_vel = torch.gradient(out_pos, spacing=output_dt, dim=0)[0]
  dof_vel = torch.gradient(out_dof, spacing=output_dt, dim=0)[0]
  q_rel = quat_mul(out_quat[2:], quat_conjugate(out_quat[:-2]))
  omega = axis_angle_from_quat(q_rel) / (2.0 * output_dt)
  ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
  return out_pos, out_quat, lin_vel, ang_vel, out_dof, dof_vel


def build_sim(num_envs: int, output_fps: float, device: str):
  env_cfg = unitree_g1_flat_tracking_env_cfg()
  env_cfg.scene.num_envs = num_envs
  env_cfg.scene.terrain.env_spacing = 0.0
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return scene, sim


def convert_clip(
  csv_path: Path,
  out_path: Path,
  scene: Scene,
  sim: Simulation,
  batch: int,
  input_fps: float,
  output_fps: float,
  device: str,
):
  robot: Entity = scene["robot"]
  joint_idx = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
  joint_idx_t = torch.tensor(joint_idx, device=device, dtype=torch.long)

  pos, quat, lin_vel, ang_vel, dof_pos, dof_vel = resample(
    csv_path, input_fps, output_fps, device
  )
  n = pos.shape[0]
  log: dict[str, list[np.ndarray]] = {
    k: []
    for k in (
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    )
  }
  scene.reset()
  for start in range(0, n, batch):
    stop = min(start + batch, n)
    b = stop - start
    ids = torch.arange(b, device=device)
    root = robot.data.default_root_state[ids].clone()
    root[:, 0:3] = pos[start:stop]
    root[:, 3:7] = quat[start:stop]
    root[:, 7:10] = lin_vel[start:stop]
    root[:, 10:13] = ang_vel[start:stop]
    robot.write_root_state_to_sim(root, env_ids=ids)
    jp = robot.data.default_joint_pos[ids].clone()
    jv = robot.data.default_joint_vel[ids].clone()
    jp[:, joint_idx_t] = dof_pos[start:stop]
    jv[:, joint_idx_t] = dof_vel[start:stop]
    robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    log["joint_pos"].append(robot.data.joint_pos[ids].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[ids].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[ids].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[ids].cpu().numpy().copy())
    log["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[ids].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[ids].cpu().numpy().copy()
    )

  arrays = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in log.items()}
  arrays["fps"] = np.array([output_fps], dtype=np.float32)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, **arrays)
  return arrays


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--csv-dir", type=Path, default=LAFAN1_CSV_DIR)
  ap.add_argument(
    "--out-dir",
    type=Path,
    default=MOTIONS_DIR,
  )
  ap.add_argument("--input-fps", type=float, default=30.0)
  ap.add_argument("--output-fps", type=float, default=50.0)
  ap.add_argument("--batch", type=int, default=512)
  ap.add_argument("--device", type=str, default="cuda:0")
  args = ap.parse_args()

  if args.device.startswith("cuda") and not torch.cuda.is_available():
    args.device = "cpu"
  scene, sim = build_sim(args.batch, args.output_fps, args.device)
  files = sorted(args.csv_dir.glob("*.csv"))
  total = 0
  for f in files:
    out = args.out_dir / (f.stem + ".npz")
    arrays = convert_clip(
      f, out, scene, sim, args.batch, args.input_fps, args.output_fps, args.device
    )
    n = arrays["joint_pos"].shape[0]
    total += n
    zmin = arrays["body_pos_w"][..., 2].min()
    print(
      f"{f.stem:32s} frames={n:6d} dur={n / args.output_fps:7.2f}s zmin={zmin:+.3f}"
    )
  print(f"total frames={total} hours={total / args.output_fps / 3600:.4f}")


if __name__ == "__main__":
  main()

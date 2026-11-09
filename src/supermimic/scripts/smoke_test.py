"""Smoke-test the SuperMimic env and validate the PyTorch FK against MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from supermimic.data import MOTIONS_DIR
from supermimic.env import SuperMimicEnv
from supermimic.env_cfg import supermimic_env_cfg
from supermimic.quat import quat_apply, quat_error_magnitude


def check_fk(env: SuperMimicEnv) -> None:
  """Compare the PyTorch FK against MuJoCo in the raw world frame."""
  robot = env.scene["robot"]
  kin = env.kinematics
  idx = robot.indexing
  qpos = env.sim.data.qpos
  qvel = env.sim.data.qvel
  fq, fv = idx.free_joint_q_adr, idx.free_joint_v_adr
  root_pos = qpos[:, fq[:3]]
  root_quat = qpos[:, fq[3:7]]
  root_lin = qvel[:, fv[:3]]
  root_ang = quat_apply(root_quat, qvel[:, fv[3:6]])
  q = qpos[:, idx.joint_q_adr]
  qd = qvel[:, idx.joint_v_adr]
  pos, quat, lin, ang = kin.fk_all(root_pos, root_quat, root_lin, root_ang, q, qd)
  mj_pos = robot.data.body_link_pos_w
  mj_quat = robot.data.body_link_quat_w
  mj_lin = robot.data.body_link_lin_vel_w
  mj_ang = robot.data.body_link_ang_vel_w
  assert pos.shape == mj_pos.shape, (pos.shape, mj_pos.shape)
  dp = (pos - mj_pos).abs().max().item()
  dq = quat_error_magnitude(quat, mj_quat).max().item()
  dl = (lin - mj_lin).abs().max().item()
  da = (ang - mj_ang).abs().max().item()
  print(
    f"FK max abs error: pos={dp:.3e} quat={dq:.3e} lin_vel={dl:.3e} ang_vel={da:.3e}"
  )
  if dp > 1e-4:
    err = (pos - mj_pos).norm(dim=-1).amax(0)
    worst = int(err.argmax())
    print("worst body", kin.body_names[worst], "err", err[worst].item())
    print("  fk ", pos[0, worst].tolist(), " mj ", mj_pos[0, worst].tolist())
  assert dp < 1e-4 and dq < 1e-4 and dl < 1e-3 and da < 1e-3, "FK mismatch"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument(
    "--motion-dir",
    type=Path,
    default=MOTIONS_DIR,
  )
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--steps", type=int, default=20)
  ap.add_argument("--eval-mode", action="store_true")
  args = ap.parse_args()

  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  cfg = supermimic_env_cfg(
    str(args.motion_dir), args.num_envs, eval_mode=args.eval_mode
  )
  env = SuperMimicEnv(cfg, device="cuda:0")
  print("obs dims", {k: v.shape for k, v in env.reset()[0].items()})
  check_fk(env)

  cmd = env.sm_state.cmd
  print("clips", cmd.library.num_clips, "frames", cmd.library.total_frames)
  print("xi", env.sm_state.xi().shape, "state", env.sm_state.state().shape)

  t0 = time.time()
  for i in range(args.steps):
    a = torch.randn(env.num_envs, 29, device=env.device) * 0.1
    obs, r, term, trunc, extras = env.step(a)
    if i == 0:
      print(
        "reward",
        r.shape,
        r.mean().item(),
        "term",
        term.sum().item(),
        "trunc",
        trunc.sum().item(),
      )
      check_fk(env)
  env.sim.synchronize() if hasattr(env.sim, "synchronize") else None
  torch.cuda.synchronize()
  dt = (time.time() - t0) / args.steps
  print(
    f"step time {dt * 1e3:.2f} ms for {env.num_envs} envs -> {env.num_envs / dt:.0f} env-steps/s"
  )
  print("peak GPU GiB", torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
  main()

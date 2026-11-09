"""Diagnose the eval plateau: world-model accuracy and imagination-vs-real gap.

Loads a checkpoint, collects fresh on-policy replay, then reports:
  1. k-step state error per component block and per horizon (Fig. 3(c) detail),
  2. termination-head calibration,
  3. survival and reward of H-step imagined rollouts vs the real simulator
     seeded at the same states (model exploitation check).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from supermimic.data import MOTIONS_DIR
from supermimic.env import SuperMimicEnv
from supermimic.env_cfg import supermimic_env_cfg
from supermimic.rl.agent import AgentCfg
from supermimic.rl.setup import build, build_replay
from supermimic.state import NUM_JOINTS

BLOCKS = (
  ("height", 0, 1),
  ("root_quat", 1, 5),
  ("root_linvel", 5, 8),
  ("root_angvel", 8, 11),
  ("joint_pos", 11, 11 + NUM_JOINTS),
  ("joint_vel", 11 + NUM_JOINTS, 69),
  ("track_err", 69, 76),
)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--motion-dir", type=str, default=str(MOTIONS_DIR))
  ap.add_argument("--device", type=str, default="cuda:0")
  ap.add_argument("--collect", type=int, default=150)
  ap.add_argument("--n", type=int, default=1024)
  args = ap.parse_args()

  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True

  ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
  clips = ck.get("args", {}).get("clips")
  clips = clips.split(",") if isinstance(clips, str) and clips else None
  cfg = supermimic_env_cfg(args.motion_dir, args.n, clips=clips)
  cfg.observations["actor"].enable_corruption = False
  env = SuperMimicEnv(cfg, device=args.device)
  _, wm, agent = build(env, AgentCfg(sensor_noise=True))
  wm.dynamics.load_state_dict(ck["dynamics"])
  agent.encoder.load_state_dict(ck["encoder"])
  agent.decoder.load_state_dict(ck["decoder"])
  agent.critic.load_state_dict(ck["critic"])
  print(f"checkpoint iteration {ck.get('iteration')}", flush=True)

  cmd = env.sm_state.cmd
  replay = build_replay(env, args.collect + 8)
  xi_all = env.sm_state.xi()
  obs, _ = env.reset()
  with torch.no_grad():
    for _ in range(args.collect):
      a = agent.act(obs["actor"], obs["goal"])
      a = a + 0.1 * torch.randn_like(a)
      s_before = env.sm_state.state().clone()
      clip_b, frame_b = cmd.clip.clone(), cmd.frame.clone()
      obs, r, term, trunc, _ = env.step(a)
      replay.add(s_before, a, r, term, clip_b, frame_b, trunc)
  print(f"collected {replay.size} steps, term_rate={float(replay.d[:replay.size].float().mean()):.5f}", flush=True)

  k = wm.window
  t, e = replay.sample_windows(args.n, k)
  batch = replay.window(t, e, k)
  xi = xi_all[batch["env"]]
  clip, frame = batch["clip"], batch["frame"]
  horizon_err = []
  block_err = torch.zeros(len(BLOCKS), device=env.device)
  block_cnt = torch.zeros(len(BLOCKS), device=env.device)
  cal_pred = 0.0
  cal_label = 0.0
  with torch.no_grad():
    s_hat = batch["s"][:, 0]
    for j in range(k):
      adv = wm.library.advance_at(clip[:, j], frame[:, j])
      s_hat, logit = wm.step(s_hat, batch["a"][:, j], xi, adv)
      boundary = (batch["d"] | batch["trunc"])[:, j]
      err = ((s_hat - batch["s"][:, j + 1]) / wm.state_scale).abs()
      err = err[~boundary]
      horizon_err.append(float(err.mean()))
      for bi, (_, lo, hi) in enumerate(BLOCKS):
        block_err[bi] += err[:, lo:hi].sum()
        block_cnt[bi] += err[:, lo:hi].numel()
      p = torch.sigmoid(logit)
      lab = batch["d"][:, j].float()
      if j < k - 1:
        p, lab = p[~boundary], lab[~boundary]
      cal_pred += float(p.sum())
      cal_label += float(lab.sum())
  ntot = args.n * k
  print("HORIZON_ERR " + " ".join(f"{v:.4f}" for v in horizon_err), flush=True)
  for bi, (name, _, _) in enumerate(BLOCKS):
    print(f"BLOCK {name} {float(block_err[bi] / block_cnt[bi].clamp_min(1)):.4f}", flush=True)
  print(
    f"TERM_CAL pred={cal_pred / ntot:.6f} label={cal_label / ntot:.6f}",
    flush=True,
  )
  with torch.no_grad():
    _, logit0 = wm.dynamics(
      batch["s"][:, 0] / wm.state_scale,
      batch["a"][:, 0],
      xi / wm.xi_scale,
      wm.library.advance_at(clip[:, 0], frame[:, 0]) / wm.adv_scale,
    )
    p0 = torch.sigmoid(logit0)
    print(
      f"TERM_LOGGED pred={float(p0.mean()):.6f} "
      f"label={float(batch['d'][:, 0].float().mean()):.6f}",
      flush=True,
    )

  br = cmd.bin_return
  print(
    f"BINS n={br.numel()} mean={float(br.mean()):.3f} "
    f"p10={float(br.quantile(0.1)):.3f} p90={float(br.quantile(0.9)):.3f} "
    f"frac_near_max={float((br > 0.9 * cmd.max_return).float().mean()):.3f}",
    flush=True,
  )

  n = args.n
  t, e = replay.sample_states(n)
  xi = xi_all[e]
  s0 = replay.s[t, e]
  clip0 = replay.clip[t, e].long()
  frame0 = replay.frame[t, e].long()
  prev_a = replay.prev_action(t, e)
  hist = replay.obs_history(t, e, agent.history, lambda s, a, env: wm.obs_from_state(s, xi_all[env], a))
  hist = wm.corrupt(hist)
  with torch.no_grad():
    roll = agent.imagine(s0, hist, xi, clip0, frame0, prev_a)
  H = agent.cfg.horizon
  imag_surv = float(roll["c"][H].mean())
  imag_r = float((roll["c"][:H] * roll["r"][1 : H + 1]).sum(0).mean())
  print(f"IMAGINE H={H} survival={imag_surv:.4f} reward_sum={imag_r:.4f}", flush=True)

  ids = torch.arange(n, device=env.device)
  cmd.reset_to_frame(ids, clip0, frame0)
  env.sim.forward()
  env.observation_manager.reset(env_ids=ids)
  obs = env.observation_manager.compute(update_history=True, env_ids=ids)
  alive = torch.ones(n, dtype=torch.bool, device=env.device)
  total = torch.zeros(n, device=env.device)
  with torch.no_grad():
    for _ in range(H):
      a = agent.act(obs["actor"], obs["goal"])
      obs, r, term, trunc, _ = env.step(a)
      total += torch.where(alive, r, torch.zeros_like(r))
      alive &= ~(term | trunc)
  print(
    f"REAL H={H} survival={float(alive.float().mean()):.4f} "
    f"reward_sum={float(total.mean()):.4f}",
    flush=True,
  )
  env.close()


if __name__ == "__main__":
  main()
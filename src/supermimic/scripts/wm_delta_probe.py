"""Decisive WM-fit probe: is the dynamics network learning anything?

Compares the learned one-step delta to the actual state change and to the
zero-delta ("no change") baseline, per component block, then tracks the
one-step error and delta magnitude over extra offline WM updates.  If the WM
error equals the zero-delta baseline and |delta| ~ 0, the fit is broken; if
|delta| matches the change but error stays high, the direction is wrong.
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
  ap.add_argument("--n", type=int, default=1024)
  ap.add_argument("--collect", type=int, default=300)
  ap.add_argument("--updates", type=int, default=600)
  ap.add_argument("--eval-every", type=int, default=100)
  ap.add_argument("--lr", type=float, default=None)
  args = ap.parse_args()

  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True

  ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
  clips = ck.get("args", {}).get("clips")
  clips = clips.split(",") if isinstance(clips, str) and clips else None
  cfg = supermimic_env_cfg(args.motion_dir, args.n, clips=clips)
  env = SuperMimicEnv(cfg, device=args.device)
  _, wm, agent = build(env, AgentCfg())
  wm.dynamics.load_state_dict(ck["dynamics"])
  wm.opt.load_state_dict(ck["opt_wm"])
  agent.encoder.load_state_dict(ck["encoder"])
  agent.decoder.load_state_dict(ck["decoder"])
  if args.lr is not None:
    wm.set_lr(args.lr)
  print(f"checkpoint iteration {ck.get('iteration')} wm_lr={wm.opt.param_groups[0]['lr']:.2e}", flush=True)

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
  print(f"collected {replay.size} steps", flush=True)

  k = wm.window
  t, e = replay.sample_windows(2048, k)
  batch = replay.window(t, e, k)
  xi = xi_all[batch["env"]]
  clip, frame = batch["clip"], batch["frame"]
  scale = wm.state_scale

  @torch.no_grad()
  def measure():
    s0 = batch["s"][:, 0]
    # zero-delta baseline: state never changes
    base1 = ((s0 - batch["s"][:, 1]) / scale).abs().mean()
    base8 = ((s0 - batch["s"][:, k]) / scale).abs().mean()
    # learned rollout
    s_hat = s0
    delta0 = None
    for j in range(k):
      adv = wm.library.advance_at(clip[:, j], frame[:, j])
      s_prev = s_hat
      s_hat, _ = wm.step(s_hat, batch["a"][:, j], xi, adv)
      if j == 0:
        delta0 = (s_hat - s_prev) / scale
    adv0 = wm.library.advance_at(clip[:, 0], frame[:, 0])
    s1, _ = wm.step(s0, batch["a"][:, 0], xi, adv0)
    err1 = ((s1 - batch["s"][:, 1]) / scale).abs().mean()
    err8 = ((s_hat - batch["s"][:, k]) / scale).abs().mean()
    change0 = (batch["s"][:, 1] - s0) / scale
    return base1, base8, err1, err8, delta0, change0

  base1, base8, err1, err8, delta0, change0 = measure()
  print(f"BASELINE zero-delta 1step={base1:.4f} 8step={base8:.4f}", flush=True)
  print(f"WM       learned   1step={err1:.4f} 8step={err8:.4f}", flush=True)
  print("BLOCK delta_mag change_mag ratio (normalized, step0):", flush=True)
  for name, lo, hi in BLOCKS:
    dm = float(delta0[:, lo:hi].abs().mean())
    cm = float(change0[:, lo:hi].abs().mean())
    print(f"  {name:12s} |delta|={dm:.4f} |change|={cm:.4f} ratio={dm / max(cm, 1e-8):.3f}", flush=True)

  for u in range(1, args.updates + 1):
    wm.fit(replay, xi_all)
    if u % args.eval_every == 0 or u == args.updates:
      b1, b8, e1, e8, d0, c0 = measure()
      ratio = float(d0.abs().mean() / c0.abs().mean().clamp_min(1e-8))
      print(
        f"FIT u={u} 1step={e1:.4f} 8step={e8:.4f} delta/change={ratio:.3f} "
        f"grad_norm={wm.metrics['wm/grad_norm']:.1f}",
        flush=True,
      )
  env.close()


if __name__ == "__main__":
  main()
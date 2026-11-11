"""Probe whether the world model is underfit: fit it offline on fresh replay.

Collects a stationary replay with the checkpoint policy, then runs extra WM
updates (no policy changes) and tracks held-out 8-step error.  A large drop
means the coupled loop under-trains the model; a flat curve means the fit is
capacity/optimisation-limited.
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


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--motion-dir", type=str, default=str(MOTIONS_DIR))
  ap.add_argument("--device", type=str, default="cuda:0")
  ap.add_argument("--n", type=int, default=2048)
  ap.add_argument("--collect", type=int, default=1000)
  ap.add_argument("--updates", type=int, default=1000)
  ap.add_argument("--eval-every", type=int, default=100)
  ap.add_argument("--grad-clip", type=float, default=None)
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
  env.sm_state.cmd.bin_return.copy_(ck["bin_return"])
  env.sm_state.cmd.past_weight.copy_(ck["past_weight"])
  if args.grad_clip is not None:
    wm.grad_clip = args.grad_clip
  print(f"checkpoint iteration {ck.get('iteration')} grad_clip={wm.grad_clip}", flush=True)

  cmd = env.sm_state.cmd
  replay = build_replay(env, args.collect)
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

  @torch.no_grad()
  def held_out(n=2048):
    t, e = replay.sample_windows(n, k)
    b = replay.window(t, e, k)
    out = wm.prediction_error(
      b["s"][:, 0], b["a"], xi_all[b["env"]], b["clip"][:, :k], b["frame"][:, :k], b["s"][:, k]
    )
    return out["wm/pred_err_abs"], out["wm/pred_err_rmse"]

  err0 = held_out()
  print(f"PROBE update=0 abs={err0[0]:.5f} rmse={err0[1]:.5f}", flush=True)
  for u in range(1, args.updates + 1):
    m = wm.fit(replay, xi_all)
    if u % args.eval_every == 0 or u == args.updates:
      err = held_out()
      print(
        f"PROBE update={u} abs={err[0]:.5f} rmse={err[1]:.5f} "
        f"state_loss={m['wm/state_loss']:.3f} grad_norm={m['wm/grad_norm']:.1f}",
        flush=True,
      )
  env.close()


if __name__ == "__main__":
  main()
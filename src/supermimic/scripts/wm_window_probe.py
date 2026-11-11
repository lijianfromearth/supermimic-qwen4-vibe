"""Test whether the k-step autoregressive window length limits WM fit quality.

Fits the dynamics from the same checkpoint with training windows k=1,2,4,8 and
measures the held-out 1-step and 8-step error after each.  If short windows fit
much better, the long autoregressive BPTT (exploding Jacobians) is the bottleneck.
"""

from __future__ import annotations

import argparse
import copy
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
  ap.add_argument("--n", type=int, default=1024)
  ap.add_argument("--collect", type=int, default=300)
  ap.add_argument("--updates", type=int, default=800)
  ap.add_argument("--lr", type=float, default=2e-4)
  ap.add_argument("--windows", type=str, default="1,2,4,8")
  args = ap.parse_args()

  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True

  ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
  clips = ck.get("args", {}).get("clips")
  clips = clips.split(",") if isinstance(clips, str) and clips else None
  cfg = supermimic_env_cfg(args.motion_dir, args.n, clips=clips)
  env = SuperMimicEnv(cfg, device=args.device)
  _, wm, agent = build(env, AgentCfg())
  agent.encoder.load_state_dict(ck["encoder"])
  agent.decoder.load_state_dict(ck["decoder"])
  init_dyn = copy.deepcopy(ck["dynamics"])
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
  print(f"collected {replay.size} steps", flush=True)

  # fixed evaluation windows (8 steps), independent of the training window
  te, ee = replay.sample_windows(2048, 8)
  ev = replay.window(te, ee, 8)
  exi = xi_all[ev["env"]]
  eclip, eframe = ev["clip"], ev["frame"]
  scale = wm.state_scale

  @torch.no_grad()
  def eval_err():
    s0 = ev["s"][:, 0]
    adv0 = wm.library.advance_at(eclip[:, 0], eframe[:, 0])
    s1, _ = wm.step(s0, ev["a"][:, 0], exi, adv0)
    e1 = float(((s1 - ev["s"][:, 1]) / scale).abs().mean())
    s_hat = s0
    for j in range(8):
      adv = wm.library.advance_at(eclip[:, j], eframe[:, j])
      s_hat, _ = wm.step(s_hat, ev["a"][:, j], exi, adv)
    e8 = float(((s_hat - ev["s"][:, 8]) / scale).abs().mean())
    return e1, e8

  for tw in [int(x) for x in args.windows.split(",")]:
    wm.dynamics.load_state_dict(init_dyn)
    wm.opt = torch.optim.Adam(wm.dynamics.parameters(), lr=args.lr)
    wm.window = tw
    e1, e8 = eval_err()
    print(f"WINDOW k={tw} start 1step={e1:.4f} 8step={e8:.4f}", flush=True)
    for u in range(1, args.updates + 1):
      wm.fit(replay, xi_all)
      if u % 200 == 0 or u == args.updates:
        e1, e8 = eval_err()
        print(
          f"WINDOW k={tw} u={u} 1step={e1:.4f} 8step={e8:.4f} "
          f"gn={wm.metrics['wm/grad_norm']:.0f}",
          flush=True,
        )
  env.close()


if __name__ == "__main__":
  main()
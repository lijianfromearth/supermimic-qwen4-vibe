"""Sanity check: can the dynamics network overfit a tiny fixed batch?

Trains the WM repeatedly on the SAME 256 windows.  A 2048^3 MLP should drive the
error toward zero by memorisation.  If it plateaus high, the gradient path from
the loss to the dynamics output is broken; if it overfits, the fit problem is
data distribution / generalisation, not the optimisation graph.
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
  ap.add_argument("--n", type=int, default=512)
  ap.add_argument("--collect", type=int, default=120)
  ap.add_argument("--steps", type=int, default=1000)
  ap.add_argument("--batch", type=int, default=256)
  ap.add_argument("--lr", type=float, default=1e-3)
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
  agent.encoder.load_state_dict(ck["encoder"])
  agent.decoder.load_state_dict(ck["decoder"])
  wm.opt = torch.optim.Adam(wm.dynamics.parameters(), lr=args.lr)
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

  k = wm.window
  t, e = replay.sample_windows(args.batch, k)
  batch = replay.window(t, e, k)
  xi = xi_all[batch["env"]]
  scale = wm.state_scale

  @torch.no_grad()
  def err():
    s_hat = batch["s"][:, 0]
    e1 = e8 = None
    for j in range(k):
      adv = wm.library.advance_at(batch["clip"][:, j], batch["frame"][:, j])
      s_hat, _ = wm.step(s_hat, batch["a"][:, j], xi, adv)
      if j == 0:
        e1 = float(((s_hat - batch["s"][:, 1]) / scale).abs().mean())
    e8 = float(((s_hat - batch["s"][:, k]) / scale).abs().mean())
    return e1, e8

  e1, e8 = err()
  print(f"OVERFIT start 1step={e1:.4f} 8step={e8:.4f}", flush=True)
  wm.dynamics.train()
  for u in range(1, args.steps + 1):
    wm.opt.zero_grad(set_to_none=True)
    sl, tl = wm.window_loss(batch, xi)
    (sl + tl).backward()
    gn = torch.nn.utils.clip_grad_norm_(wm.dynamics.parameters(), 1.0)
    wm.opt.step()
    if u % 100 == 0 or u == args.steps:
      e1, e8 = err()
      print(
        f"OVERFIT u={u} 1step={e1:.4f} 8step={e8:.4f} state_loss={float(sl):.3f} gn={float(gn):.0f}",
        flush=True,
      )
  env.close()


if __name__ == "__main__":
  main()
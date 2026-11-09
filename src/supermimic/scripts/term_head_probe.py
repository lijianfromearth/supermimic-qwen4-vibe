"""Validate the separate-termination-head hypothesis offline.

The current head is the 77th output of the dynamics MLP, so its BCE gradient
shares the 2048^3 trunk with the ~3600x larger state loss and collapses to the
base rate (TERM_AT_FALL pred_p50=0.0014).  This probe trains small SEPARATE heads
on the same logged replay with unweighted BCE + Adam and checks whether they
recognize real fall states.  Two sampling regimes disentangle "shared trunk" from
"class imbalance": natural (paper) vs balanced positives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from supermimic.data import MOTIONS_DIR
from supermimic.env import SuperMimicEnv
from supermimic.env_cfg import supermimic_env_cfg
from supermimic.rl.agent import AgentCfg
from supermimic.rl.setup import build, build_replay
from supermimic.state import NUM_JOINTS, REF_ADV_DIM, STATE_DIM, XI_DIM


class Head(nn.Module):
  def __init__(self, in_dim: int, hidden=(256, 128)) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    prev = in_dim
    for w in hidden:
      layers += [nn.Linear(prev, w), nn.ELU()]
      prev = w
    layers += [nn.Linear(prev, 1)]
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x).squeeze(-1)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--motion-dir", type=str, default=str(MOTIONS_DIR))
  ap.add_argument("--device", type=str, default="cuda:0")
  ap.add_argument("--collect", type=int, default=400)
  ap.add_argument("--n", type=int, default=2048)
  ap.add_argument("--steps", type=int, default=4000)
  ap.add_argument("--batch", type=int, default=8192)
  args = ap.parse_args()

  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  dev = args.device

  ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
  clips = ck.get("args", {}).get("clips")
  clips = clips.split(",") if isinstance(clips, str) and clips else None
  cfg = supermimic_env_cfg(args.motion_dir, args.n, clips=clips)
  cfg.observations["actor"].enable_corruption = False
  env = SuperMimicEnv(cfg, device=dev)
  _, wm, agent = build(env, AgentCfg(sensor_noise=True))
  wm.dynamics.load_state_dict(ck["dynamics"])
  agent.encoder.load_state_dict(ck["encoder"])
  agent.decoder.load_state_dict(ck["decoder"])
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
  size = replay.size
  rate = float(replay.d[:size].float().mean())
  print(f"collected {size} steps x {args.n} envs, term_rate={rate:.5f}", flush=True)

  in_dim = STATE_DIM + NUM_JOINTS + XI_DIM + REF_ADV_DIM
  ss, xs, avs = wm.state_scale, wm.xi_scale, wm.adv_scale

  def gather(t, e):
    s = replay.s[t, e] / ss
    a = replay.a[t, e]
    xi = xi_all[e] / xs
    adv = wm.library.advance_at(replay.clip[t, e].long(), replay.frame[t, e].long()) / avs
    return torch.cat([s, a, xi, adv], dim=-1)

  # All valid slots and the subset that are real falls.
  all_t = torch.arange(size, device=dev).repeat_interleave(args.n)
  all_e = torch.arange(args.n, device=dev).repeat(size)
  dmask = replay.d[:size].reshape(-1).bool()
  fall_t, fall_e = all_t[dmask], all_e[dmask]
  nfall = fall_t.numel()
  perm = torch.randperm(nfall, device=dev)
  ntest = max(nfall // 4, 1)
  test_t, test_e = fall_t[perm[:ntest]], fall_e[perm[:ntest]]
  train_fall_t, train_fall_e = fall_t[perm[ntest:]], fall_e[perm[ntest:]]
  print(f"falls total={nfall} train={train_fall_t.numel()} test={ntest}", flush=True)

  x_test_fall = gather(test_t, test_e)
  nrm = torch.randint(0, all_t.numel(), (4096,), device=dev)
  x_test_norm = gather(all_t[nrm], all_e[nrm])

  def report(tag, head):
    with torch.no_grad():
      pf = torch.sigmoid(head(x_test_fall))
      pn = torch.sigmoid(head(x_test_norm))
    print(
      f"{tag} FALL pred_mean={float(pf.mean()):.4f} p50={float(pf.median()):.4f} "
      f"frac>0.5={float((pf > 0.5).float().mean()):.3f} | "
      f"NORM pred_mean={float(pn.mean()):.5f} frac>0.5={float((pn > 0.5).float().mean()):.4f}",
      flush=True,
    )

  # Current shared head, for reference.
  with torch.no_grad():
    s = replay.s[test_t, test_e] / ss
    a = replay.a[test_t, test_e]
    xi = xi_all[test_e] / xs
    adv = wm.library.advance_at(replay.clip[test_t, test_e].long(), replay.frame[test_t, test_e].long()) / avs
    _, logit = wm.dynamics(s, a, xi, adv)
    p = torch.sigmoid(logit)
  print(
    f"SHARED(current) FALL pred_mean={float(p.mean()):.4f} p50={float(p.median()):.4f} "
    f"frac>0.5={float((p > 0.5).float().mean()):.3f}",
    flush=True,
  )

  npos = train_fall_t.numel()
  nneg = all_t.numel() - nfall
  pos_weight = torch.tensor([min(nneg / max(npos, 1), 1000.0)], device=dev)

  for tag, balanced, pw in (
    ("SEP_natural", False, None),
    ("SEP_balanced", True, None),
    ("SEP_posweight", False, pos_weight),
  ):
    head = Head(in_dim).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=2e-4)
    head.train()
    for it in range(args.steps):
      if balanced:
        k = args.batch // 2
        fi = torch.randint(0, npos, (k,), device=dev)
        ni = torch.randint(0, all_t.numel(), (args.batch - k,), device=dev)
        t = torch.cat([train_fall_t[fi], all_t[ni]])
        e = torch.cat([train_fall_e[fi], all_e[ni]])
      else:
        idx = torch.randint(0, all_t.numel(), (args.batch,), device=dev)
        t, e = all_t[idx], all_e[idx]
      x = gather(t, e)
      y = replay.d[t, e].float()
      logit = head(x)
      loss = nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)
      opt.zero_grad(set_to_none=True)
      loss.backward()
      opt.step()
      if (it + 1) % 1000 == 0:
        print(f"  {tag} step={it+1} loss={float(loss.detach()):.5f}", flush=True)
    head.eval()
    report(tag, head)

  env.close()


if __name__ == "__main__":
  main()
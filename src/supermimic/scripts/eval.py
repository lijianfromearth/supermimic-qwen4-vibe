"""Checkpoint evaluation (paper F.1).

Every motion plays once from frame 0 with foot friction 1.0, no domain
randomization, no observation noise and deterministic physics.  A motion fails
if the training termination triggers.  The reported score is the normalized
return: the undiscounted sum of per-step rewards divided by the largest value
that sum can take, so an early failure scores low.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic.data import MOTIONS_DIR
from supermimic.env import SuperMimicEnv
from supermimic.env_cfg import supermimic_env_cfg
from supermimic.rl.agent import AgentCfg
from supermimic.rl.setup import build

DEFAULT_MOTIONS = str(MOTIONS_DIR)


@torch.no_grad()
def evaluate(env: SuperMimicEnv, agent, out_path: Path | None = None) -> dict:
  cmd = env.sm_state.cmd
  lib = cmd.library
  math = env.tracking_math
  n = env.num_envs
  assert n == lib.num_clips, f"eval needs one env per clip, got {n} vs {lib.num_clips}"
  max_r = math.max_reward_per_step

  obs, _ = env.reset()
  lengths = lib.lengths[cmd.clip]
  denom = max_r * (lengths - 1).clamp_min(1).float()
  total = torch.zeros(n, device=env.device)
  steps = torch.zeros(n, dtype=torch.long, device=env.device)
  alive = torch.ones(n, dtype=torch.bool, device=env.device)
  failed = torch.zeros(n, dtype=torch.bool, device=env.device)
  frames = torch.zeros(n, dtype=torch.long, device=env.device)

  while bool(alive.any()):
    a = agent.act(obs["actor"], obs["goal"])
    obs, r, term, trunc, _ = env.step(a)
    total += torch.where(alive, r, torch.zeros_like(r))
    steps += alive.long()
    frames = torch.where(alive, cmd.frame, frames)
    newly = alive & (term | trunc)
    failed |= alive & term
    alive &= ~(term | trunc)
    if int(newly.sum()) and int(alive.sum()) == 0:
      break

  norm = total / denom
  score = float(norm.mean())
  per_clip = [
    {
      "clip": lib.clip_names[i],
      "normalized_return": float(norm[i]),
      "failed": bool(failed[i]),
      "steps": int(steps[i]),
      "frames": int(lengths[i]),
    }
    for i in range(n)
  ]
  result = {
    "normalized_return": score,
    "failed_motions": int(failed.sum()),
    "num_clips": n,
    "clips": per_clip,
  }
  if out_path is not None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
  return result


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--motion-dir", type=str, default=DEFAULT_MOTIONS)
  ap.add_argument("--device", type=str, default="cuda:0")
  ap.add_argument("--out", type=Path, default=None)
  ap.add_argument("--horizon", type=int, default=32)
  args = ap.parse_args()

  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True

  ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
  train_args = ck.get("args", {})
  clips = train_args.get("clips")
  clips = clips.split(",") if isinstance(clips, str) and clips else None

  names = sorted(p.stem for p in Path(args.motion_dir).glob("*.npz"))
  if clips is not None:
    names = [n for n in names if n in set(clips)]
  cfg = supermimic_env_cfg(args.motion_dir, len(names), clips=clips, eval_mode=True)
  cfg.commands["motion"].eval_clips = list(range(len(names)))
  env = SuperMimicEnv(cfg, device=args.device)

  _, wm, agent = build(env, AgentCfg(horizon=args.horizon))
  agent.encoder.load_state_dict(ck["encoder"])
  agent.decoder.load_state_dict(ck["decoder"])
  wm.dynamics.load_state_dict(ck["dynamics"])
  wm.term.load_state_dict(ck["term"])
  agent.critic.load_state_dict(ck["critic"])

  out = args.out
  if out is None:
    out = args.checkpoint.with_suffix(".eval.json")
  result = evaluate(env, agent, out)
  print(
    f"iteration={ck.get('iteration')} normalized_return={result['normalized_return']:.4f} "
    f"failed={result['failed_motions']}/{result['num_clips']}",
    flush=True,
  )
  worst = sorted(result["clips"], key=lambda c: c["normalized_return"])[:10]
  for c in worst:
    print(
      f"  {c['clip']:32s} ret={c['normalized_return']:.4f} "
      f"steps={c['steps']}/{c['frames']} failed={c['failed']}",
      flush=True,
    )
  print(f"wrote {out}", flush=True)
  env.close()


if __name__ == "__main__":
  main()

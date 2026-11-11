"""SuperMimic training loop (paper Algorithm A.1, Table A.5).

One iteration collects one vectorised simulator step with 4096 environments and
pushes it to the FIFO replay.  The update-to-data ratio of 0.4 schedules two
update cycles per five iterations; a cycle is one world-model fit (Eq. 1) plus
one policy improvement (Eq. 2) and critic fit (Eq. 3).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from supermimic.data import MOTIONS_DIR, RUNS_DIR
from supermimic.env import SuperMimicEnv
from supermimic.env_cfg import supermimic_env_cfg
from supermimic.rl.agent import AgentCfg
from supermimic.rl.setup import build, build_replay

DEFAULT_MOTIONS = str(MOTIONS_DIR)


def lr_at(it: int, base: float, decay_iters: int, floor: float) -> float:
  if it >= decay_iters:
    return base * floor
  cos = 0.5 * (1.0 + math.cos(math.pi * it / decay_iters))
  return base * (floor + (1.0 - floor) * cos)


def save_checkpoint(path: Path, env, wm, agent, replay, it: int, args) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "iteration": it,
      "args": vars(args),
      "dynamics": wm.dynamics.state_dict(),
      "term": wm.term.state_dict(),
      "encoder": agent.encoder.state_dict(),
      "decoder": agent.decoder.state_dict(),
      "critic": agent.critic.state_dict(),
      "opt_wm": wm.opt.state_dict(),
      "opt_term": wm.opt_term.state_dict(),
      "opt_theta": agent.opt_theta.state_dict(),
      "opt_psi": agent.opt_psi.state_dict(),
      "scales": agent.scales.state_dict(),
      "replay_size": replay.size,
      "replay_ptr": replay.ptr,
      "bin_return": env.sm_state.cmd.bin_return,
      "past_weight": env.sm_state.cmd.past_weight,
    },
    path,
  )


def load_checkpoint(path: Path, wm, agent, env, replay) -> int:
  ck = torch.load(path, map_location=agent.device, weights_only=False)
  wm.dynamics.load_state_dict(ck["dynamics"])
  wm.term.load_state_dict(ck["term"])
  agent.encoder.load_state_dict(ck["encoder"])
  agent.decoder.load_state_dict(ck["decoder"])
  agent.critic.load_state_dict(ck["critic"])
  wm.opt.load_state_dict(ck["opt_wm"])
  wm.opt_term.load_state_dict(ck["opt_term"])
  agent.opt_theta.load_state_dict(ck["opt_theta"])
  agent.opt_psi.load_state_dict(ck["opt_psi"])
  env.sm_state.cmd.bin_return.copy_(ck["bin_return"])
  env.sm_state.cmd.past_weight.copy_(ck["past_weight"])
  return int(ck["iteration"])


@torch.no_grad()
def online_prediction_error(replay, wm, xi_all, n_env: int = 512) -> dict[str, float]:
  """Figure 3(c): state-only k-step error on the freshest policy rollouts."""
  k = wm.window
  if replay.size <= k:
    return {}
  t0 = (replay.ptr - k - 1) % replay.capacity
  if t0 + k >= replay.capacity:
    return {}
  n = min(n_env, replay.num_envs)
  e = torch.randperm(replay.num_envs, device=replay.device)[:n]
  t = torch.full((n,), t0, dtype=torch.long, device=replay.device)
  idx = t0 + torch.arange(k + 1, device=replay.device)
  ok = ~replay.done[idx[:k], e[:, None]].any(dim=1)
  if int(ok.sum()) < 8:
    return {}
  batch = replay.window(t[ok], e[ok], k)
  return wm.prediction_error(
    batch["s"][:, 0],
    batch["a"],
    xi_all[batch["env"]],
    batch["clip"][:, :k],
    batch["frame"][:, :k],
    batch["s"][:, k],
  )


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--motion-dir", type=str, default=DEFAULT_MOTIONS)
  ap.add_argument("--out-dir", type=Path, default=RUNS_DIR / "sm")
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--iterations", type=int, default=320000)
  ap.add_argument("--device", type=str, default="cuda:0")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--utd", type=float, default=0.4)
  ap.add_argument("--horizon", type=int, default=32)
  ap.add_argument("--rollouts", type=int, default=4096)
  ap.add_argument("--replay-steps", type=int, default=1000)
  ap.add_argument("--lr", type=float, default=2e-4)
  ap.add_argument("--lr-decay-iters", type=int, default=250000)
  ap.add_argument("--lr-floor", type=float, default=0.03)
  ap.add_argument("--wm-chunks", type=int, default=10)
  ap.add_argument("--wm-chunk-size", type=int, default=2048)
  ap.add_argument("--window", type=int, default=8)
  ap.add_argument("--log-every", type=int, default=50)
  ap.add_argument("--ckpt-every", type=int, default=4000)
  ap.add_argument("--pred-every", type=int, default=50)
  ap.add_argument("--action-noise", type=float, default=0.1)
  ap.add_argument(
    "--decoder-gain", type=float, default=0.01,
    help="orthogonal init gain of the policy decoder head (exploration scale)",
  )
  ap.add_argument(
    "--no-noise", action="store_true", help="disable imagination sensor noise"
  )
  ap.add_argument(
    "--no-recompute", action="store_true", help="disable gradient checkpointing"
  )
  ap.add_argument(
    "--no-survival-grad",
    action="store_true",
    help="ablation: treat the survival weight as a constant in the actor objective",
  )
  ap.add_argument(
    "--compile",
    action="store_true",
    help="torch.compile the WM window loss and the imagination step (~1.6x faster)",
  )
  ap.add_argument("--resume", type=Path, default=None)
  ap.add_argument("--clips", type=str, default=None, help="comma-separated clip subset")
  args = ap.parse_args()

  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.manual_seed(args.seed)

  args.out_dir.mkdir(parents=True, exist_ok=True)
  clips = args.clips.split(",") if args.clips else None
  cfg = supermimic_env_cfg(args.motion_dir, args.num_envs, clips=clips)
  env = SuperMimicEnv(cfg, device=args.device)
  env.seed(args.seed)
  obs, _ = env.reset()

  agent_cfg = AgentCfg(
    horizon=args.horizon,
    lr=args.lr,
    rollouts=args.rollouts,
    sensor_noise=not args.no_noise,
    recompute=not args.no_recompute,
    survival_grad=not args.no_survival_grad,
    decoder_gain=args.decoder_gain,
  )
  scales, wm, agent = build(
    env,
    agent_cfg,
    {
      "lr": args.lr,
      "window": args.window,
      "chunks": args.wm_chunks,
      "chunk_size": args.wm_chunk_size,
    },
  )
  replay = build_replay(env, args.replay_steps)
  if args.compile:
    # Both call sites have static shapes (chunk_size x window, rollouts x H), so
    # dynamic=False keeps a single graph.  CUDA-graph modes are incompatible with
    # the activation checkpointing that H=32 needs on a 16 GB device.
    wm.window_loss = torch.compile(wm.window_loss, dynamic=False)
    agent._imagine_step = torch.compile(agent._imagine_step, dynamic=False)
  cmd = env.sm_state.cmd
  lib = cmd.library
  max_r = env.tracking_math.max_reward_per_step
  print(
    f"envs={env.num_envs} clips={lib.num_clips} frames={lib.total_frames} "
    f"obs={obs['actor'].shape[1]} goal={obs['goal'].shape[1]} max_r={max_r:.4f}",
    flush=True,
  )

  start_it = 0
  if args.resume is not None:
    start_it = load_checkpoint(args.resume, wm, agent, env, replay)
    print(f"resumed at iteration {start_it} (replay not restored)", flush=True)

  log_path = args.out_dir / "log.jsonl"
  log_f = open(log_path, "a", buffering=1)
  xi_all = env.sm_state.xi()
  ep_sum = torch.zeros(env.num_envs, device=env.device)
  remaining = (lib.lengths[cmd.clip] - 1 - cmd.frame).clamp_min(1).float()
  ep_max = max_r * remaining
  returns: list[float] = []
  credit = 0.0
  updates = 0
  t_start = time.time()
  t_collect = 0.0
  t_update = 0.0
  checked = False

  for it in range(start_it, args.iterations):
    lr = lr_at(it, args.lr, args.lr_decay_iters, args.lr_floor)
    wm.set_lr(lr)
    agent.set_lr(lr)

    t0 = time.time()
    with torch.no_grad():
      a = agent.act(obs["actor"], obs["goal"])
      a = a + args.action_noise * torch.randn_like(a)
      s_before = env.sm_state.state().clone()
      clip_before = cmd.clip.clone()
      frame_before = cmd.frame.clone()
    obs, r, term, trunc, _ = env.step(a)
    torch.cuda.synchronize()
    t_collect += time.time() - t0

    done = term | trunc
    if not checked:
      with torch.no_grad():
        keep = ~done
        s_post = env.sm_state.state()
        robot, ref = wm.robot_and_ref(s_post, cmd.clip, cmd.frame)
        act = env.sm_state.action()
        r_hat = wm.reward(robot, ref, env.sm_state.prev_action(), act)
        o_hat = wm.obs_from_state(s_post, xi_all, act)
        g_hat = wm.goal(robot, s_post, cmd.clip, cmd.frame)
        d_hat = wm.math.terminated(robot, ref)
        errs = {
          "reward": float((r_hat[keep] - r[keep]).abs().max()),
          "proprio": float((o_hat - env.sm_state.observation()).abs().max()),
          "goal": float((g_hat - obs["goal"]).abs().max()),
          "termination": float((d_hat[keep] != term[keep]).float().mean()),
        }
      print("imagination heads vs simulator:", errs, flush=True)
      assert errs["reward"] < 1e-5 and errs["proprio"] < 1e-5 and errs["goal"] < 1e-5
      assert errs["termination"] == 0.0
      checked = True

    replay.add(s_before, a, r, term, clip_before, frame_before, trunc)
    cmd.update_bin_returns(r, term, clip_before, frame_before)
    xi_all = env.sm_state.xi()

    ep_sum += r
    if bool(done.any()):
      norm = ep_sum[done] / ep_max[done].clamp_min(1e-8)
      returns.extend(norm.tolist())
      ep_sum[done] = 0.0
    remaining = (lib.lengths[cmd.clip] - 1 - cmd.frame).clamp_min(1).float()
    ep_max = torch.where(done, max_r * remaining, ep_max)

    credit += args.utd
    n_cycles = 0
    while credit >= 1.0 and replay.size > args.window:
      credit -= 1.0
      n_cycles += 1
    if n_cycles:
      t0 = time.time()
      for _ in range(n_cycles):
        wm_metrics = wm.fit(replay, xi_all)
        torch.cuda.synchronize()
        t_wm = time.time() - t0
        t0 = time.time()
        ag_metrics = agent.update(replay, xi_all)
        torch.cuda.synchronize()
        t_agent = time.time() - t0
        updates += 1
        t0 = time.time()
      t_update += t_wm + t_agent
      wm_metrics["wm/ms"] = t_wm * 1e3
      ag_metrics["agent/ms"] = t_agent * 1e3
    else:
      wm_metrics, ag_metrics = {}, {}

    if (it + 1) % args.pred_every == 0:
      t0 = time.time()
      pred = online_prediction_error(replay, wm, xi_all)
      torch.cuda.synchronize()
      t_update += time.time() - t0
    else:
      pred = {}

    if (it + 1) % args.log_every == 0 or it == start_it:
      elapsed = time.time() - t_start
      recent = returns[-4096:]
      train_return = sum(recent) / len(recent) if recent else 0.0
      rec = {
        "iteration": it + 1,
        "updates": updates,
        "lr": lr,
        "hours": elapsed / 3600,
        "ms_per_iter": elapsed / max(it + 1 - start_it, 1) * 1e3,
        "collect_frac": t_collect / max(elapsed, 1e-9),
        "update_frac": t_update / max(elapsed, 1e-9),
        "train_return": train_return,
        "episodes": len(returns),
        "reward_mean": float(r.mean()),
        "term_rate": float(term.float().mean()),
        "trunc_rate": float(trunc.float().mean()),
        "replay": replay.size,
        "bin_return_mean": float(cmd.bin_return.mean()),
        "gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        **wm_metrics,
        **ag_metrics,
        **pred,
      }
      log_f.write(json.dumps(rec) + "\n")
      print(
        f"it={it + 1:7d} upd={updates:7d} ret={train_return:.4f} "
        f"r={rec['reward_mean']:.4f} term={rec['term_rate']:.3f} "
        f"ms={rec['ms_per_iter']:.1f} h={rec['hours']:.3f} "
        f"wm={wm_metrics.get('wm/state_loss', float('nan')):.4f} "
        f"obj={ag_metrics.get('agent/objective', float('nan')):.4f} "
        f"pred={pred.get('wm/pred_err_abs', float('nan')):.5f} "
        f"gpu={rec['gpu_gib']:.2f}",
        flush=True,
      )
      returns = returns[-8192:]

    if (it + 1) % args.ckpt_every == 0:
      save_checkpoint(
        args.out_dir / f"ckpt_{it + 1:07d}.pt", env, wm, agent, replay, it + 1, args
      )
      save_checkpoint(args.out_dir / "latest.pt", env, wm, agent, replay, it + 1, args)

  save_checkpoint(
    args.out_dir / "final.pt", env, wm, agent, replay, args.iterations, args
  )
  log_f.close()
  env.close()


if __name__ == "__main__":
  main()

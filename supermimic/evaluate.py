"""Evaluation protocols for SuperMimic on LAFAN1 (Appendix F.1 & Figure 4a)."""

from __future__ import annotations

import time
import torch

from supermimic.env import G1TrackingEnv
from supermimic.models import MotionPolicy


@torch.no_grad()
def evaluate_full_clips(
  eval_env: G1TrackingEnv,
  policy: MotionPolicy,
) -> dict[str, float]:
  """Evaluate policy deterministically on all 40 LAFAN1 clips from frame 0 (Appendix F.1).
  eval_env must have num_envs >= num_clips (40) and play_mode=True.
  """
  policy.eval()
  motion_lib = eval_env.motion_lib
  num_clips = motion_lib.num_clips
  assert eval_env.num_envs >= num_clips, f"eval_env needs >= {num_clips} envs"

  all_ids = torch.arange(eval_env.num_envs, dtype=torch.long, device=eval_env.device)
  clip_ids = torch.arange(eval_env.num_envs, dtype=torch.long, device=eval_env.device) % num_clips
  zero_frames = torch.zeros(eval_env.num_envs, dtype=torch.long, device=eval_env.device)

  eval_env.play_mode = True
  eval_env.reset_envs(all_ids, clip_ids=clip_ids, local_frames=zero_frames)

  clip_max_steps = (motion_lib.clip_lengths[:num_clips] - 1).clone()  # (40,)
  max_total_steps = int(clip_max_steps.max().item())

  alive = torch.ones(num_clips, dtype=torch.bool, device=eval_env.device)
  failed = torch.zeros(num_clips, dtype=torch.bool, device=eval_env.device)
  cum_reward = torch.zeros(num_clips, dtype=torch.float32, device=eval_env.device)
  surv_steps = torch.zeros(num_clips, dtype=torch.long, device=eval_env.device)

  t0 = time.perf_counter()
  for step in range(max_total_steps):
    active_clips = (step < clip_max_steps) & alive
    if not torch.any(active_clips):
      break

    obs_flat, goal_seq = eval_env.get_policy_inputs()
    actions = policy(obs_flat, goal_seq)
    _, reward, terminated, _ = eval_env.step(actions, auto_reset=False)

    r_40 = reward[:num_clips]
    term_40 = terminated[:num_clips]

    cum_reward = torch.where(active_clips, cum_reward + r_40, cum_reward)
    surv_steps = torch.where(active_clips, surv_steps + 1, surv_steps)

    # Check if termination triggered before the last step of the clip
    newly_failed = active_clips & term_40 & ((step + 1) < clip_max_steps)
    failed = failed | newly_failed
    alive = alive & (~newly_failed)

  # Max possible per-step reward is 5.0 * dt = 0.10
  max_possible_return = 5.0 * eval_env.step_dt * clip_max_steps.float()
  norm_returns = cum_reward / max_possible_return

  mean_norm_return = float(norm_returns.mean().item())
  weighted_norm_return = float((cum_reward.sum() / max_possible_return.sum()).item())
  num_failed = int(failed.sum().item())
  mean_surv_frac = float((surv_steps.float() / clip_max_steps.float()).mean().item())
  dt = time.perf_counter() - t0

  policy.train()
  return {
    "eval/normalized_return": mean_norm_return,
    "eval/weighted_normalized_return": weighted_norm_return,
    "eval/failed_clips": num_failed,
    "eval/survival_fraction": mean_surv_frac,
    "eval/time_s": dt,
  }


@torch.no_grad()
def evaluate_Window_segments(
  env: G1TrackingEnv,
  policy: MotionPolicy,
  segment_steps: int = 500,
) -> dict[str, float]:
  """Fast parallel evaluation over all 10s (500-step) segments across the 40 clips using the training env.
  Restores the training env's play_mode=False and resets all envs afterward.
  """
  policy.eval()
  motion_lib = env.motion_lib
  device = env.device

  # Construct deterministic grid of 500-step segments across all 40 clips
  seg_clips = []
  seg_starts = []
  for c in range(motion_lib.num_clips):
    length = int(motion_lib.clip_lengths[c].item())
    for s in range(0, max(length - segment_steps, 1), segment_steps):
      seg_clips.append(c)
      seg_starts.append(s)

  num_segs = len(seg_clips)
  seg_clips_t = torch.tensor(seg_clips, dtype=torch.long, device=device)
  seg_starts_t = torch.tensor(seg_starts, dtype=torch.long, device=device)

  # Tile across all num_envs (4096)
  env_ids = torch.arange(env.num_envs, dtype=torch.long, device=device)
  tiled_clips = seg_clips_t[env_ids % num_segs]
  tiled_starts = seg_starts_t[env_ids % num_segs]

  snap = env.save_snapshot()
  env.play_mode = True
  env.max_episode_length = int(1e9)
  env.reset_envs(env_ids, clip_ids=tiled_clips, local_frames=tiled_starts)

  alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
  cum_reward = torch.zeros(env.num_envs, dtype=torch.float32, device=device)

  for _ in range(segment_steps):
    if not torch.any(alive):
      break
    obs_flat, goal_seq = env.get_policy_inputs()
    actions = policy(obs_flat, goal_seq)
    _, reward, terminated, timed_out = env.step(actions, auto_reset=False)
    cum_reward = torch.where(alive, cum_reward + reward, cum_reward)
    alive = alive & (~terminated) & (~timed_out)

  max_ret = 5.0 * env.step_dt * segment_steps
  norm_rets = (cum_reward[:num_segs] / max_ret)
  failed_segs = (~alive[:num_segs]).sum().item()

  # Restore exact training env state without aborting ongoing episodes or corrupting replay buffer
  env.load_snapshot(snap)
  policy.train()

  return {
    "eval_seg/normalized_return": float(norm_rets.mean().item()),
    "eval_seg/failed_segments": int(failed_segs),
    "eval_seg/total_segments": num_segs,
  }

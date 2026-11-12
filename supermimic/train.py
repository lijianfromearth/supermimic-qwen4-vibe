"""Main training loop for SuperMimic (Algorithm A.1 & Table A.5)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from supermimic.buffer import ReplayBuffer
from supermimic.data import MotionLibrary
from supermimic.env import G1TrackingEnv
from supermimic.evaluate import evaluate_Window_segments, evaluate_full_clips
from supermimic.models import Critic, MotionPolicy, RunningNormalizer, WorldModel
from supermimic.state import (
  OBS_FRAME_DIM,
  RobotState,
  compute_clean_obs,
  compute_state_loss,
  compute_target_delta,
  compute_tracking_reward,
  sample_obs_noise,
  state_to_canonical,
)


def cosine_lr(step: int, base_lr: float = 2e-4, decay_steps: int = 250_000, min_ratio: float = 0.03) -> float:
  """Cosine learning rate schedule to 0.03x over 250,000 iterations, then constant (Table A.5)."""
  if step >= decay_steps:
    return base_lr * min_ratio
  progress = float(step) / float(decay_steps)
  cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
  return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
  for param_group in optimizer.param_groups:
    param_group["lr"] = lr


@torch.no_grad()
def update_ema_policy(ema_policy: MotionPolicy, policy: MotionPolicy, decay: float) -> None:
  """Update Polyak EMA shadow policy parameters and copy running normalizer buffers."""
  for p_ema, p in zip(ema_policy.parameters(), policy.parameters(), strict=True):
    p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
  for b_ema, b in zip(ema_policy.buffers(), policy.buffers(), strict=True):
    b_ema.copy_(b)


@torch.no_grad()
def average_gradients(model: nn.Module, world_size: int) -> None:
  """Average parameter gradients across GPUs in multi-GPU data-parallel mode (Table A.5)."""
  if world_size <= 1:
    return
  grads = [p.grad for p in model.parameters() if p.grad is not None]
  if not grads:
    return
  flat = torch.cat([g.reshape(-1) for g in grads])
  dist.all_reduce(flat, op=dist.ReduceOp.AVG)
  offset = 0
  for g in grads:
    numel = g.numel()
    g.copy_(flat[offset : offset + numel].view_as(g))
    offset += numel


@torch.no_grad()
def sync_normalizers_and_bins(
  norms: list[RunningNormalizer], motion_lib: MotionLibrary, world_size: int
) -> None:
  """Synchronize running normalizer statistics and bin returns across GPUs."""
  if world_size <= 1:
    return
  for n in norms:
    dist.all_reduce(n.mean, op=dist.ReduceOp.AVG)
    dist.all_reduce(n.var, op=dist.ReduceOp.AVG)
    dist.all_reduce(n.std, op=dist.ReduceOp.AVG)
    dist.all_reduce(n.count, op=dist.ReduceOp.AVG)
  dist.all_reduce(motion_lib.bin_returns, op=dist.ReduceOp.AVG)
  dist.all_reduce(motion_lib.bin_past_softmax, op=dist.ReduceOp.AVG)


def _slice_robot_state(state: RobotState, sl: slice) -> RobotState:
  return RobotState(
    body_pos_w=state.body_pos_w[sl],
    body_rot_w=state.body_rot_w[sl],
    body_lin_vel_w=state.body_lin_vel_w[sl],
    body_ang_vel_w=state.body_ang_vel_w[sl],
    joint_pos=state.joint_pos[sl],
    joint_vel=state.joint_vel[sl],
  )


def train_world_model_step(
  world_model: WorldModel,
  wm_optimizer: torch.optim.Optimizer,
  buffer: ReplayBuffer,
  motion_lib: MotionLibrary,
  default_joint_pos: torch.Tensor,
  num_batches: int = 10,
  batch_size: int = 2048,
  window_k: int = 8,
  world_size: int = 1,
) -> dict[str, float]:
  """Fit world model f_omega and d_omega on k=8 autoregressive replay windows (Eq. 1 & Table A.5)."""
  world_model.train()
  device = default_joint_pos.device
  total_state_loss = torch.zeros((), device=device)
  total_delta_loss = torch.zeros((), device=device)
  total_bce_loss = torch.zeros((), device=device)
  total_8step_err = torch.zeros((), device=device)

  for _ in range(num_batches):
    batch = buffer.sample_windows(batch_size=batch_size, k=window_k)
    curr_state: RobotState = batch["s0"]
    actions: torch.Tensor = batch["actions"]          # (B, k, 29)
    sim_ctx: torch.Tensor = batch["sim_context"]      # (B, 33)
    target_states: list[RobotState] = batch["target_states"]
    target_terms: torch.Tensor = batch["target_terms"]  # (B, k)
    valid_mask: torch.Tensor = batch["valid_mask"]      # (B, k)
    clip_ids: torch.Tensor = batch["clip_ids"]          # (B, k)
    next_ts: torch.Tensor = batch["next_time_steps"]    # (B, k)

    loss_state_acc = torch.zeros((), device=device)
    loss_delta_acc = torch.zeros((), device=device)
    loss_bce_acc = torch.zeros((), device=device)
    err_acc = torch.zeros((), device=device)
    valid_steps_count = valid_mask.sum().clamp(min=1.0)

    s_tilde, cos_yaw, sin_yaw = state_to_canonical(curr_state, default_joint_pos)
    for j in range(window_k):
      act_j = actions[:, j]
      tgt_state_j = target_states[j]
      mask_j = valid_mask[:, j]

      pred_next_state, pred_delta = world_model.predict_next_state(
        curr_state, act_j, sim_ctx, default_joint_pos, s_tilde, cos_yaw, sin_yaw
      )

      # Target delta from current predicted state to ground-truth next state
      with torch.no_grad():
        tgt_delta = compute_target_delta(
          curr_state, s_tilde, cos_yaw, sin_yaw, tgt_state_j, default_joint_pos
        )
        norm_tgt_delta = world_model.delta_norm.normalize(tgt_delta)
      norm_pred_delta = (pred_delta - world_model.delta_norm.mean) / world_model.delta_norm.std
      delta_mse = torch.mean(torch.square(norm_pred_delta - norm_tgt_delta), dim=-1)

      s_loss, norm_err = compute_state_loss(pred_next_state, tgt_state_j)

      # Termination prediction d_hat_j
      flat_ref_idx = motion_lib.get_global_indices(clip_ids[:, j], next_ts[:, j])
      ref_rpos = motion_lib.root_pos_w[flat_ref_idx]
      ref_rrot = motion_lib.root_rot_w[flat_ref_idx]
      next_s_tilde, next_cos_yaw, next_sin_yaw = state_to_canonical(pred_next_state, default_joint_pos)
      term_logit = world_model.predict_termination_logit(
        pred_next_state, ref_rpos, ref_rrot, default_joint_pos, s_tilde=next_s_tilde
      )
      bce_j = F.binary_cross_entropy_with_logits(term_logit, target_terms[:, j], reduction="none")

      loss_state_acc = loss_state_acc + (s_loss * mask_j).sum()
      loss_delta_acc = loss_delta_acc + (delta_mse * mask_j).sum()
      loss_bce_acc = loss_bce_acc + (bce_j * mask_j).sum()
      err_acc = err_acc + (norm_err.detach() * mask_j).sum()

      curr_state = pred_next_state
      s_tilde, cos_yaw, sin_yaw = next_s_tilde, next_cos_yaw, next_sin_yaw

    loss_state_mean = loss_state_acc / valid_steps_count
    loss_delta_mean = loss_delta_acc / valid_steps_count
    loss_bce_mean = loss_bce_acc / valid_steps_count

    total_loss = loss_state_mean + 0.5 * loss_delta_mean + 0.1 * loss_bce_mean

    wm_optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    average_gradients(world_model, world_size)
    torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=1.0)
    wm_optimizer.step()

    total_state_loss += loss_state_mean.detach()
    total_delta_loss += loss_delta_mean.detach()
    total_bce_loss += loss_bce_mean.detach()
    total_8step_err += (err_acc / valid_steps_count).detach()

  inv_n = 1.0 / num_batches
  return {
    "wm/state_loss": float(total_state_loss.item()) * inv_n,
    "wm/delta_loss": float(total_delta_loss.item()) * inv_n,
    "wm/bce_loss": float(total_bce_loss.item()) * inv_n,
    "wm/8step_pred_error": float(total_8step_err.item()) * inv_n,
  }


def train_policy_and_critic_step(
  world_model: WorldModel,
  policy: MotionPolicy,
  critic: Critic,
  policy_optimizer: torch.optim.Optimizer,
  critic_optimizer: torch.optim.Optimizer,
  buffer: ReplayBuffer,
  motion_lib: MotionLibrary,
  default_joint_pos: torch.Tensor,
  soft_joint_pos_limits: torch.Tensor,
  batch_size: int = 4096,
  microbatch_size: int = 1024,
  horizon: int = 32,
  gamma: float = 0.99,
  gae_lambda: float = 0.95,
  critic_epochs: int = 5,
  critic_minibatches: int = 4,
  step_dt: float = 0.02,
  world_size: int = 1,
  safety_margin_z: float = 0.0,
  safety_margin_tilt: float = 0.0,
) -> dict[str, float]:
  """Improve policy pi_theta via H=32 step BPTT (Eq. 2) and fit critic V_psi to GAE targets (Eq. 3)."""
  starts = buffer.sample_imagination_starts(batch_size=batch_size)
  s0_full: RobotState = starts["s0"]
  sim_ctx_full: torch.Tensor = starts["sim_context"]            # (B, 33)
  clip_ids_full: torch.Tensor = starts["clip_ids"]              # (B,)
  start_ts_full: torch.Tensor = starts["time_steps"]            # (B,)
  past_obs_full: torch.Tensor = starts["past_noisy_obs"]        # (B, 9, 93)
  prev_act_full: torch.Tensor = starts["last_action"]           # (B, 29)

  device = sim_ctx_full.device

  # Freeze world_model and critic weights during policy BPTT so only input Jacobians are computed
  for p in world_model.parameters():
    p.requires_grad_(False)
  for p in critic.parameters():
    p.requires_grad_(False)
  policy.train()
  policy_optimizer.zero_grad(set_to_none=True)

  # Buffers across microbatches for detached critic GAE training
  mb_s_tilde_chunks: list[torch.Tensor] = []
  mb_goal_chunks: list[torch.Tensor] = []
  mb_reward_chunks: list[torch.Tensor] = []
  mb_term_chunks: list[torch.Tensor] = []
  mb_s_tilde_H_chunks: list[torch.Tensor] = []
  mb_goal_H_chunks: list[torch.Tensor] = []

  total_policy_obj = 0.0
  total_final_surv = 0.0

  for mb_start in range(0, batch_size, microbatch_size):
    mb_end = min(mb_start + microbatch_size, batch_size)
    mb_len = mb_end - mb_start
    weight = float(mb_len) / float(batch_size)
    sl = slice(mb_start, mb_end)

    curr_state = _slice_robot_state(s0_full, sl)
    sim_ctx = sim_ctx_full[sl]
    clip_ids = clip_ids_full[sl]
    start_time_steps = start_ts_full[sl]
    past_noisy_obs = past_obs_full[sl]
    prev_action = prev_act_full[sl]
    clip_lens = motion_lib.clip_lengths[clip_ids]

    c_surv = torch.ones(mb_len, dtype=torch.float32, device=device)
    discount = 1.0
    policy_return = torch.zeros(mb_len, dtype=torch.float32, device=device)

    det_s_tilde_list: list[torch.Tensor] = []
    det_goal_list: list[torch.Tensor] = []
    det_reward_list: list[torch.Tensor] = []
    det_term_prob_list: list[torch.Tensor] = []

    s_tilde, cos_yaw, sin_yaw = state_to_canonical(curr_state, default_joint_pos)
    for t in range(horizon):
      curr_ts = start_time_steps + t

      clean_o = compute_clean_obs(curr_state, sim_ctx, prev_action, default_joint_pos)
      noisy_o = clean_o + sample_obs_noise(mb_len, device)
      obs_hist = torch.cat((past_noisy_obs, noisy_o.unsqueeze(1)), dim=1)  # (mb_len, 10, 93)
      obs_flat = obs_hist.reshape(mb_len, 10 * OBS_FRAME_DIM)

      goal_t = motion_lib.get_goal_sequence(
        clip_ids, curr_ts, curr_state.root_pos_w, curr_state.root_rot_w, default_joint_pos
      )

      action_t = policy(obs_flat, goal_t)

      next_state, _ = world_model.predict_next_state(
        curr_state, action_t, sim_ctx, default_joint_pos, s_tilde, cos_yaw, sin_yaw
      )

      next_ts = curr_ts + 1
      flat_next_idx = motion_lib.get_global_indices(clip_ids, next_ts)
      ref_b_pos = motion_lib.body_pos_w[flat_next_idx]
      ref_b_rot = motion_lib.body_rot_w[flat_next_idx]
      ref_b_lvel = motion_lib.body_lin_vel_w[flat_next_idx]
      ref_b_avel = motion_lib.body_ang_vel_w[flat_next_idx]

      r_hat = compute_tracking_reward(
        next_state,
        action_t,
        prev_action,
        ref_b_pos,
        ref_b_rot,
        ref_b_lvel,
        ref_b_avel,
        soft_joint_pos_limits,
        dt=step_dt,
      )

      next_s_tilde, next_cos_yaw, next_sin_yaw = state_to_canonical(next_state, default_joint_pos)
      d_logit = world_model.predict_termination_logit(
        next_state,
        ref_b_pos[:, 0],
        ref_b_rot[:, 0],
        default_joint_pos,
        s_tilde=next_s_tilde,
        safety_margin_z=safety_margin_z,
        safety_margin_tilt=safety_margin_tilt,
      )
      d_hat = torch.sigmoid(d_logit)
      in_clip = (next_ts < (clip_lens - 1)).float()

      policy_return = policy_return + (discount * c_surv * in_clip) * r_hat

      eff_non_term = (1.0 - d_hat.detach()) * in_clip
      det_s_tilde_list.append(s_tilde.detach())
      det_goal_list.append(goal_t.detach())
      det_reward_list.append((r_hat.detach() * in_clip))
      det_term_prob_list.append(1.0 - eff_non_term)

      c_surv = c_surv * (1.0 - d_hat) * in_clip
      discount *= gamma
      past_noisy_obs = obs_hist[:, 1:]
      prev_action = action_t
      curr_state = next_state
      s_tilde, cos_yaw, sin_yaw = next_s_tilde, next_cos_yaw, next_sin_yaw

    # Terminal value bootstrap at step H (Eq. 2)
    final_ts = start_time_steps + horizon
    s_tilde_H = s_tilde
    goal_H = motion_lib.get_goal_sequence(
      clip_ids, final_ts, curr_state.root_pos_w, curr_state.root_rot_w, default_joint_pos
    )
    v_H = critic(s_tilde_H, goal_H, sim_ctx)
    mb_obj = (policy_return + (discount * c_surv) * v_H).mean()
    mb_loss = -mb_obj * weight
    mb_loss.backward()

    total_policy_obj += float(mb_obj.item()) * weight
    total_final_surv += float(c_surv.mean().item()) * weight

    mb_s_tilde_chunks.append(torch.stack(det_s_tilde_list, dim=0))  # (H, mb_len, 266)
    mb_goal_chunks.append(torch.stack(det_goal_list, dim=0))        # (H, mb_len, 730)
    mb_reward_chunks.append(torch.stack(det_reward_list, dim=0))    # (H, mb_len)
    mb_term_chunks.append(torch.stack(det_term_prob_list, dim=0))   # (H, mb_len)
    mb_s_tilde_H_chunks.append(s_tilde_H.detach())
    mb_goal_H_chunks.append(goal_H.detach())

  average_gradients(policy, world_size)
  # Clip policy gradients: global norm 1.0, then element-wise 0.1 (Table A.5)
  grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0).item())
  torch.nn.utils.clip_grad_value_(policy.parameters(), clip_value=0.1)
  if math.isfinite(grad_norm):
    policy_optimizer.step()

  # Unfreeze critic and fit V_psi to GAE targets (Eq. 3 & Table A.5)
  for p in critic.parameters():
    p.requires_grad_(True)
  critic.train()

  with torch.no_grad():
    s_tilde_all = torch.cat(mb_s_tilde_chunks, dim=1)          # (H, B, 266)
    goal_all = torch.cat(mb_goal_chunks, dim=1)                # (H, B, 730)
    rewards_all = torch.cat(mb_reward_chunks, dim=1)           # (H, B)
    terms_all = torch.cat(mb_term_chunks, dim=1)               # (H, B)
    s_tilde_H_all = torch.cat(mb_s_tilde_H_chunks, dim=0)      # (B, 266)
    goal_H_all = torch.cat(mb_goal_H_chunks, dim=0)            # (B, 730)
    ctx_all = sim_ctx_full.unsqueeze(0).expand(horizon, -1, -1)  # (H, B, 33)

    # Update critic input normalizer
    critic_raw_flat = torch.cat(
      (s_tilde_all.flatten(0, 1), goal_all.flatten(0, 1), ctx_all.flatten(0, 1)), dim=-1
    )
    critic.in_norm.update(critic_raw_flat)

    v_all = critic(
      s_tilde_all.flatten(0, 1), goal_all.flatten(0, 1), ctx_all.flatten(0, 1)
    ).view(horizon, batch_size)
    v_boot = critic(s_tilde_H_all, goal_H_all, sim_ctx_full).view(batch_size)

    # Backward GAE recursion (Eq. 3): R_hat_t = r_{t+1} + gamma * (1 - d_{t+1}) * ((1 - lam) * V_{t+1} + lam * R_hat_{t+1})
    r_targets = torch.zeros_like(rewards_all)
    next_r_hat = v_boot
    next_v = v_boot
    for t in range(horizon - 1, -1, -1):
      non_term = 1.0 - terms_all[t]
      r_hat_t = rewards_all[t] + gamma * non_term * ((1.0 - gae_lambda) * next_v + gae_lambda * next_r_hat)
      r_targets[t] = r_hat_t
      next_r_hat = r_hat_t
      next_v = v_all[t]

    flat_s = s_tilde_all.reshape(horizon * batch_size, -1)
    flat_g = goal_all.reshape(horizon * batch_size, -1)
    flat_c = ctx_all.reshape(horizon * batch_size, -1)
    flat_targets = r_targets.reshape(horizon * batch_size)

  total_samples = horizon * batch_size
  mb_size = total_samples // critic_minibatches
  critic_loss_acc = 0.0

  for _ in range(critic_epochs):
    perm = torch.randperm(total_samples, device=device)
    for mb in range(critic_minibatches):
      mb_idx = perm[mb * mb_size : (mb + 1) * mb_size]
      v_pred = critic(flat_s[mb_idx], flat_g[mb_idx], flat_c[mb_idx])
      v_loss = F.mse_loss(v_pred, flat_targets[mb_idx])

      critic_optimizer.zero_grad(set_to_none=True)
      v_loss.backward()
      average_gradients(critic, world_size)
      torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
      critic_optimizer.step()
      critic_loss_acc += float(v_loss.item())

  # Restore world_model requires_grad
  for p in world_model.parameters():
    p.requires_grad_(True)

  return {
    "policy/imagined_obj": total_policy_obj,
    "policy/mean_step_reward": float((rewards_all.mean() / (5.0 * step_dt)).item()),
    "policy/final_survival": total_final_surv,
    "policy/grad_norm": grad_norm,
    "critic/loss": critic_loss_acc / (critic_epochs * critic_minibatches),
    "critic/mean_value": float(v_all.mean().item()),
  }


def main():
  root_dir = Path(__file__).resolve().parent.parent
  parser = argparse.ArgumentParser(description="Train SuperMimic on LAFAN1 Unitree G1")
  parser.add_argument("--dataset", type=str, default=str(root_dir / "data" / "lafan1_g1_50fps.pt"))
  parser.add_argument("--logdir", type=str, default=str(root_dir / "logs" / "supermimic_lafan1"))
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--buffer-capacity", type=int, default=1000)
  parser.add_argument("--wm-batches", type=int, default=10)
  parser.add_argument("--wm-batch-size", type=int, default=2048)
  parser.add_argument("--wm-window", type=int, default=8)
  parser.add_argument("--imagination-batch", type=int, default=4096)
  parser.add_argument("--imagination-microbatch", type=int, default=1024)
  parser.add_argument("--imagination-horizon", type=int, default=32)
  parser.add_argument("--lr", type=float, default=2e-4)
  parser.add_argument("--action-noise", type=float, default=0.1)
  parser.add_argument("--max-episode-length", type=int, default=500)
  parser.add_argument("--long-episode-fraction", type=float, default=0.2)
  parser.add_argument("--rsi-scale", type=float, default=0.0)
  parser.add_argument("--safety-margin-z", type=float, default=0.0)
  parser.add_argument("--safety-margin-tilt", type=float, default=0.0)
  parser.add_argument("--ema-decay", type=float, default=0.995)
  parser.add_argument("--resume-from-ema", action="store_true")
  parser.add_argument("--max-iterations", type=int, default=320_000)
  parser.add_argument("--warmup-steps", type=int, default=200)
  parser.add_argument("--wm-warmup-updates", type=int, default=50)
  parser.add_argument("--eval-interval", type=int, default=1000)
  parser.add_argument("--full-eval-interval", type=int, default=4000)
  parser.add_argument("--target-return", type=float, default=0.83)
  parser.add_argument("--resume", type=str, default="")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"
  else:
    local_rank = 0
    rank = 0
    world_size = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

  is_main = rank == 0
  rank_seed = args.seed + rank * 10007
  torch.manual_seed(rank_seed)
  np.random.seed(rank_seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.set_float32_matmul_precision("high")

  # Scale learning rate by sqrt(N_GPU) per Table A.5
  effective_base_lr = args.lr * math.sqrt(world_size)

  logdir = Path(args.logdir)
  if is_main:
    logdir.mkdir(parents=True, exist_ok=True)
  metrics_file = logdir / "metrics.jsonl"

  if is_main:
    print(
      f"[SuperMimic] World size={world_size} GPUs | Effective base LR={effective_base_lr:.6f} | "
      f"Loading dataset from {args.dataset}..."
    )
  motion_lib = MotionLibrary(dataset_path=args.dataset, device=device)
  if is_main:
    print(
      f"[SuperMimic] Loaded {motion_lib.num_clips} clips, {motion_lib.total_frames} frames "
      f"({motion_lib.num_feasible_bins}/{motion_lib.num_bins} feasible 1s bins)."
    )
    print(
      f"[SuperMimic] Initializing training environment "
      f"(num_envs={args.num_envs} per GPU, max_episode_length={args.max_episode_length}, "
      f"long_episode_fraction={args.long_episode_fraction}, rsi_scale={args.rsi_scale})..."
    )
  env = G1TrackingEnv(
    motion_lib=motion_lib,
    num_envs=args.num_envs,
    device=device,
    play_mode=False,
    max_episode_length=args.max_episode_length,
    long_episode_fraction=args.long_episode_fraction,
    rsi_scale=args.rsi_scale,
  )

  eval_env = None
  if is_main:
    print("[SuperMimic] Initializing 40-clip deterministic evaluation environment...")
    eval_env = G1TrackingEnv(motion_lib=motion_lib, num_envs=motion_lib.num_clips, device=device, play_mode=True)

  buffer = ReplayBuffer(num_envs=args.num_envs, capacity=args.buffer_capacity, device=device)

  # Initialize models with identical seed across all ranks
  torch.manual_seed(args.seed)
  world_model = WorldModel().to(device)
  policy = MotionPolicy().to(device)
  critic = Critic().to(device)
  torch.manual_seed(rank_seed)

  wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=effective_base_lr)
  policy_optimizer = torch.optim.Adam(policy.parameters(), lr=effective_base_lr, betas=(0.9, 0.95))
  critic_optimizer = torch.optim.Adam(critic.parameters(), lr=effective_base_lr)

  start_iter = 0
  update_count = 0
  best_eval_return = 0.0
  best_seg_return = 0.0

  ema_policy: MotionPolicy | None = None
  if args.ema_decay > 0.0:
    ema_policy = copy.deepcopy(policy)
    ema_policy.eval()
    for p in ema_policy.parameters():
      p.requires_grad_(False)

  if args.resume and os.path.exists(args.resume):
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    world_model.load_state_dict(ckpt["world_model"])
    if "raw_policy" in ckpt and not args.resume_from_ema:
      policy.load_state_dict(ckpt["raw_policy"])
    else:
      policy.load_state_dict(ckpt["policy"])
    if ema_policy is not None:
      ema_policy.load_state_dict(ckpt["policy"])
    critic.load_state_dict(ckpt["critic"])
    wm_optimizer.load_state_dict(ckpt["wm_optimizer"])
    policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
    critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
    if "bin_returns" in ckpt:
      motion_lib.bin_returns.copy_(ckpt["bin_returns"])
    if "bin_past_softmax" in ckpt:
      motion_lib.bin_past_softmax.copy_(ckpt["bin_past_softmax"])
      motion_lib.bin_past_softmax.mul_(motion_lib.bin_feasible.float())
    # Re-reset training envs using the resumed & feasibility-filtered bin distribution
    env.reset_envs(torch.arange(args.num_envs, device=device))
    start_iter = ckpt.get("iteration", 0)
    update_count = ckpt.get("update", 0)
    if (logdir / "best.pt").exists():
      best_eval_return = ckpt.get("best_eval_return", 0.0)
      best_seg_return = ckpt.get("best_seg_return", 0.0)
    if is_main:
      print(
        f"[SuperMimic] Resumed from {args.resume} at iteration {start_iter}, update {update_count} "
        f"(best_eval_return={best_eval_return:.4f}, best_seg_return={best_seg_return:.4f})"
      )

  all_norms = [
    world_model.in_norm,
    world_model.delta_norm,
    policy.obs_norm,
    policy.goal_norm,
    critic.in_norm,
  ]

  @torch.no_grad()
  def collect_sim_step() -> None:
    state_t = env.current_state
    clean_obs_t = env.clean_obs.clone()
    sim_ctx_t = env.sim_context.clone()
    clip_id_t = env.clip_ids.clone()
    ts_t = env.time_steps.clone()
    ep_id_t = env.episode_id.clone()

    obs_flat, goal_seq = env.get_policy_inputs()
    policy.obs_norm.update(env.obs_history)
    policy.goal_norm.update(goal_seq)

    det_action = policy(obs_flat, goal_seq)
    # Action noise epsilon_t ~ N(0, sigma^2) (Table A.5)
    noisy_action = det_action + args.action_noise * torch.randn_like(det_action)

    next_state_t, reward_t, terminated_t, timed_out_t = env.step(noisy_action, auto_reset=True)

    # Update world model input and residual normalizers
    s_tilde_t, cos_y, sin_y = state_to_canonical(state_t, env.default_joint_pos)
    wm_in = torch.cat((s_tilde_t, noisy_action, sim_ctx_t), dim=-1)
    tgt_delta = compute_target_delta(
      state_t, s_tilde_t, cos_y, sin_y, next_state_t, env.default_joint_pos
    )
    world_model.in_norm.update(wm_in)
    world_model.delta_norm.update(tgt_delta)

    buffer.add(
      state=state_t,
      action=noisy_action,
      next_state=next_state_t,
      reward=reward_t,
      terminated=terminated_t,
      timed_out=timed_out_t,
      clean_obs=clean_obs_t,
      sim_context=sim_ctx_t,
      clip_id=clip_id_t,
      time_step=ts_t,
      episode_id=ep_id_t,
    )

  if is_main:
    print(f"[SuperMimic] Prefilling replay buffer with {args.warmup_steps} simulator steps...")
  for _ in range(args.warmup_steps):
    collect_sim_step()
  sync_normalizers_and_bins(all_norms, motion_lib, world_size)

  if start_iter == 0 and args.wm_warmup_updates > 0:
    if is_main:
      print(f"[SuperMimic] Running {args.wm_warmup_updates} world-model warmup updates...")
    for _ in range(args.wm_warmup_updates):
      train_world_model_step(
        world_model=world_model,
        wm_optimizer=wm_optimizer,
        buffer=buffer,
        motion_lib=motion_lib,
        default_joint_pos=env.default_joint_pos,
        num_batches=args.wm_batches,
        batch_size=args.wm_batch_size,
        window_k=args.wm_window,
        world_size=world_size,
      )

  train_start_time = time.perf_counter()

  if is_main:
    print("[SuperMimic] Starting main training loop (UTD = 0.4: 2 updates per 5 simulator steps)...")
  for iteration in range(start_iter + 1, args.max_iterations + 1):
    # Collect 1 simulator step per iteration
    collect_sim_step()

    # Update-to-data ratio = 0.4 -> perform WM + Policy + Critic update on 2 out of every 5 steps (steps 2 and 5)
    mod = iteration % 5
    if mod not in (2, 0):
      continue

    update_count += 1
    if update_count % 5 == 0:
      sync_normalizers_and_bins(all_norms, motion_lib, world_size)

    curr_lr = cosine_lr(iteration, base_lr=effective_base_lr)
    set_optimizer_lr(wm_optimizer, curr_lr)
    set_optimizer_lr(policy_optimizer, curr_lr)
    set_optimizer_lr(critic_optimizer, curr_lr)

    wm_metrics = train_world_model_step(
      world_model=world_model,
      wm_optimizer=wm_optimizer,
      buffer=buffer,
      motion_lib=motion_lib,
      default_joint_pos=env.default_joint_pos,
      num_batches=args.wm_batches,
      batch_size=args.wm_batch_size,
      window_k=args.wm_window,
      world_size=world_size,
    )

    # Ramp imagination horizon smoothly from 8 -> 16 -> 32 over first 150 updates
    if update_count < 40:
      curr_H = min(8, args.imagination_horizon)
    elif update_count < 120:
      curr_H = min(16, args.imagination_horizon)
    else:
      curr_H = args.imagination_horizon

    pc_metrics = train_policy_and_critic_step(
      world_model=world_model,
      policy=policy,
      critic=critic,
      policy_optimizer=policy_optimizer,
      critic_optimizer=critic_optimizer,
      buffer=buffer,
      motion_lib=motion_lib,
      default_joint_pos=env.default_joint_pos,
      soft_joint_pos_limits=env.soft_joint_pos_limits,
      batch_size=args.imagination_batch,
      microbatch_size=args.imagination_microbatch,
      horizon=curr_H,
      step_dt=env.step_dt,
      world_size=world_size,
      safety_margin_z=args.safety_margin_z,
      safety_margin_tilt=args.safety_margin_tilt,
    )
    if ema_policy is not None:
      update_ema_policy(ema_policy, policy, args.ema_decay)

    should_stop = False
    if is_main and (update_count % 20 == 0 or iteration % args.eval_interval == 0):
      eval_policy = ema_policy if ema_policy is not None else policy
      elapsed_h = (time.perf_counter() - train_start_time) / 3600.0
      train_ret = env.get_mean_episode_return()
      log_data = {
        "iteration": iteration,
        "update": update_count,
        "wall_hours": elapsed_h,
        "lr": curr_lr,
        "train/episode_return": train_ret,
        **wm_metrics,
        **pc_metrics,
      }

      if iteration % args.eval_interval == 0:
        seg_metrics = evaluate_Window_segments(env, eval_policy, segment_steps=500)
        log_data.update(seg_metrics)
        seg_ret = seg_metrics["eval_seg/normalized_return"]
        if seg_ret >= best_seg_return:
          best_seg_return = seg_ret
          torch.save(
            {
              "iteration": iteration,
              "update": update_count,
              "world_model": world_model.state_dict(),
              "policy": eval_policy.state_dict(),
              "raw_policy": policy.state_dict(),
              "critic": critic.state_dict(),
              "wm_optimizer": wm_optimizer.state_dict(),
              "policy_optimizer": policy_optimizer.state_dict(),
              "critic_optimizer": critic_optimizer.state_dict(),
              "bin_returns": motion_lib.bin_returns,
              "bin_past_softmax": motion_lib.bin_past_softmax,
              "best_eval_return": best_eval_return,
              "best_seg_return": best_seg_return,
              "metrics": log_data,
            },
            logdir / "best_seg.pt",
          )

      if iteration % args.full_eval_interval == 0 or (
        iteration % args.eval_interval == 0 and log_data.get("eval_seg/normalized_return", 0.0) >= 0.80
      ):
        assert eval_env is not None
        full_metrics = evaluate_full_clips(eval_env, eval_policy)
        log_data.update(full_metrics)
        eval_ret = full_metrics["eval/normalized_return"]
        if eval_ret >= best_eval_return:
          best_eval_return = eval_ret
          torch.save(
            {
              "iteration": iteration,
              "update": update_count,
              "world_model": world_model.state_dict(),
              "policy": eval_policy.state_dict(),
              "raw_policy": policy.state_dict(),
              "critic": critic.state_dict(),
              "wm_optimizer": wm_optimizer.state_dict(),
              "policy_optimizer": policy_optimizer.state_dict(),
              "critic_optimizer": critic_optimizer.state_dict(),
              "bin_returns": motion_lib.bin_returns,
              "bin_past_softmax": motion_lib.bin_past_softmax,
              "best_eval_return": best_eval_return,
              "best_seg_return": best_seg_return,
              "metrics": log_data,
            },
            logdir / "best.pt",
          )

      if iteration % args.eval_interval == 0:
        torch.save(
          {
            "iteration": iteration,
            "update": update_count,
            "world_model": world_model.state_dict(),
            "policy": eval_policy.state_dict(),
            "raw_policy": policy.state_dict(),
            "critic": critic.state_dict(),
            "wm_optimizer": wm_optimizer.state_dict(),
            "policy_optimizer": policy_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "bin_returns": motion_lib.bin_returns,
            "bin_past_softmax": motion_lib.bin_past_softmax,
            "best_eval_return": best_eval_return,
            "best_seg_return": best_seg_return,
            "metrics": log_data,
          },
          logdir / "latest.pt",
        )

      with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")

      msg = (
        f"[Iter {iteration:6d} | Upd {update_count:5d} | {elapsed_h:.2f}h] "
        f"train_ret={train_ret:.4f} | step_r={pc_metrics['policy/mean_step_reward']:.4f} | "
        f"wm_8step={wm_metrics['wm/8step_pred_error']:.5f} | surv={pc_metrics['policy/final_survival']:.3f}"
      )
      if "eval_seg/normalized_return" in log_data:
        msg += f" | seg_ret={log_data['eval_seg/normalized_return']:.4f} (fail={log_data['eval_seg/failed_segments']})"
      if "eval/normalized_return" in log_data:
        msg += (
          f" | FULL_EVAL={log_data['eval/normalized_return']:.4f} "
          f"(weighted={log_data['eval/weighted_normalized_return']:.4f}, "
          f"failed={log_data['eval/failed_clips']}/40)"
        )
      print(msg, flush=True)

      if best_eval_return >= args.target_return:
        print(
          f"[SuperMimic] Target normalized return {args.target_return} reached "
          f"(best_eval_return={best_eval_return:.4f})! Saving final checkpoint."
        )
        should_stop = True

    if world_size > 1 and (update_count % 20 == 0 or iteration % args.eval_interval == 0):
      stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.int32, device=device)
      dist.broadcast(stop_tensor, src=0)
      should_stop = bool(stop_tensor.item() == 1)

    if should_stop:
      break

  if world_size > 1:
    dist.destroy_process_group()


if __name__ == "__main__":
  main()

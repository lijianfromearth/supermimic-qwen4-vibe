"""Multi-clip motion library and return-based reset sampling (Appendix D.1)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from supermimic.math_utils import matrix_to_rot6d, quat_to_matrix


class MotionLibrary:
  """GPU-resident library of 50Hz reference motions with return-based reset sampling."""

  def __init__(
    self,
    dataset_path: str | Path = "/home/sunteng/carbon-test/data/lafan1_g1_50fps.pt",
    device: str = "cuda",
    fps: int = 50,
    future_horizon: int = 10,
    gamma: float = 0.99,
  ) -> None:
    self.device = device
    self.fps = fps
    self.dt = 1.0 / fps
    self.future_horizon = future_horizon
    self.gamma = gamma

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    self.clip_names: list[str] = data["clip_names"]
    self.num_clips = len(self.clip_names)
    self.clip_lengths = data["clip_lengths"].to(device=device, dtype=torch.long)
    self.clip_starts = data["clip_starts"].to(device=device, dtype=torch.long)
    self.total_frames = int(self.clip_lengths.sum().item())

    self.joint_pos = data["joint_pos"].to(device=device, dtype=torch.float32)
    self.joint_vel = data["joint_vel"].to(device=device, dtype=torch.float32)
    self.body_pos_w = data["body_pos_w"].to(device=device, dtype=torch.float32)
    self.body_quat_w = data["body_quat_w"].to(device=device, dtype=torch.float32)
    self.body_rot_w = quat_to_matrix(self.body_quat_w)
    self.body_lin_vel_w = data["body_lin_vel_w"].to(device=device, dtype=torch.float32)
    self.body_ang_vel_w = data["body_ang_vel_w"].to(device=device, dtype=torch.float32)

    # Root is index 0 ('pelvis') of the 14 tracked bodies
    self.root_pos_w = self.body_pos_w[:, 0]
    self.root_quat_w = self.body_quat_w[:, 0]
    self.root_rot_w = self.body_rot_w[:, 0]
    self.root_lin_vel_w = self.body_lin_vel_w[:, 0]
    self.root_ang_vel_w = self.body_ang_vel_w[:, 0]

    # Build 1s (50-frame) bins across all clips for return-based reset sampling (Appendix D.1 & Appendix E)
    bin_clip_ids = []
    bin_start_frames = []
    bin_end_frames = []
    bin_feasible = []
    bpos_cpu = data["body_pos_w"]
    rlvel_cpu = data["body_lin_vel_w"][:, 0]
    cstarts_cpu = data["clip_starts"]
    for c in range(self.num_clips):
      length = int(self.clip_lengths[c].item())
      c_start = int(cstarts_cpu[c].item())
      # Leave at least 50 frames (1s) before end of clip for training resets
      max_start = max(length - fps, 1)
      num_bins_c = max((max_start + fps - 1) // fps, 1)
      for b in range(num_bins_c):
        s_frame = b * fps
        e_frame = min((b + 1) * fps, max_start)
        if e_frame <= s_frame:
          e_frame = s_frame + 1
        bin_clip_ids.append(c)
        bin_start_frames.append(s_frame)
        bin_end_frames.append(e_frame)
        # Appendix E feasibility gate for flat-ground resets: exclude stationary elevated-support
        # / mid-air chair-sitting bins (min body height > 0.12 m while planar/3D speed < 0.5 m/s)
        g_s = c_start + s_frame
        g_e = c_start + e_frame
        z_min = bpos_cpu[g_s:g_e, :, 2].min(dim=-1).values
        spd = rlvel_cpu[g_s:g_e].norm(dim=-1)
        feas_ratio = float((~((z_min > 0.12) & (spd < 0.5))).float().mean().item())
        bin_feasible.append(feas_ratio >= 0.5)

    self.num_bins = len(bin_clip_ids)
    self.bin_clip_ids = torch.tensor(bin_clip_ids, dtype=torch.long, device=device)
    self.bin_start_frames = torch.tensor(bin_start_frames, dtype=torch.long, device=device)
    self.bin_end_frames = torch.tensor(bin_end_frames, dtype=torch.long, device=device)
    self.bin_feasible = torch.tensor(bin_feasible, dtype=torch.bool, device=device)
    self.num_feasible_bins = int(self.bin_feasible.sum().item())

    # Max per-step reward is 5.0 * dt = 0.10; G_max = 0.10 / (1 - gamma) = 10.0
    self.g_max = 5.0 * self.dt / (1.0 - self.gamma)
    self.bin_returns = torch.zeros(self.num_bins, dtype=torch.float32, device=device)
    self.bin_past_softmax = torch.ones(self.num_bins, dtype=torch.float32, device=device) / self.num_bins
    self.bin_ema_alpha = 0.1

    # Future offsets 1..K
    self.future_offsets = torch.arange(
      1, self.future_horizon + 1, dtype=torch.long, device=device
    ).unsqueeze(0)  # (1, K)

  def get_global_indices(self, clip_ids: torch.Tensor, local_frames: torch.Tensor) -> torch.Tensor:
    """Convert (clip_ids, local_frames) to flat indices into the concatenated tensors."""
    max_frames = self.clip_lengths[clip_ids] - 1
    clamped = torch.clamp(local_frames, min=torch.zeros_like(max_frames), max=max_frames)
    return self.clip_starts[clip_ids] + clamped

  def get_goal_sequence(
    self,
    clip_ids: torch.Tensor,
    local_frames: torch.Tensor,
    robot_root_pos: torch.Tensor,
    robot_root_rot: torch.Tensor,
    default_joint_pos: torch.Tensor,
  ) -> torch.Tensor:
    """Compute K=10 future reference sequence g_t relative to current robot root pose.
    clip_ids, local_frames: (B,)
    robot_root_pos: (B, 3)
    robot_root_rot: (B, 3, 3)
    default_joint_pos: (1, 29) or (29,)
    Returns:
      g_t: (B, K * 73) = (B, 730)
    """
    B = clip_ids.shape[0]
    K = self.future_horizon
    fut_frames = local_frames.unsqueeze(1) + self.future_offsets  # (B, K)
    max_frames = (self.clip_lengths[clip_ids] - 1).unsqueeze(1)  # (B, K)
    fut_frames = fut_frames.clamp(min=0).minimum(max_frames)
    flat_idx = (self.clip_starts[clip_ids].unsqueeze(1) + fut_frames).reshape(-1)  # (B*K,)

    ref_q = self.joint_pos[flat_idx].view(B, K, 29) - default_joint_pos.view(1, 1, 29)
    ref_qd = self.joint_vel[flat_idx].view(B, K, 29)
    ref_rpos = self.root_pos_w[flat_idx].view(B, K, 3)
    ref_rrot = self.root_rot_w[flat_idx].view(B, K, 3, 3)
    ref_rlvel = self.root_lin_vel_w[flat_idx].view(B, K, 3)
    ref_ravel = self.root_ang_vel_w[flat_idx].view(B, K, 3)

    # Express future reference root poses and velocities in current robot root frame
    # R_rob^T: (B, 1, 3, 3)
    R_rob_T = robot_root_rot.transpose(-1, -2).unsqueeze(1)
    pos_diff = (ref_rpos - robot_root_pos.unsqueeze(1)).unsqueeze(-1)  # (B, K, 3, 1)
    rel_pos_b = torch.matmul(R_rob_T, pos_diff).squeeze(-1)  # (B, K, 3)

    rel_rot_b = torch.matmul(R_rob_T, ref_rrot)  # (B, K, 3, 3)
    rel_ori_6d = matrix_to_rot6d(rel_rot_b)  # (B, K, 6)

    rel_lvel_b = torch.matmul(R_rob_T, ref_rlvel.unsqueeze(-1)).squeeze(-1)  # (B, K, 3)
    rel_avel_b = torch.matmul(R_rob_T, ref_ravel.unsqueeze(-1)).squeeze(-1)  # (B, K, 3)

    g_seq = torch.cat(
      (ref_q, ref_qd, rel_pos_b, rel_ori_6d, rel_lvel_b, rel_avel_b), dim=-1
    )  # (B, K, 73)
    return g_seq.reshape(B, K * 73)

  @torch.no_grad()
  def sample_reset_bins(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample reset (clip_ids, local_frames, bin_ids) using return-based reset sampling (Appendix D.1).
    Mixes:
      50% uniform over bins
      40% from softmax_b(-25 * G_b / G_max)
      10% proportional to each bin's past softmax weight
    """
    feas_f = self.bin_feasible.float()
    uniform_prob = feas_f / feas_f.sum().clamp(min=1.0)
    logits = (-25.0 * self.bin_returns / self.g_max).masked_fill(~self.bin_feasible, -1.0e9)
    softmax_prob = F.softmax(logits, dim=0) * feas_f
    softmax_prob = softmax_prob / softmax_prob.sum().clamp(min=1e-12)
    self.bin_past_softmax.mul_(feas_f).add_(softmax_prob)
    past_prob = self.bin_past_softmax / self.bin_past_softmax.sum().clamp(min=1e-12)

    probs = 0.50 * uniform_prob + 0.40 * softmax_prob + 0.10 * past_prob
    probs = probs / probs.sum()

    sampled_bins = torch.multinomial(probs, num_samples, replacement=True)
    clip_ids = self.bin_clip_ids[sampled_bins]
    starts = self.bin_start_frames[sampled_bins]
    ends = self.bin_end_frames[sampled_bins]
    span = (ends - starts).clamp(min=1).float()
    offsets = (torch.rand(num_samples, device=self.device) * span).long()
    local_frames = starts + offsets
    return clip_ids, local_frames, sampled_bins

  @torch.no_grad()
  def update_bin_returns(self, bin_ids: torch.Tensor, episode_disc_returns: torch.Tensor) -> None:
    """Update running estimate G_b for completed or reset episodes."""
    if bin_ids.numel() == 0:
      return
    # Compute mean discounted return per unique bin in this batch
    unique_bins, inverse_idx = torch.unique(bin_ids, return_inverse=True)
    counts = torch.bincount(inverse_idx, minlength=unique_bins.numel()).float().clamp(min=1.0)
    sums = torch.zeros_like(counts)
    sums.scatter_add_(0, inverse_idx, episode_disc_returns)
    mean_returns = sums / counts
    self.bin_returns[unique_bins] = (
      (1.0 - self.bin_ema_alpha) * self.bin_returns[unique_bins]
      + self.bin_ema_alpha * mean_returns
    )

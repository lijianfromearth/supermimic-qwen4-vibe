"""GPU-resident circular replay buffer for SuperMimic (1000 steps x 4096 envs)."""

from __future__ import annotations

import torch

from supermimic.state import (
  OBS_FRAME_DIM,
  PACKED_STATE_DIM,
  RobotState,
  SIM_CONTEXT_DIM,
  pack_state,
  sample_obs_noise,
  unpack_state,
)


class ReplayBuffer:
  """Circular GPU replay buffer storing transitions and supporting k-step window & H-step start sampling."""

  def __init__(
    self,
    num_envs: int = 4096,
    capacity: int = 1000,
    history_len: int = 10,
    device: str = "cuda",
  ) -> None:
    self.num_envs = num_envs
    self.capacity = capacity
    self.history_len = history_len
    self.device = device

    self.ptr = 0
    self.size = 0

    self.state = torch.zeros(
      (capacity, num_envs, PACKED_STATE_DIM), dtype=torch.float32, device=device
    )
    self.action = torch.zeros((capacity, num_envs, 29), dtype=torch.float32, device=device)
    self.clean_obs = torch.zeros(
      (capacity, num_envs, OBS_FRAME_DIM), dtype=torch.float32, device=device
    )
    self.sim_context = torch.zeros(
      (capacity, num_envs, SIM_CONTEXT_DIM), dtype=torch.float32, device=device
    )
    self.reward = torch.zeros((capacity, num_envs), dtype=torch.float32, device=device)
    self.terminated = torch.zeros((capacity, num_envs), dtype=torch.bool, device=device)
    self.done = torch.zeros((capacity, num_envs), dtype=torch.bool, device=device)
    self.clip_id = torch.zeros((capacity, num_envs), dtype=torch.long, device=device)
    self.time_step = torch.zeros((capacity, num_envs), dtype=torch.long, device=device)
    self.episode_id = torch.zeros((capacity, num_envs), dtype=torch.long, device=device)

    # Compact storage for exact post-step next_state on episode-ending steps (terminated or timed_out)
    self.reset_bank_cap = 131072
    self.reset_bank_ptr = 0
    self.reset_next_states = torch.zeros(
      (self.reset_bank_cap, PACKED_STATE_DIM), dtype=torch.float32, device=device
    )
    self.reset_bank_idx = torch.full(
      (capacity, num_envs), -1, dtype=torch.long, device=device
    )

  @torch.no_grad()
  def add(
    self,
    state: RobotState,
    action: torch.Tensor,
    next_state: RobotState,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    timed_out: torch.Tensor,
    clean_obs: torch.Tensor,
    sim_context: torch.Tensor,
    clip_id: torch.Tensor,
    time_step: torch.Tensor,
    episode_id: torch.Tensor,
  ) -> None:
    """Push one step of transitions across all parallel environments."""
    idx = self.ptr
    self.state[idx] = pack_state(state)
    self.action[idx] = action
    self.clean_obs[idx] = clean_obs
    self.sim_context[idx] = sim_context
    self.reward[idx] = reward
    self.terminated[idx] = terminated
    done = terminated | timed_out
    self.done[idx] = done
    self.clip_id[idx] = clip_id
    self.time_step[idx] = time_step
    self.episode_id[idx] = episode_id

    self.reset_bank_idx[idx].fill_(-1)
    done_envs = done.nonzero(as_tuple=False).squeeze(-1)
    n_done = done_envs.numel()
    if n_done > 0:
      packed_next = pack_state(next_state)[done_envs]
      bank_indices = (
        torch.arange(n_done, dtype=torch.long, device=self.device) + self.reset_bank_ptr
      ) % self.reset_bank_cap
      self.reset_next_states[bank_indices] = packed_next
      self.reset_bank_idx[idx, done_envs] = bank_indices
      self.reset_bank_ptr = (self.reset_bank_ptr + n_done) % self.reset_bank_cap

    self.ptr = (self.ptr + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  @torch.no_grad()
  def _get_next_packed_state(self, t_idx: torch.Tensor, env_idx: torch.Tensor) -> torch.Tensor:
    """Retrieve exact post-step state s_{t+1} for (t_idx, env_idx)."""
    next_t = (t_idx + 1) % self.capacity
    next_packed = self.state[next_t, env_idx]
    bank_idx = self.reset_bank_idx[t_idx, env_idx]
    reset_mask = (bank_idx >= 0).unsqueeze(-1)
    safe_bank = bank_idx.clamp(min=0)
    return torch.where(reset_mask, self.reset_next_states[safe_bank], next_packed)

  @torch.no_grad()
  def sample_windows(self, batch_size: int = 2048, k: int = 8) -> dict[str, torch.Tensor | list[RobotState]]:
    """Sample k-step windows for fitting the world model (Eq. 1).
    Returns:
      s0: RobotState at step 0 (batch_size,)
      actions: (batch_size, k, 29)
      sim_context: (batch_size, 33)
      target_states: list of k RobotStates for steps 1..k
      target_terms: (batch_size, k) float32 in {0.0, 1.0}
      valid_mask: (batch_size, k) float32 (1.0 while within the same episode, 0.0 after reset)
      clip_ids: (batch_size, k)
      next_time_steps: (batch_size, k)
    """
    assert self.size > k + 2, "ReplayBuffer does not have enough steps yet."
    # Avoid sampling windows that cross the circular buffer write head self.ptr
    # Convert chronological offset [0, self.size - k - 1] to circular index
    oldest = (self.ptr - self.size) % self.capacity
    max_offset = self.size - k - 1
    offsets = torch.randint(0, max_offset, (batch_size,), device=self.device)
    t0 = (oldest + offsets) % self.capacity
    envs = torch.randint(0, self.num_envs, (batch_size,), device=self.device)

    s0 = unpack_state(self.state[t0, envs])
    sim_ctx = self.sim_context[t0, envs]
    ep0 = self.episode_id[t0, envs]

    actions_list = []
    target_states_list = []
    target_terms_list = []
    valid_mask_list = []
    clip_ids_list = []
    next_ts_list = []

    alive = torch.ones(batch_size, dtype=torch.bool, device=self.device)
    for j in range(k):
      tj = (t0 + j) % self.capacity
      next_tj = (tj + 1) % self.capacity
      has_valid_next = (self.episode_id[next_tj, envs] == ep0) | (self.reset_bank_idx[tj, envs] >= 0)
      same_ep = (self.episode_id[tj, envs] == ep0) & alive & has_valid_next
      valid_mask_list.append(same_ep.float())
      actions_list.append(self.action[tj, envs])
      target_terms_list.append(self.terminated[tj, envs].float())
      clip_ids_list.append(self.clip_id[tj, envs])
      next_ts_list.append(self.time_step[tj, envs] + 1)

      next_packed = self._get_next_packed_state(tj, envs)
      target_states_list.append(unpack_state(next_packed))
      # If done occurred at step tj, subsequent steps j' > j belong to a new episode
      alive = same_ep & (~self.done[tj, envs])

    return {
      "s0": s0,
      "actions": torch.stack(actions_list, dim=1),
      "sim_context": sim_ctx,
      "target_states": target_states_list,
      "target_terms": torch.stack(target_terms_list, dim=1),
      "valid_mask": torch.stack(valid_mask_list, dim=1),
      "clip_ids": torch.stack(clip_ids_list, dim=1),
      "next_time_steps": torch.stack(next_ts_list, dim=1),
    }

  @torch.no_grad()
  def sample_imagination_starts(self, batch_size: int = 4096) -> dict[str, torch.Tensor | RobotState]:
    """Sample start states s0 ~ B and their 9-step prior observation history for H=32 imagined rollouts."""
    oldest = (self.ptr - self.size) % self.capacity
    # Leave at least history_len - 1 steps after oldest so history is inside the buffer
    min_offset = min(self.history_len - 1, self.size - 2)
    max_offset = self.size - 1
    offsets = torch.randint(min_offset, max_offset, (batch_size,), device=self.device)
    t0 = (oldest + offsets) % self.capacity
    envs = torch.randint(0, self.num_envs, (batch_size,), device=self.device)

    s0 = unpack_state(self.state[t0, envs])
    sim_ctx = self.sim_context[t0, envs]
    clip_ids = self.clip_id[t0, envs]
    time_steps = self.time_step[t0, envs]
    ep0 = self.episode_id[t0, envs]

    # Reconstruct past 9 frames of noisy observations (t0 - 9 .. t0 - 1), backfilling across episode boundaries
    # Chronological order: lag = 9, 8, ..., 1
    hist_clean = []
    hist_same_ep = []
    for lag in range(self.history_len - 1, 0, -1):
      tl = (t0 - lag) % self.capacity
      hist_clean.append(self.clean_obs[tl, envs])
      hist_same_ep.append(self.episode_id[tl, envs] == ep0)

    # Also include t0 clean_obs as fallback for backfilling if episode started at t0
    obs_t0 = self.clean_obs[t0, envs]
    hist_clean_t = torch.stack(hist_clean, dim=1)  # (B, 9, 93)
    same_ep_t = torch.stack(hist_same_ep, dim=1)   # (B, 9)

    # For any slot not from the same episode, backfill with the earliest valid observation in the episode
    # Walk backward from lag=1 to lag=9 so the earliest valid observation propagates backward
    first_valid = obs_t0.clone()
    for idx in range(self.history_len - 2, -1, -1):
      mask = same_ep_t[:, idx].unsqueeze(-1)
      first_valid = torch.where(mask, hist_clean_t[:, idx], first_valid)
      hist_clean_t[:, idx] = first_valid

    # Add fresh sensor noise to the 9 historical frames
    noise = sample_obs_noise(batch_size * (self.history_len - 1), self.device).view(
      batch_size, self.history_len - 1, OBS_FRAME_DIM
    )
    past_noisy_obs = hist_clean_t + noise  # (B, 9, 93)
    last_action_t0 = obs_t0[:, 64:93].clone()

    return {
      "s0": s0,
      "sim_context": sim_ctx,
      "clip_ids": clip_ids,
      "time_steps": time_steps,
      "past_noisy_obs": past_noisy_obs,  # (B, 9, 93)
      "last_action": last_action_t0,    # (B, 29)
    }

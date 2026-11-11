"""Vectorized MuJoCo Warp Unitree G1 tracking environment for SuperMimic."""

from __future__ import annotations

import re
import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform
from mjlab.utils.lab_api.string import resolve_matching_names_values

from supermimic.data import MotionLibrary
from supermimic.math_utils import quat_to_matrix
from supermimic.preprocess_data import TRACKED_BODY_NAMES
from supermimic.state import (
  OBS_FRAME_DIM,
  SIM_CONTEXT_DIM,
  RobotState,
  compute_clean_obs,
  compute_termination_errors,
  compute_tracking_reward,
  sample_obs_noise,
)


class G1TrackingEnv:
  """High-throughput MuJoCo Warp environment for Unitree G1 motion tracking."""

  def __init__(
    self,
    motion_lib: MotionLibrary,
    num_envs: int = 4096,
    device: str = "cuda",
    history_len: int = 10,
    play_mode: bool = False,
    max_episode_length: int = 500,
  ) -> None:
    self.motion_lib = motion_lib
    self.num_envs = num_envs
    self.device = device
    self.history_len = history_len
    self.play_mode = play_mode
    self.max_episode_length = int(1e9) if play_mode else max_episode_length

    self.physics_dt = 0.005
    self.decimation = 4
    self.step_dt = self.physics_dt * self.decimation  # 0.02s (50Hz)

    scene_cfg = SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_g1_robot_cfg()},
      num_envs=num_envs,
    )
    sim_cfg = SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=self.physics_dt,
        iterations=10,
        ls_iterations=20,
      ),
    )

    self.scene = Scene(scene_cfg, device=device)
    self.sim = Simulation(
      num_envs=num_envs,
      cfg=sim_cfg,
      spec=self.scene.spec,
      variant_info=self.scene.collect_variant_info(),
      device=device,
    )
    self.scene.initialize(
      mj_model=self.sim.mj_model,
      model=self.sim.model,
      data=self.sim.data,
    )
    self.sim.wp_model.opt.warn_overflow = 0
    # Expand domain randomization fields across environments (Table A.3)
    self.sim.expand_model_fields(("geom_friction", "body_ipos"))

    self.robot = self.scene["robot"]
    self.tracked_body_ids = torch.tensor(
      self.robot.find_bodies(TRACKED_BODY_NAMES, preserve_order=True)[0],
      dtype=torch.long,
      device=device,
    )
    torso_local_id = self.robot.find_bodies(("torso_link",))[0][0]
    self.torso_body_id = int(self.robot.indexing.body_ids[torso_local_id].item())

    # Find foot collision geoms: ^(left|right)_foot[1-7]_collision$
    foot_local_ids = self.robot.find_geoms(r"^(left|right)_foot[1-7]_collision$")[0]
    assert len(foot_local_ids) == 14, f"Expected 14 foot geoms, got {len(foot_local_ids)}"
    self.foot_geom_ids = self.robot.indexing.geom_ids[foot_local_ids].to(
      device=device, dtype=torch.long
    )

    # Action scale (1, 29) and default joint pos (1, 29)
    joint_names = self.robot.joint_names
    self.action_scale = torch.ones((1, len(joint_names)), dtype=torch.float32, device=device)
    idx_list, _, val_list = resolve_matching_names_values(G1_ACTION_SCALE, joint_names)
    self.action_scale[:, idx_list] = torch.tensor(val_list, dtype=torch.float32, device=device)
    self.default_joint_pos = self.robot.data.default_joint_pos[0:1].clone()  # (1, 29)
    self.soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[0].clone()  # (29, 2)
    self.default_torso_ipos = self.sim.model.body_ipos[0, self.torso_body_id].clone()  # (3,)

    # Per-environment tracking state
    self.clip_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.reset_bins = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.episode_length = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.max_ep_steps = torch.full((num_envs,), self.max_episode_length, dtype=torch.long, device=device)
    self.episode_id = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._next_episode_id = 1

    # Running return trackers for training metrics & return-based reset sampling
    self.ep_disc_return = torch.zeros(num_envs, dtype=torch.float32, device=device)
    self.ep_undisc_return = torch.zeros(num_envs, dtype=torch.float32, device=device)
    self.ep_gamma_pow = torch.ones(num_envs, dtype=torch.float32, device=device)
    self.completed_ep_returns: list[float] = []

    # Sim context xi (num_envs, 33): [friction(1), encoder_bias(29), torso_com_offset(3)]
    self.sim_context = torch.zeros((num_envs, SIM_CONTEXT_DIM), dtype=torch.float32, device=device)
    self.last_action = torch.zeros((num_envs, 29), dtype=torch.float32, device=device)

    # 10-step observation history buffer: (num_envs, history_len, 93), chronological oldest -> newest
    self.obs_history = torch.zeros(
      (num_envs, history_len, OBS_FRAME_DIM), dtype=torch.float32, device=device
    )
    self.clean_obs = torch.zeros((num_envs, OBS_FRAME_DIM), dtype=torch.float32, device=device)

    # Reset all environments initially
    all_env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    self.reset_envs(all_env_ids)

  @torch.no_grad()
  def _randomize_sim_context(self, env_ids: torch.Tensor) -> None:
    """Sample per-episode domain randomization parameters xi (Table A.3)."""
    n = env_ids.numel()
    if self.play_mode:
      friction = torch.ones((n, 1), dtype=torch.float32, device=self.device)
      enc_bias = torch.zeros((n, 29), dtype=torch.float32, device=self.device)
      com_offset = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
    else:
      friction = torch.empty((n, 1), dtype=torch.float32, device=self.device).uniform_(0.3, 1.2)
      enc_bias = torch.empty((n, 29), dtype=torch.float32, device=self.device).uniform_(-0.01, 0.01)
      com_x = torch.empty((n, 1), dtype=torch.float32, device=self.device).uniform_(-0.025, 0.025)
      com_y = torch.empty((n, 1), dtype=torch.float32, device=self.device).uniform_(-0.05, 0.05)
      com_z = torch.empty((n, 1), dtype=torch.float32, device=self.device).uniform_(-0.05, 0.05)
      com_offset = torch.cat((com_x, com_y, com_z), dim=-1)

    self.sim_context[env_ids, 0:1] = friction
    self.sim_context[env_ids, 1:30] = enc_bias
    self.sim_context[env_ids, 30:33] = com_offset

    # Write to MuJoCo model & robot data
    self.robot.data.encoder_bias[env_ids] = enc_bias
    self.sim.model.geom_friction[env_ids[:, None], self.foot_geom_ids[None, :], 0] = friction
    self.sim.model.body_ipos[env_ids, self.torso_body_id] = (
      self.default_torso_ipos.unsqueeze(0) + com_offset
    )

  @torch.no_grad()
  def get_robot_state(self) -> RobotState:
    """Extract current world-frame RobotState (with env_origins subtracted)."""
    origins = self.scene.env_origins.unsqueeze(1)  # (B, 1, 3)
    b_pos = self.robot.data.body_link_pos_w[:, self.tracked_body_ids] - origins
    b_quat = self.robot.data.body_link_quat_w[:, self.tracked_body_ids]
    b_rot = quat_to_matrix(b_quat)
    b_lvel = self.robot.data.body_link_lin_vel_w[:, self.tracked_body_ids]
    b_avel = self.robot.data.body_link_ang_vel_w[:, self.tracked_body_ids]
    j_pos = self.robot.data.joint_pos.clone()
    j_vel = self.robot.data.joint_vel.clone()
    return RobotState(
      body_pos_w=b_pos,
      body_rot_w=b_rot,
      body_lin_vel_w=b_lvel,
      body_ang_vel_w=b_avel,
      joint_pos=j_pos,
      joint_vel=j_vel,
    )

  @torch.no_grad()
  def reset_envs(
    self,
    env_ids: torch.Tensor,
    clip_ids: torch.Tensor | None = None,
    local_frames: torch.Tensor | None = None,
  ) -> None:
    """Reset specified environments to reference motion states."""
    n = env_ids.numel()
    if n == 0:
      return

    if clip_ids is None or local_frames is None:
      c_ids, l_frames, b_ids = self.motion_lib.sample_reset_bins(n)
      self.reset_bins[env_ids] = b_ids
    else:
      c_ids = clip_ids
      l_frames = local_frames

    self.clip_ids[env_ids] = c_ids
    self.time_steps[env_ids] = l_frames
    self.episode_length[env_ids] = 0
    rem_frames = (self.motion_lib.clip_lengths[c_ids] - 1 - l_frames).clamp(min=1)
    self.max_ep_steps[env_ids] = rem_frames.clamp(max=self.max_episode_length)
    self.ep_disc_return[env_ids] = 0.0
    self.ep_undisc_return[env_ids] = 0.0
    self.ep_gamma_pow[env_ids] = 1.0
    self.episode_id[env_ids] = torch.arange(
      self._next_episode_id, self._next_episode_id + n, dtype=torch.long, device=self.device
    )
    self._next_episode_id += n

    # Domain randomization (Table A.3)
    self._randomize_sim_context(env_ids)

    # Read reference state at (c_ids, l_frames)
    flat_idx = self.motion_lib.get_global_indices(c_ids, l_frames)
    root_pos = self.motion_lib.root_pos_w[flat_idx] + self.scene.env_origins[env_ids]
    root_ori = self.motion_lib.root_quat_w[flat_idx].clone()
    root_lvel = self.motion_lib.root_lin_vel_w[flat_idx].clone()
    root_avel = self.motion_lib.root_ang_vel_w[flat_idx].clone()
    joint_pos = self.motion_lib.joint_pos[flat_idx].clone()
    joint_vel = self.motion_lib.joint_vel[flat_idx].clone()

    if not self.play_mode:
      # Small initial state perturbation around reference (matching mjlab tracking_env_cfg)
      pose_low = torch.tensor([-0.05, -0.05, -0.01, -0.1, -0.1, -0.2], device=self.device)
      pose_high = torch.tensor([0.05, 0.05, 0.01, 0.1, 0.1, 0.2], device=self.device)
      pose_rand = sample_uniform(pose_low, pose_high, (n, 6), device=self.device)
      root_pos = root_pos + pose_rand[:, 0:3]
      ori_delta = quat_from_euler_xyz(pose_rand[:, 3], pose_rand[:, 4], pose_rand[:, 5])
      root_ori = quat_mul(ori_delta, root_ori)

      vel_low = torch.tensor([-0.5, -0.5, -0.2, -0.52, -0.52, -0.78], device=self.device)
      vel_high = torch.tensor([0.5, 0.5, 0.2, 0.52, 0.52, 0.78], device=self.device)
      vel_rand = sample_uniform(vel_low, vel_high, (n, 6), device=self.device)
      root_lvel = root_lvel + vel_rand[:, 0:3]
      root_avel = root_avel + vel_rand[:, 3:6]

      joint_pos = joint_pos + torch.empty_like(joint_pos).uniform_(-0.1, 0.1)

    q_min = self.soft_joint_pos_limits[:, 0].unsqueeze(0)
    q_max = self.soft_joint_pos_limits[:, 1].unsqueeze(0)
    joint_pos = torch.clamp(joint_pos, q_min, q_max)

    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root_state = torch.cat((root_pos, root_ori, root_lvel, root_avel), dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    self.scene.write_data_to_sim(env_ids=env_ids)
    self.sim.forward()

    self.last_action[env_ids] = 0.0
    state = self.get_robot_state()
    clean_o = compute_clean_obs(state, self.sim_context, self.last_action, self.default_joint_pos)
    self.clean_obs[env_ids] = clean_o[env_ids]

    if self.play_mode:
      noisy_o = clean_o[env_ids]
    else:
      noisy_o = clean_o[env_ids] + sample_obs_noise(n, self.device)
    # Backfill all 10 history slots with the initial observation
    self.obs_history[env_ids] = noisy_o.unsqueeze(1).expand(n, self.history_len, OBS_FRAME_DIM).clone()
    self.current_state = state

  @torch.no_grad()
  def save_snapshot(self) -> dict:
    """Save exact simulator and environment state so evaluation on training env is side-effect free."""
    return {
      "qpos": self.sim.data.qpos.clone(),
      "qvel": self.sim.data.qvel.clone(),
      "ctrl": self.sim.data.ctrl.clone(),
      "joint_pos_target": self.robot.data.joint_pos_target.clone(),
      "encoder_bias": self.robot.data.encoder_bias.clone(),
      "geom_friction": self.sim.model.geom_friction.clone(),
      "body_ipos": self.sim.model.body_ipos.clone(),
      "clip_ids": self.clip_ids.clone(),
      "time_steps": self.time_steps.clone(),
      "reset_bins": self.reset_bins.clone(),
      "episode_length": self.episode_length.clone(),
      "max_ep_steps": self.max_ep_steps.clone(),
      "episode_id": self.episode_id.clone(),
      "next_episode_id": self._next_episode_id,
      "ep_disc_return": self.ep_disc_return.clone(),
      "ep_undisc_return": self.ep_undisc_return.clone(),
      "ep_gamma_pow": self.ep_gamma_pow.clone(),
      "sim_context": self.sim_context.clone(),
      "last_action": self.last_action.clone(),
      "obs_history": self.obs_history.clone(),
      "clean_obs": self.clean_obs.clone(),
      "play_mode": self.play_mode,
      "max_episode_length": self.max_episode_length,
      "bin_past_softmax": self.motion_lib.bin_past_softmax.clone(),
    }

  @torch.no_grad()
  def load_snapshot(self, snap: dict) -> None:
    """Restore exact simulator and environment state saved by save_snapshot()."""
    self.sim.data.qpos[:] = snap["qpos"]
    self.sim.data.qvel[:] = snap["qvel"]
    self.sim.data.ctrl[:] = snap["ctrl"]
    self.robot.data.joint_pos_target[:] = snap["joint_pos_target"]
    self.robot.data.encoder_bias[:] = snap["encoder_bias"]
    self.sim.model.geom_friction[:] = snap["geom_friction"]
    self.sim.model.body_ipos[:] = snap["body_ipos"]
    self.sim.forward()

    self.clip_ids.copy_(snap["clip_ids"])
    self.time_steps.copy_(snap["time_steps"])
    self.reset_bins.copy_(snap["reset_bins"])
    self.episode_length.copy_(snap["episode_length"])
    self.max_ep_steps.copy_(snap["max_ep_steps"])
    self.episode_id.copy_(snap["episode_id"])
    self._next_episode_id = snap["next_episode_id"]
    self.ep_disc_return.copy_(snap["ep_disc_return"])
    self.ep_undisc_return.copy_(snap["ep_undisc_return"])
    self.ep_gamma_pow.copy_(snap["ep_gamma_pow"])
    self.sim_context.copy_(snap["sim_context"])
    self.last_action.copy_(snap["last_action"])
    self.obs_history.copy_(snap["obs_history"])
    self.clean_obs.copy_(snap["clean_obs"])
    self.play_mode = snap["play_mode"]
    self.max_episode_length = snap["max_episode_length"]
    self.motion_lib.bin_past_softmax.copy_(snap["bin_past_softmax"])
    self.current_state = self.get_robot_state()

  @torch.no_grad()
  def get_policy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened 10-step observation history (B, 930) and goal sequence g_t (B, 730)."""
    obs_flat = self.obs_history.reshape(self.num_envs, self.history_len * OBS_FRAME_DIM)
    goal_seq = self.motion_lib.get_goal_sequence(
      self.clip_ids,
      self.time_steps,
      self.current_state.root_pos_w,
      self.current_state.root_rot_w,
      self.default_joint_pos,
    )
    return obs_flat, goal_seq

  @torch.no_grad()
  def step(
    self, noisy_action: torch.Tensor, auto_reset: bool = True
  ) -> tuple[RobotState, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute noisy_action (B, 29) in simulation for decimation=4 substeps.
    Returns (next_state_before_reset, reward, terminated, timed_out).
    """
    # Joint setpoint q^sp = alpha * a_t + q^0 - encoder_bias (Appendix A & mjlab JointPositionAction)
    encoder_bias = self.sim_context[:, 1:30]
    target_q = noisy_action * self.action_scale + self.default_joint_pos - encoder_bias
    self.robot.set_joint_position_target(target_q)

    for _ in range(self.decimation):
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    # Refresh kinematics to get exact post-step state s_{t+1} BEFORE resetting done envs
    self.sim.forward()
    next_state = self.get_robot_state()

    # Advance reference frame index to t + 1
    next_time_steps = self.time_steps + 1
    flat_next_idx = self.motion_lib.get_global_indices(self.clip_ids, next_time_steps)

    ref_body_pos = self.motion_lib.body_pos_w[flat_next_idx]
    ref_body_rot = self.motion_lib.body_rot_w[flat_next_idx]
    ref_body_lvel = self.motion_lib.body_lin_vel_w[flat_next_idx]
    ref_body_avel = self.motion_lib.body_ang_vel_w[flat_next_idx]

    reward = compute_tracking_reward(
      next_state,
      noisy_action,
      self.last_action,
      ref_body_pos,
      ref_body_rot,
      ref_body_lvel,
      ref_body_avel,
      self.soft_joint_pos_limits,
      dt=self.step_dt,
    )

    _, _, terminated = compute_termination_errors(
      next_state, ref_body_pos[:, 0], ref_body_rot[:, 0]
    )
    clip_lens = self.motion_lib.clip_lengths[self.clip_ids]

    # Update episode return trackers
    self.episode_length += 1
    self.ep_disc_return += self.ep_gamma_pow * reward
    self.ep_undisc_return += reward
    self.ep_gamma_pow *= self.motion_lib.gamma

    timed_out = (next_time_steps >= (clip_lens - 1)) | (self.episode_length >= self.max_ep_steps)

    self.time_steps = next_time_steps
    self.last_action = noisy_action.clone()
    self.current_state = next_state

    # Update observation history for continuing environments
    clean_o = compute_clean_obs(next_state, self.sim_context, self.last_action, self.default_joint_pos)
    self.clean_obs = clean_o
    if self.play_mode:
      noisy_o = clean_o
    else:
      noisy_o = clean_o + sample_obs_noise(self.num_envs, self.device)
    self.obs_history = torch.cat((self.obs_history[:, 1:], noisy_o.unsqueeze(1)), dim=1)

    if auto_reset:
      done_mask = terminated | timed_out
      done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
      if done_ids.numel() > 0:
        # Bootstrap discounted return for timed_out (non-failed) episodes so end-of-clip / 10s episodes aren't penalized
        disc_ret = self.ep_disc_return[done_ids].clone()
        to_mask = timed_out[done_ids] & (~terminated[done_ids])
        if torch.any(to_mask):
          avg_step_r = self.ep_undisc_return[done_ids][to_mask] / self.episode_length[done_ids][to_mask].float().clamp(min=1.0)
          rem_disc = self.ep_gamma_pow[done_ids][to_mask] * avg_step_r / (1.0 - self.motion_lib.gamma)
          disc_ret[to_mask] = disc_ret[to_mask] + rem_disc
        self.motion_lib.update_bin_returns(self.reset_bins[done_ids], disc_ret)

        # Normalized return = undiscounted sum of per-step reward over episode / largest value this sum can take (§4)
        norm_rets = (
          self.ep_undisc_return[done_ids]
          / (5.0 * self.step_dt * self.max_ep_steps[done_ids].float().clamp(min=1.0))
        )
        for val in norm_rets.cpu().tolist():
          self.completed_ep_returns.append(val)
        if len(self.completed_ep_returns) > 2000:
          self.completed_ep_returns = self.completed_ep_returns[-2000:]

        self.reset_envs(done_ids)

    return next_state, reward, terminated, timed_out

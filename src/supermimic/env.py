"""mjlab environment subclass wiring the analytic SuperMimic MDP heads."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

from supermimic.env_state import EnvState
from supermimic.kinematics import Kinematics
from supermimic.mdp.tracking_math import TrackingMath


class SuperMimicEnv(ManagerBasedRlEnv):
  cfg: ManagerBasedRlEnvCfg

  def load_managers(self) -> None:
    robot = self.scene["robot"]
    motion_cfg = self.cfg.commands["motion"]
    body_names = list(motion_cfg.body_names)
    anchor = motion_cfg.anchor_body_name
    self.kinematics = Kinematics.from_mujoco(
      self.sim.mj_model,
      robot.body_names,
      body_names,
      robot.joint_names,
      anchor,
      prefix="robot/",
    ).to(self.device)
    self.tracking_math = TrackingMath(
      default_joint_pos=robot.data.default_joint_pos[0].clone(),
      soft_joint_pos_limits=robot.data.soft_joint_pos_limits[0].clone(),
      anchor_index=body_names.index(anchor),
      dt=self.step_dt,
    ).to(self.device)
    # Must exist before the managers: the command term and the startup domain
    # randomization events both read and write it.
    self.sm_state = EnvState(self, self.kinematics, self.tracking_math)
    super().load_managers()

  def step(self, action: torch.Tensor):
    """mjlab's step with the command advanced *before* reward/termination.

    The base env scores r_{t+1} against the reference frame the action was
    chosen on, which leaves the reward, the goal observation and the canonical
    state's tracking-error block one frame apart.  Advancing first makes
    r_{t+1} = R(s_{t+1}, ref_{t+1}, a_t, a_{t-1}) so the world model's analytic
    reward head reproduces the logged reward exactly.
    """
    self.extras["log"] = dict()
    self.action_manager.process_action(action.to(self.device))
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.command_manager.compute(dt=self.step_dt)
    self.sm_state.invalidate()

    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim(env_ids=reset_env_ids)
    self.sim.forward()
    # Every env already advanced above; this pass only refreshes command
    # metrics and clears the post-reset bookkeeping.
    self.command_manager.compute(dt=torch.zeros(self.num_envs, device=self.device))
    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)
    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def seed(self, seed: int) -> int:
    seed = super().seed(seed)
    torch.manual_seed(seed)
    return seed

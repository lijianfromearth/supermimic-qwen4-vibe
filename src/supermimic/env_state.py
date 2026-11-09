"""Simulator-side provider of the canonical state and the analytic MDP heads.

mjlab computes rewards and terminations before ``sim.forward()``, so the derived
``xpos``/``xquat``/``cvel`` fields lag qpos/qvel by one physics substep.  This
provider reads qpos/qvel directly and runs the differentiable FK, then routes
*everything* (observations, rewards, terminations) through the same canonical
representation the world model uses, so both sides agree by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from supermimic.kinematics import Kinematics
from supermimic.mdp.tracking_math import RefFrame, RobotFrame, TrackingMath
from supermimic.quat import quat_apply
from supermimic.state import (
  goal_from_future,
  pack_state,
  reconstruct_reference,
  robot_frame,
  sim_context,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from supermimic.mdp.motion_library import SuperMimicCommand


class EnvState:
  def __init__(
    self,
    env: "ManagerBasedRlEnv",
    kin: Kinematics,
    math: TrackingMath,
    command_name: str = "motion",
    entity_name: str = "robot",
  ) -> None:
    self.env = env
    self.kin = kin
    self.math = math
    self.command_name = command_name
    self.entity_name = entity_name
    self.friction = torch.ones(env.num_envs, device=env.device)
    self.com_offset = torch.zeros(env.num_envs, 3, device=env.device)
    self.com_default: torch.Tensor | None = None
    self._version = 0
    self._key: tuple[int, int] | None = None
    self._state: torch.Tensor | None = None
    self._robot: RobotFrame | None = None
    self._components: dict[str, torch.Tensor] | None = None
    self._terminated: torch.Tensor | None = None

  @property
  def cmd(self) -> "SuperMimicCommand":
    return self.env.command_manager.get_term(self.command_name)

  @property
  def robot_entity(self):
    return self.env.scene[self.entity_name]

  def invalidate(self) -> None:
    self._version += 1
    self._key = None
    self._components = None
    self._terminated = None

  def _refresh(self) -> None:
    key = (self.env.common_step_counter, self._version)
    if self._key == key and self._state is not None:
      return
    env = self.env
    robot = self.robot_entity
    idx = robot.indexing
    qpos = env.sim.data.qpos
    qvel = env.sim.data.qvel
    fq = idx.free_joint_q_adr
    fv = idx.free_joint_v_adr
    root_pos = qpos[:, fq[:3]]
    root_quat = qpos[:, fq[3:7]]
    root_lin = qvel[:, fv[:3]]
    # Free-joint qvel stores the angular part in the body frame.
    root_ang = quat_apply(root_quat, qvel[:, fv[3:6]])
    cmd = self.cmd
    self._state = pack_state(
      root_pos,
      root_quat,
      root_lin,
      root_ang,
      qpos[:, idx.joint_q_adr],
      qvel[:, idx.joint_v_adr],
      cmd.ref_anchor_pos_w,
      cmd.ref_anchor_quat_w,
    )
    self._robot = robot_frame(self._state, self.kin)
    self._key = key

  def state(self) -> torch.Tensor:
    self._refresh()
    return self._state

  def robot(self) -> RobotFrame:
    self._refresh()
    return self._robot

  def reference(self) -> RefFrame:
    return reconstruct_reference(self.cmd.current_canon(), self.state())

  def xi(self) -> torch.Tensor:
    return sim_context(
      self.friction, self.robot_entity.data.encoder_bias, self.com_offset
    )

  def prev_action(self) -> torch.Tensor:
    return self.env.action_manager.prev_action

  def action(self) -> torch.Tensor:
    return self.env.action_manager.action

  def observation(self) -> torch.Tensor:
    # Matches mjlab's ``last_action`` term: the action just applied, which is the
    # "previous action" from the point of view of the next decision.
    return self.math.proprio(
      self.robot(), self.robot_entity.data.encoder_bias, self.action()
    )

  def goal_features(self) -> torch.Tensor:
    s = self.state()
    robot = self.robot()
    cmd = self.cmd
    joint_pos, joint_vel, anchor_z = cmd.library.future_gather(
      cmd.clip, cmd.frame, cmd.offsets
    )
    return goal_from_future(
      robot.anchor_pos_w,
      robot.anchor_quat_w,
      s[:, 69:72],
      s[:, 72:76],
      joint_pos,
      joint_vel,
      anchor_z,
    )

  def reward(self) -> torch.Tensor:
    return self.math.reward(
      self.robot(), self.reference(), self.prev_action(), self.action()
    )

  def reward_components(self) -> dict[str, torch.Tensor]:
    self._refresh()
    if self._components is None:
      self._components = self.math.reward_components(
        self._robot, self.reference(), self.prev_action(), self.action()
      )
    return self._components

  def terminated(self) -> torch.Tensor:
    self._refresh()
    if self._terminated is None:
      self._terminated = self.math.terminated(self._robot, self.reference())
    return self._terminated

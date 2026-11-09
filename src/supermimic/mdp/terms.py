"""mjlab manager terms backed by the analytic SuperMimic heads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def proprioception(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return env.sm_state.observation()


def motion_goal(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return env.sm_state.cmd.future_features()


def reward_term(env: "ManagerBasedRlEnv", name: str) -> torch.Tensor:
  return env.sm_state.reward_components()[name]


def anchor_height(env: "ManagerBasedRlEnv") -> torch.Tensor:
  h, _ = env.tracking_math.termination_errors(
    env.sm_state.robot(), env.sm_state.reference()
  )
  return h > env.tracking_math.height_threshold


def anchor_tilt(env: "ManagerBasedRlEnv") -> torch.Tensor:
  _, t = env.tracking_math.termination_errors(
    env.sm_state.robot(), env.sm_state.reference()
  )
  return t > env.tracking_math.tilt_threshold


def motion_end(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return env.sm_state.cmd.at_clip_end()

"""Construct the SuperMimic models from a built environment."""

from __future__ import annotations

from supermimic.env import SuperMimicEnv
from supermimic.env_cfg import HISTORY
from supermimic.rl.agent import AgentCfg, SuperMimicAgent
from supermimic.rl.networks import (
  Critic,
  Dynamics,
  MotionEncoder,
  PolicyDecoder,
  Scales,
  TerminationHead,
)
from supermimic.rl.replay import ReplayBuffer
from supermimic.rl.world_model import WorldModel


def build(env: SuperMimicEnv, agent_cfg: AgentCfg, wm_kwargs: dict | None = None):
  device = env.device
  math = env.tracking_math
  scales = Scales(math.K, HISTORY).to(device)
  dynamics = Dynamics().to(device)
  term = TerminationHead().to(device)
  encoder = MotionEncoder(math.goal_dim).to(device)
  decoder = PolicyDecoder(32, math.obs_dim, HISTORY).to(device)
  critic = Critic(math.goal_dim).to(device)
  wm = WorldModel(
    dynamics,
    term,
    scales,
    env.kinematics,
    math,
    env.sm_state.cmd.library,
    **(wm_kwargs or {}),
  )
  agent = SuperMimicAgent(wm, encoder, decoder, critic, scales, HISTORY, agent_cfg)
  return scales, wm, agent


def build_replay(env: SuperMimicEnv, capacity: int = 1000) -> ReplayBuffer:
  return ReplayBuffer(capacity, env.num_envs, env.device)

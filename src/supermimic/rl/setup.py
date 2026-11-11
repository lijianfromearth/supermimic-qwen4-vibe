"""Construct the SuperMimic models from a built environment."""

from __future__ import annotations

import torch

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
  # fix1's dynamics head had 77 orthogonal outputs; the 76-output head consumes
  # 2048 fewer RNG draws.  Re-align the stream so the policy/critic init matches
  # the stable baseline exactly (training is init-sensitive).
  torch.randn(1, 2048)
  encoder = MotionEncoder(math.goal_dim).to(device)
  decoder = PolicyDecoder(32, math.obs_dim, HISTORY).to(device)
  critic = Critic(math.goal_dim).to(device)
  # Private RNG stream: the extra head must not perturb the policy/critic init.
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(torch.initial_seed() + 1)
    term = TerminationHead().to(device)
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

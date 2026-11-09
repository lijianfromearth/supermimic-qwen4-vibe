"""Domain-randomisation events that also record the sim context xi.

The paper (Table A.3) randomises foot friction, joint encoder calibration and
torso COM offset, and omits BeyondMimic's root-velocity pushes.  These wrappers
delegate the actual model writes to mjlab's DR helpers (so field expansion and
constant recomputation stay correct) and record the sampled values into
``env.sm_state`` for the world model.
"""

from __future__ import annotations

import torch
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg


def _resolve_env_ids(env, env_ids: torch.Tensor | None) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device)
  return env_ids


def foot_friction(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  ranges: tuple[float, float] = (0.3, 1.2),
) -> None:
  dr.geom_friction(
    env,
    env_ids,
    ranges=ranges,
    asset_cfg=asset_cfg,
    operation="abs",
    shared_random=True,
  )
  ids = _resolve_env_ids(env, env_ids)
  geom_ids = asset_cfg.geom_ids
  gid = int(geom_ids[0]) if not isinstance(geom_ids, slice) else 0
  env.sm_state.friction[ids] = env.sim.model.geom_friction[ids, gid, 0]


foot_friction.model_fields = dr.geom_friction.model_fields
foot_friction.recompute = dr.geom_friction.recompute


def encoder_bias(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  bias_range: tuple[float, float] = (-0.01, 0.01),
) -> None:
  dr.encoder_bias(env, env_ids, bias_range=bias_range, asset_cfg=asset_cfg)


encoder_bias.model_fields = getattr(dr.encoder_bias, "model_fields", ())
encoder_bias.recompute = getattr(dr.encoder_bias, "recompute", RecomputeLevel.none)


def torso_com(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  ranges: dict[int, tuple[float, float]] | None = None,
) -> None:
  if ranges is None:
    ranges = {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}
  sm = env.sm_state
  bid = int(asset_cfg.body_ids[0])
  if sm.com_default is None:
    sm.com_default = env.sim.model.body_ipos[:, bid].clone()
  dr.body_com_offset(env, env_ids, ranges=ranges, asset_cfg=asset_cfg, operation="add")
  ids = _resolve_env_ids(env, env_ids)
  sm.com_offset[ids] = env.sim.model.body_ipos[ids, bid] - sm.com_default[ids]


torso_com.model_fields = dr.body_com_offset.model_fields
torso_com.recompute = dr.body_com_offset.recompute


def eval_context(
  env,
  env_ids: torch.Tensor | None,
  geom_cfg: SceneEntityCfg,
  body_cfg: SceneEntityCfg,
) -> None:
  """Paper F.1: foot friction 1.0, no calibration error, nominal torso COM."""
  ids = _resolve_env_ids(env, env_ids)
  sm = env.sm_state
  dr.geom_friction(
    env,
    ids,
    ranges=(1.0, 1.0),
    asset_cfg=geom_cfg,
    operation="abs",
    shared_random=True,
  )
  sm.friction[ids] = 1.0
  bid = int(body_cfg.body_ids[0])
  if sm.com_default is not None:
    env.sim.model.body_ipos[ids, bid] = sm.com_default[ids]
    env.sim.recompute_constants(RecomputeLevel.set_const)
  sm.com_offset[ids] = 0.0
  env.scene["robot"].data.encoder_bias[ids] = 0.0
  sm.invalidate()


eval_context.model_fields = ("geom_friction", "body_ipos")
eval_context.recompute = RecomputeLevel.set_const

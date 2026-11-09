"""SuperMimic environment configuration (paper Appendix A, Tables A.1-A.3).

Built on mjlab's G1 flat tracking configuration, with the paper's deviations:
no self-collision reward/sensor, joint-limit weight -2, no end-effector
termination, no velocity pushes, and a single analytic proprioception term
stacked over the last 10 frames.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sim import SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from supermimic.mdp import events as sm_events
from supermimic.mdp import terms as sm_terms
from supermimic.mdp.motion_library import SuperMimicCommandCfg
from supermimic.mdp.tracking_math import REWARD_WEIGHTS
from supermimic.state import NUM_JOINTS

ANCHOR_BODY = "torso_link"
TRACKED_BODIES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
FOOT_GEOMS = r"^(left|right)_foot[1-7]_collision$"
HISTORY = 10


def proprio_noise() -> UniformNoiseCfg:
  """Table A.1: per-block uniform noise, no noise on the previous action."""
  lo = (
    (-0.01,) * NUM_JOINTS
    + (-0.5,) * NUM_JOINTS
    + (-0.2,) * 3
    + (-0.05,) * 3
    + (0.0,) * NUM_JOINTS
  )
  hi = tuple(-v for v in lo)
  return UniformNoiseCfg(n_min=lo, n_max=hi)


@dataclass(kw_only=True)
class SuperMimicSimCfg(SimulationCfg):
  """Silences solver overflow warnings before the CUDA graphs are captured.

  mjlab bakes ``warn_overflow`` into the solver kernels with ``wp.static`` and
  captures ``step``/``forward``/``reset`` graphs during ``Simulation.__init__``,
  so clearing the flag afterwards (e.g. in ``load_managers``) leaves the print
  recorded in the replayed graph.  Early training hits the iteration limits on
  most worlds, which otherwise emits hundreds of lines per step.
  """

  def apply_wp_opt(self, wp_opt) -> None:
    super().apply_wp_opt(wp_opt)
    wp_opt.warn_overflow = 0


def supermimic_env_cfg(
  motion_dir: str,
  num_envs: int = 4096,
  clips: list[str] | None = None,
  eval_mode: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_tracking_env_cfg()
  cfg.sim = SuperMimicSimCfg(
    **{f.name: getattr(cfg.sim, f.name) for f in fields(cfg.sim)}
  )
  cfg.scene.num_envs = num_envs
  cfg.scene.sensors = ()
  cfg.scene.terrain.env_spacing = 2.0

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "proprio": ObservationTermCfg(
          func=sm_terms.proprioception,
          noise=proprio_noise(),
          history_length=HISTORY,
          flatten_history_dim=True,
        )
      },
      concatenate_terms=True,
      enable_corruption=not eval_mode,
    ),
    "goal": ObservationGroupCfg(
      terms={"motion_goal": ObservationTermCfg(func=sm_terms.motion_goal)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE
  joint_pos_action.use_default_offset = True

  cfg.commands = {
    "motion": SuperMimicCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=False,
      motion_dir=motion_dir,
      anchor_body_name=ANCHOR_BODY,
      body_names=TRACKED_BODIES,
      clips=clips,
      num_future_frames=10,
      sampling_mode="start" if eval_mode else "return_based",
      pose_range={},
      velocity_range={},
      joint_position_range=(0.0, 0.0),
    )
  }

  foot_cfg = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)
  torso_cfg = SceneEntityCfg("robot", body_names=(ANCHOR_BODY,))
  cfg.events = {}
  if eval_mode:
    cfg.events["eval_context"] = EventTermCfg(
      mode="startup",
      func=sm_events.eval_context,
      params={"geom_cfg": foot_cfg, "body_cfg": torso_cfg},
    )
  else:
    cfg.events["foot_friction"] = EventTermCfg(
      mode="startup",
      func=sm_events.foot_friction,
      params={"asset_cfg": foot_cfg, "ranges": (0.3, 1.2)},
    )
    cfg.events["encoder_bias"] = EventTermCfg(
      mode="startup",
      func=sm_events.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    )
    cfg.events["torso_com"] = EventTermCfg(
      mode="startup",
      func=sm_events.torso_com,
      params={"asset_cfg": torso_cfg},
    )

  cfg.rewards = {
    name: RewardTermCfg(func=sm_terms.reward_term, weight=weight, params={"name": name})
    for name, weight in REWARD_WEIGHTS.items()
  }
  cfg.scale_rewards_by_dt = True

  cfg.terminations = {
    "anchor_height": TerminationTermCfg(func=sm_terms.anchor_height),
    "anchor_tilt": TerminationTermCfg(func=sm_terms.anchor_tilt),
    "motion_end": TerminationTermCfg(func=sm_terms.motion_end, time_out=True),
  }

  cfg.curriculum = {}
  cfg.metrics = {}
  cfg.recorders = {}
  # Clips run to their last frame; ``motion_end`` truncates instead.
  cfg.episode_length_s = 1.0e9
  cfg.viewer.body_name = ANCHOR_BODY
  return cfg

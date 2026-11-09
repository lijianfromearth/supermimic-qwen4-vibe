"""Multi-clip motion library and the SuperMimic motion command term.

mjlab's bundled MotionCommand loads a single clip; SuperMimic trains one policy
over a whole library, so clips are concatenated on the time axis and each
environment carries a (clip, frame) pair.  The library is stored in the
reference-canonical frame of :mod:`supermimic.state`.  Reset start points follow
the paper's return-based sampling (Appendix D.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from supermimic.quat import quat_apply, quat_from_euler_xyz, quat_mul, sample_uniform
from supermimic.state import CanonRef, canonicalize_reference, reference_advance

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

BODY_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)
CANON_KEYS = (
  "anchor_pos",
  "anchor_quat",
  "body_pos",
  "body_quat",
  "body_lin_vel",
  "body_ang_vel",
  "joint_pos",
  "joint_vel",
)


class MotionLibrary:
  def __init__(
    self,
    motion_dir: str,
    body_names: tuple[str, ...],
    robot_body_names: list[str],
    device: str,
    anchor_body_name: str,
    clips: list[str] | None = None,
  ) -> None:
    files = sorted(Path(motion_dir).glob("*.npz"))
    if clips is not None:
      keep = set(clips)
      files = [f for f in files if f.stem in keep]
    assert files, f"no motions in {motion_dir}"
    body_idx = torch.tensor([robot_body_names.index(n) for n in body_names])
    self.clip_names = [f.stem for f in files]
    self.device = device
    self.body_names = body_names
    self.num_bodies = len(body_names)
    self.anchor_index = body_names.index(anchor_body_name)

    parts: dict[str, list[torch.Tensor]] = {k: [] for k in CANON_KEYS}
    raw_pos: list[torch.Tensor] = []
    raw_quat: list[torch.Tensor] = []
    lengths = []
    for f in files:
      d = np.load(f)
      lengths.append(d["joint_pos"].shape[0])
      ai = self.anchor_index
      ap = torch.from_numpy(d["body_pos_w"][:, ai]).float()
      aq = torch.from_numpy(d["body_quat_w"][:, ai]).float()
      raw_pos.append(ap)
      raw_quat.append(aq)
      c = canonicalize_reference(
        anchor_pos=ap,
        anchor_quat=aq,
        body_pos=torch.from_numpy(d["body_pos_w"][:, body_idx]).float(),
        body_quat=torch.from_numpy(d["body_quat_w"][:, body_idx]).float(),
        body_lin_vel=torch.from_numpy(d["body_lin_vel_w"][:, body_idx]).float(),
        body_ang_vel=torch.from_numpy(d["body_ang_vel_w"][:, body_idx]).float(),
        joint_pos=torch.from_numpy(d["joint_pos"]).float(),
        joint_vel=torch.from_numpy(d["joint_vel"]).float(),
      )
      for k in CANON_KEYS:
        parts[k].append(getattr(c, k))
    self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    self.starts = torch.nn.functional.pad(self.lengths.cumsum(0)[:-1], (1, 0))
    self.num_clips = len(files)
    self.total_frames = int(self.lengths.sum())
    self.num_joints = parts["joint_pos"][0].shape[1]
    self.frame_clip = torch.repeat_interleave(
      torch.arange(self.num_clips, device=device), self.lengths
    )
    for k in CANON_KEYS:
      setattr(self, k, torch.cat(parts[k], dim=0).to(device))
    del parts
    # Raw anchor poses are needed for the advance: canonicalisation erases the
    # horizontal displacement between consecutive frames.
    self.raw_anchor_pos = torch.cat(raw_pos, dim=0).to(device)
    self.raw_anchor_quat = torch.cat(raw_quat, dim=0).to(device)
    self._build_advance()

  def _build_advance(self) -> None:
    """Per-frame exogenous reference advance, zero-padded at each clip end."""
    gi = torch.arange(self.total_frames, device=self.device)
    clip = self.frame_clip
    frame = gi - self.starts[clip]
    at_end = frame >= self.lengths[clip] - 1
    nxt = self.starts[clip] + torch.minimum(frame + 1, self.lengths[clip] - 1)
    adv = reference_advance(
      self.raw_anchor_pos[gi],
      self.raw_anchor_quat[gi],
      self.raw_anchor_pos[nxt],
      self.raw_anchor_quat[nxt],
    )
    adv[at_end] = 0.0
    adv[at_end, 4] = 1.0
    adv[at_end, 5:] = self.anchor_quat[gi[at_end]]
    self.ref_adv = adv

  def _gather(self, gi: torch.Tensor) -> CanonRef:
    return CanonRef(**{k: getattr(self, k)[gi] for k in CANON_KEYS})

  def global_index(self, clip: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    frame = torch.clamp(frame, min=0)
    frame = torch.minimum(frame, self.lengths[clip] - 1)
    return self.starts[clip] + frame

  def canon_at(self, clip: torch.Tensor, frame: torch.Tensor) -> CanonRef:
    return self._gather(self.global_index(clip, frame))

  def advance_at(self, clip: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    return self.ref_adv[self.global_index(clip, frame)]

  def future_gather(
    self, clip: torch.Tensor, frame: torch.Tensor, offsets: tuple[int, ...]
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Goal-relevant library fields for K future frames, clamped at the clip end.

    Returns ``(joint_pos, joint_vel, anchor_z)`` shaped (B, K, nj), (B, K, nj),
    (B, K) — exactly the inputs :func:`supermimic.state.goal_from_future` needs.
    """
    o = torch.as_tensor(offsets, dtype=torch.long, device=self.device)
    last = (self.lengths[clip] - 1)[:, None]
    f = torch.minimum(frame[:, None] + o, last).clamp_min(0)
    gi = self.starts[clip][:, None] + f
    return self.joint_pos[gi], self.joint_vel[gi], self.anchor_pos[gi, 2]


@dataclass(kw_only=True)
class SuperMimicCommandCfg(CommandTermCfg):
  motion_dir: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str = "robot"
  clips: list[str] | None = None
  eval_clips: list[int] | None = None
  num_future_frames: int = 10
  future_offsets: tuple[int, ...] | None = None
  sampling_mode: Literal["return_based", "uniform", "start"] = "return_based"
  bin_seconds: float = 1.0
  uniform_ratio: float = 0.5
  hard_ratio: float = 0.4
  history_ratio: float = 0.1
  softmax_scale: float = 25.0
  bin_return_ema: float = 0.1
  past_weight_ema: float = 0.001
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (0.0, 0.0)
  gamma: float = 0.99
  max_reward_per_step: float = 0.1

  def build(self, env: "ManagerBasedRlEnv") -> "SuperMimicCommand":
    return SuperMimicCommand(self, env)


class SuperMimicCommand(CommandTerm):
  cfg: SuperMimicCommandCfg

  def __init__(self, cfg: SuperMimicCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.library = MotionLibrary(
      cfg.motion_dir,
      cfg.body_names,
      self.robot.body_names,
      self.device,
      cfg.anchor_body_name,
      cfg.clips,
    )
    self.anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.clip = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # The library is stored per-frame canonicalised, so the reference's world
    # pose is integrated from ``ref_adv`` starting at the reset frame.
    self.ref_yaw = torch.zeros(self.num_envs, device=self.device)
    self.ref_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self._advance_mask: torch.Tensor | None = None
    self._resampled_this_step = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.offsets = cfg.future_offsets or tuple(range(cfg.num_future_frames))

    lib = self.library
    self.bin_size = max(1, round(cfg.bin_seconds / env.step_dt))
    self.bin_count = (lib.total_frames + self.bin_size - 1) // self.bin_size
    self.frame_bin = torch.arange(lib.total_frames, device=self.device) // self.bin_size
    self.max_return = cfg.max_reward_per_step / (1.0 - cfg.gamma)
    self.bin_return = torch.full((self.bin_count,), self.max_return, device=self.device)
    self.past_weight = torch.full(
      (self.bin_count,), 1.0 / self.bin_count, device=self.device
    )
    self.metrics["bin_return_mean"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["clip"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["frame"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.future_features()

  @property
  def env_origins(self) -> torch.Tensor:
    return self._env.scene.env_origins

  def current_canon(self) -> CanonRef:
    return self.library.canon_at(self.clip, self.frame)

  @property
  def ref_anchor_quat_w(self) -> torch.Tensor:
    zero = torch.zeros_like(self.ref_yaw)
    yaw_q = quat_from_euler_xyz(zero, zero, self.ref_yaw)
    return quat_mul(yaw_q, self.current_canon().anchor_quat)

  def current_adv(self) -> torch.Tensor:
    return self.library.advance_at(self.clip, self.frame)

  def future_features(self) -> torch.Tensor:
    return self._env.sm_state.goal_features()

  def at_clip_end(self) -> torch.Tensor:
    return self.frame >= self.library.lengths[self.clip] - 1

  def compute(self, dt, env_ids=None) -> None:
    self._advance_mask = (dt > 0) if isinstance(dt, torch.Tensor) else None
    super().compute(dt, env_ids)

  def _update_metrics(self) -> None:
    self.metrics["bin_return_mean"][:] = self.bin_return.mean()
    self.metrics["clip"][:] = self.clip.float().mean()
    self.metrics["frame"][:] = self.frame.float().mean()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    if self.cfg.eval_clips is not None:
      table = torch.tensor(self.cfg.eval_clips, dtype=torch.long, device=self.device)
      clip = table[env_ids]
      frame = torch.zeros_like(clip)
    else:
      clip, frame = self.sample_start(len(env_ids))
    self.reset_to_frame(env_ids, clip, frame)

  def sample_start(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    mode = self.cfg.sampling_mode
    if mode == "start":
      z = torch.zeros(n, dtype=torch.long, device=self.device)
      return z, z.clone()
    if mode == "uniform":
      g = torch.randint(0, self.library.total_frames, (n,), device=self.device)
    else:
      bin_id = self._sample_bins(n)
      lo = bin_id * self.bin_size
      hi = (lo + self.bin_size).clamp(max=self.library.total_frames)
      g = lo + (torch.rand(n, device=self.device) * (hi - lo).float()).long()
      g = g.clamp(max=self.library.total_frames - 1)
    clip = self.library.frame_clip[g]
    return clip, g - self.library.starts[clip]

  def _sample_bins(self, n: int) -> torch.Tensor:
    cfg = self.cfg
    hard = torch.softmax(-cfg.softmax_scale * self.bin_return / self.max_return, dim=0)
    mix = (
      cfg.uniform_ratio / self.bin_count
      + cfg.hard_ratio * hard
      + cfg.history_ratio * self.past_weight
    )
    out = torch.multinomial(mix / mix.sum(), n, replacement=True)
    self.past_weight = (
      cfg.past_weight_ema * hard + (1 - cfg.past_weight_ema) * self.past_weight
    )
    return out

  def update_bin_returns(
    self,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    clip: torch.Tensor | None = None,
    frame: torch.Tensor | None = None,
  ) -> None:
    """Bootstrap the per-bin discounted return along the policy's own rollouts.

    ``clip``/``frame`` override the term's current pair so the caller can pass
    the frame the action was chosen on, which the step has already advanced.
    """
    clip = self.clip if clip is None else clip
    frame = self.frame if frame is None else frame
    gi = self.library.global_index(clip, frame)
    b = self.frame_bin[gi]
    b_next = self.frame_bin[self.library.global_index(clip, frame + 1)]
    alive = (~terminated) & (frame < self.library.lengths[clip] - 1)
    target = reward + self.cfg.gamma * alive.float() * self.bin_return[b_next]
    count = torch.zeros(self.bin_count, device=self.device)
    total = torch.zeros(self.bin_count, device=self.device)
    count.index_add_(0, b, torch.ones_like(reward))
    total.index_add_(0, b, target)
    seen = count > 0
    a = self.cfg.bin_return_ema
    self.bin_return[seen] = (1 - a) * self.bin_return[seen] + a * (
      total[seen] / count[seen]
    )

  def reset_to_frame(
    self, env_ids: torch.Tensor, clip: torch.Tensor, frame: torch.Tensor
  ) -> None:
    self.clip[env_ids] = clip
    self.frame[env_ids] = frame
    self.episode_step[env_ids] = 0
    canon = self.library.canon_at(clip, frame)
    self.ref_yaw[env_ids] = 0.0
    self.ref_anchor_pos_w[env_ids] = self.env_origins[env_ids] + canon.anchor_pos
    self._resampled_this_step[env_ids] = True
    self._write_reference_to_sim(env_ids)

  def _write_reference_to_sim(self, env_ids: torch.Tensor) -> None:
    ref = self.library.canon_at(self.clip[env_ids], self.frame[env_ids])
    origins = self.env_origins[env_ids]
    # Body 0 is the pelvis (root).  Placing the reference anchor at the env
    # origin with zero yaw makes the robot state equal the reference state.
    root_pos = ref.body_pos[:, 0] + origins
    root_ori = ref.body_quat[:, 0].clone()
    root_lin = ref.body_lin_vel[:, 0].clone()
    root_ang = ref.body_ang_vel[:, 0].clone()
    joint_pos = ref.joint_pos.clone()
    joint_vel = ref.joint_vel.clone()

    cfg = self.cfg
    n = len(env_ids)
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    if cfg.pose_range:
      ranges = torch.tensor(
        [cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device
      )
      s = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), self.device)
      root_pos = root_pos + s[:, 0:3]
      root_ori = quat_mul(quat_from_euler_xyz(s[:, 3], s[:, 4], s[:, 5]), root_ori)
    if cfg.velocity_range:
      ranges = torch.tensor(
        [cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device
      )
      s = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), self.device)
      root_lin = root_lin + s[:, :3]
      root_ang = root_ang + s[:, 3:]
    if cfg.joint_position_range != (0.0, 0.0):
      joint_pos = joint_pos + sample_uniform(
        cfg.joint_position_range[0],
        cfg.joint_position_range[1],
        joint_pos.shape,
        self.device,
      )
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self.robot.write_root_state_to_sim(
      torch.cat([root_pos, root_ori, root_lin, root_ang], dim=-1), env_ids=env_ids
    )
    self.robot.reset(env_ids=env_ids)
    self._env.sm_state.invalidate()

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    # env_ids marks the explicit reset() path: the robot was just written at
    # ``frame``, so the reference must not advance past it.
    if env_ids is not None:
      self._resampled_this_step[env_ids] = False
      return
    step = (
      torch.ones_like(self.frame)
      if self._advance_mask is None
      else self._advance_mask.long()
    )
    step = step * (~self._resampled_this_step).long()
    adv = self.current_adv()
    zero = torch.zeros_like(self.ref_yaw)
    yaw_q = quat_from_euler_xyz(zero, zero, self.ref_yaw)
    self.ref_anchor_pos_w += quat_apply(yaw_q, adv[:, :3]) * step[:, None].float()
    self.ref_yaw += torch.atan2(adv[:, 3], adv[:, 4]) * step.float()
    self.frame += step
    self.episode_step += step
    self._resampled_this_step[:] = False

"""Differentiable forward kinematics for the G1, in pure PyTorch.

The world model predicts state only; observation, reward and termination heads
are analytic functions of that state (the paper allows analytic heads, which
then carry no loss term).  This module reproduces MuJoCo's ``xpos``/``xquat``
and the body-frame-origin velocity that mjlab exposes as ``body_link_*_w``, so
one implementation serves both the simulator side and imagined rollouts.

Joint vectors are *hinge-only*, ordered by ``joint_names`` (which must match
mjlab's ``robot.joint_names`` and the motion library's ``joint_pos`` columns).
Root velocity is always world frame; the free-joint body-frame angular velocity
conversion lives in :mod:`supermimic.state`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch

from supermimic.quat import normalize, quat_apply, quat_mul


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  """axis (..., 3) unit, angle (...,) -> quaternion wxyz (..., 4)."""
  half = 0.5 * angle
  s = torch.sin(half).unsqueeze(-1)
  c = torch.cos(half).unsqueeze(-1)
  return torch.cat([c, axis * s], dim=-1)


@dataclass
class Kinematics:
  """Topologically ordered hinge tree rooted at a free-floating body."""

  body_names: list[str]
  joint_names: list[str]
  parent: torch.Tensor  # (nb,) long, -1 for root
  local_pos: torch.Tensor  # (nb, 3)
  local_quat: torch.Tensor  # (nb, 4) wxyz
  joint_axis: torch.Tensor  # (nb, 3) zero for root
  hinge_index: torch.Tensor  # (nb,) long, -1 for root
  levels: list[torch.Tensor]  # body indices per depth
  tracked: torch.Tensor  # (nt,) long indices into the body list
  tracked_anchor: int  # index of the anchor body within ``tracked``
  root: int
  free_q_adr: int
  free_v_adr: int
  hinge_q_adr: torch.Tensor  # (nh,) raw qpos addresses, in joint_names order
  hinge_v_adr: torch.Tensor  # (nh,) raw qvel addresses

  @classmethod
  def from_mujoco(
    cls,
    model: mujoco.MjModel,
    body_names: list[str],
    tracked_names: list[str],
    joint_names: list[str],
    anchor_name: str,
    prefix: str = "",
  ) -> "Kinematics":
    ids = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + n) for n in body_names
    ]
    assert all(i >= 0 for i in ids), body_names
    id_set = set(ids)
    nb = len(ids)
    parent = torch.full((nb,), -1, dtype=torch.long)
    local_pos = torch.zeros(nb, 3)
    local_quat = torch.zeros(nb, 4)
    local_quat[:, 0] = 1.0
    joint_axis = torch.zeros(nb, 3)
    hinge_index = torch.full((nb,), -1, dtype=torch.long)

    jids = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)
      for n in joint_names
    ]
    assert all(i >= 0 for i in jids), joint_names
    hinge_q_adr = torch.tensor([model.jnt_qposadr[j] for j in jids], dtype=torch.long)
    hinge_v_adr = torch.tensor([model.jnt_dofadr[j] for j in jids], dtype=torch.long)
    assert all(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in jids), (
      joint_names
    )
    adr_to_hinge = {int(a): i for i, a in enumerate(hinge_q_adr.tolist())}

    roots = []
    free_joints = []
    for k, bid in enumerate(ids):
      pid = int(model.body_parentid[bid])
      if pid in id_set:
        parent[k] = ids.index(pid)
      else:
        roots.append(k)
      local_pos[k] = torch.tensor(model.body_pos[bid], dtype=torch.float32)
      local_quat[k] = torch.tensor(model.body_quat[bid], dtype=torch.float32)
      jadr = int(model.body_jntadr[bid])
      jnum = int(model.body_jntnum[bid])
      assert jnum <= 1, f"body {body_names[k]} has {jnum} joints"
      if jnum == 1:
        jtype = int(model.jnt_type[jadr])
        if jtype == mujoco.mjtJoint.mjJNT_HINGE:
          joint_axis[k] = torch.tensor(model.jnt_axis[jadr], dtype=torch.float32)
          hinge_index[k] = adr_to_hinge[int(model.jnt_qposadr[jadr])]
        elif jtype == mujoco.mjtJoint.mjJNT_FREE:
          free_joints.append((k, jadr))
        else:
          raise AssertionError(f"unsupported joint type {jtype} on {body_names[k]}")
    assert len(roots) == 1, roots
    assert len(free_joints) == 1, free_joints
    root = roots[0]
    assert free_joints[0][0] == root
    assert int(hinge_index[root]) == -1
    assert all(int(h) >= 0 for k, h in enumerate(hinge_index) if k != root)
    free_jadr = free_joints[0][1]

    depth = [-1] * nb
    depth[root] = 0
    for _ in range(nb):
      changed = False
      for k in range(nb):
        p = int(parent[k])
        if depth[k] < 0 and p >= 0 and depth[p] >= 0:
          depth[k] = depth[p] + 1
          changed = True
      if not changed:
        break
    assert all(d >= 0 for d in depth), depth
    levels = [
      torch.tensor([k for k in range(nb) if depth[k] == d], dtype=torch.long)
      for d in range(max(depth) + 1)
    ]
    tracked = torch.tensor(
      [body_names.index(n) for n in tracked_names], dtype=torch.long
    )
    return cls(
      body_names=body_names,
      joint_names=joint_names,
      parent=parent,
      local_pos=local_pos,
      local_quat=local_quat,
      joint_axis=joint_axis,
      hinge_index=hinge_index,
      levels=levels,
      tracked=tracked,
      tracked_anchor=tracked_names.index(anchor_name),
      root=root,
      free_q_adr=int(model.jnt_qposadr[free_jadr]),
      free_v_adr=int(model.jnt_dofadr[free_jadr]),
      hinge_q_adr=hinge_q_adr,
      hinge_v_adr=hinge_v_adr,
    )

  def to(self, device: str) -> "Kinematics":
    for f in (
      "parent",
      "local_pos",
      "local_quat",
      "joint_axis",
      "hinge_index",
      "tracked",
      "hinge_q_adr",
      "hinge_v_adr",
    ):
      setattr(self, f, getattr(self, f).to(device))
    self.levels = [lv.to(device) for lv in self.levels]
    return self

  @property
  def num_joints(self) -> int:
    return len(self.joint_names)

  @property
  def num_bodies(self) -> int:
    return len(self.body_names)

  @property
  def num_tracked(self) -> int:
    return int(self.tracked.numel())

  def forward(
    self, root_pos: torch.Tensor, root_quat: torch.Tensor, q: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World (xpos, xquat, joint-axis-in-world) for every body.

    root_pos (B,3), root_quat (B,4) wxyz, q (B,nh) hinge angles.
    """
    B = root_pos.shape[0]
    nb = len(self.body_names)
    pos = root_pos.new_zeros(B, nb, 3)
    quat = root_quat.new_zeros(B, nb, 4)
    axis_w = root_pos.new_zeros(B, nb, 3)
    pos[:, self.root] = root_pos
    quat[:, self.root] = normalize(root_quat)
    for level in self.levels[1:]:
      p = self.parent[level]
      pq = quat[:, p]
      pre_q = quat_mul(pq, self.local_quat[level])
      angle = q[:, self.hinge_index[level]]
      jq = quat_from_axis_angle(self.joint_axis[level], angle)
      quat[:, level] = quat_mul(pre_q, jq)
      pos[:, level] = pos[:, p] + quat_apply(pq, self.local_pos[level])
      axis_w[:, level] = quat_apply(pre_q, self.joint_axis[level])
    return pos, quat, axis_w

  def velocities(
    self,
    pos: torch.Tensor,
    axis_w: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    qd: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Body-frame-origin linear velocity and angular velocity, world frame."""
    B = pos.shape[0]
    nb = pos.shape[1]
    lin = pos.new_zeros(B, nb, 3)
    ang = pos.new_zeros(B, nb, 3)
    lin[:, self.root] = root_lin_vel
    ang[:, self.root] = root_ang_vel
    for level in self.levels[1:]:
      p = self.parent[level]
      w_p = ang[:, p]
      idx = self.hinge_index[level]
      ang[:, level] = w_p + axis_w[:, level] * qd[:, idx].unsqueeze(-1)
      r = pos[:, level] - pos[:, p]
      lin[:, level] = lin[:, p] + torch.linalg.cross(w_p, r)
    return lin, ang

  def fk(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tracked-body (pos, quat, lin_vel, ang_vel), all world frame."""
    pos, quat, axis_w = self.forward(root_pos, root_quat, q)
    lin, ang = self.velocities(pos, axis_w, root_lin_vel, root_ang_vel, qd)
    t = self.tracked
    return pos[:, t], normalize(quat[:, t]), lin[:, t], ang[:, t]

  def fk_all(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as :meth:`fk` but for every body (used for validation)."""
    pos, quat, axis_w = self.forward(root_pos, root_quat, q)
    lin, ang = self.velocities(pos, axis_w, root_lin_vel, root_ang_vel, qd)
    return pos, normalize(quat), lin, ang

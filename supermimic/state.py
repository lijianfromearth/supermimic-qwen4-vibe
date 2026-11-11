"""State representation, canonical frame transforms, and analytic heads (obs, reward)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.math_utils import (
  compute_relative_body_poses_rot,
  extract_yaw_cos_sin,
  geodesic_rot_error_sq,
  matrix_to_rot6d,
  rot6d_to_matrix,
  rotate_mat_z,
  rotate_vec_z,
)

# Predefined nominal errors (Table A.2 & Figure 3c)
NOMINAL_POS_ERR = 0.3
NOMINAL_ORI_ERR = 0.4
NOMINAL_LIN_VEL_ERR = 1.0
NOMINAL_ANG_VEL_ERR = 3.14
NOMINAL_JOINT_POS_ERR = 0.3
NOMINAL_JOINT_VEL_ERR = 3.14

CANONICAL_STATE_DIM = 266  # 1 + 39 + 84 + 42 + 42 + 29 + 29
PACKED_STATE_DIM = 268     # 42 + 84 + 42 + 42 + 29 + 29
OBS_FRAME_DIM = 93         # 29 + 29 + 3 + 3 + 29
SIM_CONTEXT_DIM = 33       # 1 (friction) + 29 (encoder_bias) + 3 (com_offset)


@dataclass
class RobotState:
  """World-frame state of the 14 tracked bodies and 29 joints."""

  body_pos_w: torch.Tensor      # (B, 14, 3), index 0 is root ('pelvis')
  body_rot_w: torch.Tensor      # (B, 14, 3, 3)
  body_lin_vel_w: torch.Tensor  # (B, 14, 3)
  body_ang_vel_w: torch.Tensor  # (B, 14, 3)
  joint_pos: torch.Tensor       # (B, 29)
  joint_vel: torch.Tensor       # (B, 29)

  @property
  def root_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, 0]

  @property
  def root_rot_w(self) -> torch.Tensor:
    return self.body_rot_w[:, 0]

  @property
  def root_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, 0]

  @property
  def root_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, 0]


def pack_state(state: RobotState) -> torch.Tensor:
  """Pack RobotState into a flat tensor of shape (..., 268) for replay buffer storage."""
  batch_shape = state.joint_pos.shape[:-1]
  b_pos = state.body_pos_w.reshape(*batch_shape, 42)
  b_rot6d = matrix_to_rot6d(state.body_rot_w).reshape(*batch_shape, 84)
  b_lvel = state.body_lin_vel_w.reshape(*batch_shape, 42)
  b_avel = state.body_ang_vel_w.reshape(*batch_shape, 42)
  return torch.cat((b_pos, b_rot6d, b_lvel, b_avel, state.joint_pos, state.joint_vel), dim=-1)


def unpack_state(packed: torch.Tensor) -> RobotState:
  """Unpack flat tensor (..., 268) into RobotState."""
  batch_shape = packed.shape[:-1]
  b_pos = packed[..., 0:42].reshape(*batch_shape, 14, 3)
  b_rot6d = packed[..., 42:126].reshape(*batch_shape, 14, 6)
  b_lvel = packed[..., 126:168].reshape(*batch_shape, 14, 3)
  b_avel = packed[..., 168:210].reshape(*batch_shape, 14, 3)
  j_pos = packed[..., 210:239]
  j_vel = packed[..., 239:268]
  # Reconstruct 3rd column of orthogonal rotation matrix via cross product
  mat32 = b_rot6d.reshape(*batch_shape, 14, 3, 2)
  c1 = mat32[..., 0]
  c2 = mat32[..., 1]
  c3 = torch.linalg.cross(c1, c2, dim=-1)
  b_rot = torch.stack((c1, c2, c3), dim=-1)
  return RobotState(
    body_pos_w=b_pos,
    body_rot_w=b_rot,
    body_lin_vel_w=b_lvel,
    body_ang_vel_w=b_avel,
    joint_pos=j_pos,
    joint_vel=j_vel,
  )


def state_to_canonical(
  state: RobotState, default_joint_pos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Convert world-frame RobotState to SE(2)-invariant heading-frame vector s_tilde (B, 266).
  Also returns (cos_yaw, sin_yaw) of the current root heading.
  """
  B = state.joint_pos.shape[0]
  root_pos = state.root_pos_w  # (B, 3)
  root_rot = state.root_rot_w  # (B, 3, 3)
  cos_yaw, sin_yaw = extract_yaw_cos_sin(root_rot)  # (B,)

  # Rotate into heading frame by -yaw: (cos_yaw, -sin_yaw)
  c_inv = cos_yaw.unsqueeze(1)
  s_inv = -sin_yaw.unsqueeze(1)

  root_z = root_pos[:, 2:3]  # (B, 1)
  rel_pos_w = state.body_pos_w[:, 1:] - root_pos.unsqueeze(1)  # (B, 13, 3)
  rel_pos_h = rotate_vec_z(rel_pos_w, c_inv, s_inv).reshape(B, 39)

  rot_h = rotate_mat_z(state.body_rot_w, c_inv, s_inv)  # (B, 14, 3, 3)
  rot6d_h = matrix_to_rot6d(rot_h).reshape(B, 84)

  lvel_h = rotate_vec_z(state.body_lin_vel_w, c_inv, s_inv).reshape(B, 42)
  avel_h = rotate_vec_z(state.body_ang_vel_w, c_inv, s_inv).reshape(B, 42)

  q_rel = state.joint_pos - default_joint_pos.view(1, 29)
  qd = state.joint_vel

  s_tilde = torch.cat((root_z, rel_pos_h, rot6d_h, lvel_h, avel_h, q_rel, qd), dim=-1)
  return s_tilde, cos_yaw, sin_yaw


def apply_canonical_delta(
  state: RobotState,
  s_tilde: torch.Tensor,
  cos_yaw: torch.Tensor,
  sin_yaw: torch.Tensor,
  delta: torch.Tensor,
  default_joint_pos: torch.Tensor,
) -> RobotState:
  """Apply predicted heading-frame residual delta (B, 268) to advance RobotState from t to t+1."""
  B = s_tilde.shape[0]
  delta_xy_h = delta[:, 0:2]  # (B, 2)
  next_s_h = s_tilde + delta[:, 2:268]  # (B, 266)

  next_z = next_s_h[:, 0:1]
  rel_pos_h = next_s_h[:, 1:40].reshape(B, 13, 3)
  rot6d_h = next_s_h[:, 40:124].reshape(B, 14, 6)
  lvel_h = next_s_h[:, 124:166].reshape(B, 14, 3)
  avel_h = next_s_h[:, 166:208].reshape(B, 14, 3)
  q_rel = next_s_h[:, 208:237]
  qd = next_s_h[:, 237:266]

  # Rotate from heading frame F_t back to world frame by +yaw: (cos_yaw, sin_yaw)
  root_pos = state.root_pos_w
  dx_w = cos_yaw * delta_xy_h[:, 0] - sin_yaw * delta_xy_h[:, 1]
  dy_w = sin_yaw * delta_xy_h[:, 0] + cos_yaw * delta_xy_h[:, 1]
  next_root_pos = torch.stack(
    (root_pos[:, 0] + dx_w, root_pos[:, 1] + dy_w, next_z.squeeze(-1)), dim=-1
  )  # (B, 3)

  c = cos_yaw.unsqueeze(1)
  s = sin_yaw.unsqueeze(1)
  next_other_pos = next_root_pos.unsqueeze(1) + rotate_vec_z(rel_pos_h, c, s)
  next_body_pos = torch.cat((next_root_pos.unsqueeze(1), next_other_pos), dim=1)

  rot_h = rot6d_to_matrix(rot6d_h)  # (B, 14, 3, 3)
  next_body_rot = rotate_mat_z(rot_h, c, s)
  next_body_lvel = rotate_vec_z(lvel_h, c, s)
  next_body_avel = rotate_vec_z(avel_h, c, s)

  next_q = q_rel + default_joint_pos.view(1, 29)
  return RobotState(
    body_pos_w=next_body_pos,
    body_rot_w=next_body_rot,
    body_lin_vel_w=next_body_lvel,
    body_ang_vel_w=next_body_avel,
    joint_pos=next_q,
    joint_vel=qd,
  )


def compute_target_delta(
  curr_state: RobotState,
  curr_s_tilde: torch.Tensor,
  cos_yaw: torch.Tensor,
  sin_yaw: torch.Tensor,
  target_next_state: RobotState,
  default_joint_pos: torch.Tensor,
) -> torch.Tensor:
  """Compute exact heading-frame residual delta (B, 268) that maps curr_state to target_next_state."""
  B = curr_s_tilde.shape[0]
  dp_root = target_next_state.root_pos_w - curr_state.root_pos_w
  dx_h = cos_yaw * dp_root[:, 0] + sin_yaw * dp_root[:, 1]
  dy_h = -sin_yaw * dp_root[:, 0] + cos_yaw * dp_root[:, 1]
  delta_xy_h = torch.stack((dx_h, dy_h), dim=-1)

  c_inv = cos_yaw.unsqueeze(1)
  s_inv = -sin_yaw.unsqueeze(1)

  next_z = target_next_state.root_pos_w[:, 2:3]
  rel_pos_w = target_next_state.body_pos_w[:, 1:] - target_next_state.root_pos_w.unsqueeze(1)
  rel_pos_h = rotate_vec_z(rel_pos_w, c_inv, s_inv).reshape(B, 39)

  rot_h = rotate_mat_z(target_next_state.body_rot_w, c_inv, s_inv)
  rot6d_h = matrix_to_rot6d(rot_h).reshape(B, 84)

  lvel_h = rotate_vec_z(target_next_state.body_lin_vel_w, c_inv, s_inv).reshape(B, 42)
  avel_h = rotate_vec_z(target_next_state.body_ang_vel_w, c_inv, s_inv).reshape(B, 42)

  q_rel = target_next_state.joint_pos - default_joint_pos.view(1, 29)
  qd = target_next_state.joint_vel

  next_s_h = torch.cat((next_z, rel_pos_h, rot6d_h, lvel_h, avel_h, q_rel, qd), dim=-1)
  delta_s_h = next_s_h - curr_s_tilde
  return torch.cat((delta_xy_h, delta_s_h), dim=-1)


def compute_state_loss(pred: RobotState, target: RobotState) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute normalized state prediction loss ||(s_hat - s) / sigma||^2 per sample (B,).
  Returns:
    loss_per_sample: (B,) sum of squared normalized errors
    mean_norm_err: (B,) mean per-dimension normalized squared error (matches Fig 3c metric)
  """
  pos_err = torch.square((pred.body_pos_w - target.body_pos_w) / NOMINAL_POS_ERR).flatten(1)
  ori_err = geodesic_rot_error_sq(pred.body_rot_w, target.body_rot_w) / (NOMINAL_ORI_ERR**2)
  lvel_err = torch.square((pred.body_lin_vel_w - target.body_lin_vel_w) / NOMINAL_LIN_VEL_ERR).flatten(1)
  avel_err = torch.square((pred.body_ang_vel_w - target.body_ang_vel_w) / NOMINAL_ANG_VEL_ERR).flatten(1)
  q_err = torch.square((pred.joint_pos - target.joint_pos) / NOMINAL_JOINT_POS_ERR)
  qd_err = torch.square((pred.joint_vel - target.joint_vel) / NOMINAL_JOINT_VEL_ERR)

  all_errs = torch.cat((pos_err, ori_err, lvel_err, avel_err, q_err, qd_err), dim=-1)
  return all_errs.mean(dim=-1), all_errs.mean(dim=-1)


def compute_clean_obs(
  state: RobotState,
  sim_context: torch.Tensor,
  last_action: torch.Tensor,
  default_joint_pos: torch.Tensor,
) -> torch.Tensor:
  """Analytic observation head g_omega(s_t, xi) -> o_bar_t (B, 93) (Table A.1).
  Terms:
    1. Joint positions q - q^0 + delta_q^0 (29)
    2. Joint velocities q_dot (29)
    3. Base angular velocity (IMU in pelvis frame) (3)
    4. Projected gravity in pelvis frame (3)
    5. Previous action a_{t-1} (29)
  """
  encoder_bias = sim_context[:, 1:30]
  q_rel_biased = state.joint_pos - default_joint_pos.view(1, 29) + encoder_bias
  qd = state.joint_vel

  # Pelvis rotation R_root: (B, 3, 3)
  R_root = state.root_rot_w
  # Base angular velocity in pelvis frame: R_root^T @ omega_root_w
  omega_b = torch.matmul(R_root.transpose(-1, -2), state.root_ang_vel_w.unsqueeze(-1)).squeeze(-1)
  # Projected gravity in pelvis frame: R_root^T @ [0, 0, -1]^T = -R_root[:, 2, :]
  proj_grav_b = -R_root[:, 2, :]

  return torch.cat((q_rel_biased, qd, omega_b, proj_grav_b, last_action), dim=-1)


def sample_obs_noise(batch_size: int, device: str | torch.device) -> torch.Tensor:
  """Sample per-dimension uniform sensor noise eta_t ~ rho_eta (Table A.1)."""
  noise = torch.empty((batch_size, OBS_FRAME_DIM), dtype=torch.float32, device=device)
  noise[:, 0:29].uniform_(-0.01, 0.01)   # q - q^0
  noise[:, 29:58].uniform_(-0.5, 0.5)    # q_dot
  noise[:, 58:61].uniform_(-0.2, 0.2)    # base_ang_vel
  noise[:, 61:64].uniform_(-0.05, 0.05)  # projected_gravity
  noise[:, 64:93].zero_()                # previous action
  return noise


def compute_tracking_reward(
  state: RobotState,
  action: torch.Tensor,
  prev_action: torch.Tensor,
  ref_body_pos_w: torch.Tensor,
  ref_body_rot_w: torch.Tensor,
  ref_body_lin_vel_w: torch.Tensor,
  ref_body_ang_vel_w: torch.Tensor,
  soft_joint_pos_limits: torch.Tensor,
  dt: float = 0.02,
) -> torch.Tensor:
  """Analytic motion-tracking reward head r_omega (Table A.2), scaled by dt = 0.02s.
  All inputs have batch size B.
  """
  ref_root_pos = ref_body_pos_w[:, 0]
  ref_root_rot = ref_body_rot_w[:, 0]
  rob_root_pos = state.root_pos_w
  rob_root_rot = state.root_rot_w

  # Relative desired body poses (aligning anchor/root horizontal position & yaw as in BeyondMimic)
  body_pos_rel_w, body_rot_rel_w = compute_relative_body_poses_rot(
    ref_root_pos, ref_root_rot, ref_body_pos_w, ref_body_rot_w, rob_root_pos, rob_root_rot
  )

  # 1. Body Position (weight 1.0, std 0.3)
  err_b_pos = torch.sum(torch.square(body_pos_rel_w - state.body_pos_w), dim=-1).mean(dim=-1)
  r_b_pos = torch.exp(-err_b_pos / (0.3**2))

  # 2. Body Orientation (weight 1.0, std 0.4)
  err_b_ori = geodesic_rot_error_sq(body_rot_rel_w, state.body_rot_w).mean(dim=-1)
  r_b_ori = torch.exp(-err_b_ori / (0.4**2))

  # 3. Body Linear Velocity (weight 1.0, std 1.0)
  err_b_lvel = torch.sum(torch.square(ref_body_lin_vel_w - state.body_lin_vel_w), dim=-1).mean(dim=-1)
  r_b_lvel = torch.exp(-err_b_lvel / (1.0**2))

  # 4. Body Angular Velocity (weight 1.0, std 3.14)
  err_b_avel = torch.sum(torch.square(ref_body_ang_vel_w - state.body_ang_vel_w), dim=-1).mean(dim=-1)
  r_b_avel = torch.exp(-err_b_avel / (3.14**2))

  # 5. Root Position (weight 0.5, std 0.3)
  err_r_pos = torch.sum(torch.square(ref_root_pos - rob_root_pos), dim=-1)
  r_r_pos = 0.5 * torch.exp(-err_r_pos / (0.3**2))

  # 6. Root Orientation (weight 0.5, std 0.4)
  err_r_ori = geodesic_rot_error_sq(ref_root_rot, rob_root_rot)
  r_r_ori = 0.5 * torch.exp(-err_r_ori / (0.4**2))

  # 7. Action Smoothness (weight -0.1)
  r_smooth = -0.1 * torch.sum(torch.square(action - prev_action), dim=-1)

  # 8. Joint Limit (weight -2.0)
  q_min = soft_joint_pos_limits[:, 0].unsqueeze(0)  # (1, 29)
  q_max = soft_joint_pos_limits[:, 1].unsqueeze(0)  # (1, 29)
  over_min = torch.relu(q_min - state.joint_pos)
  over_max = torch.relu(state.joint_pos - q_max)
  r_limit = -2.0 * torch.sum(over_min + over_max, dim=-1)

  total_reward = (
    r_b_pos + r_b_ori + r_b_lvel + r_b_avel + r_r_pos + r_r_ori + r_smooth + r_limit
  ) * dt
  return total_reward


def compute_termination_errors(
  state: RobotState, ref_root_pos_w: torch.Tensor, ref_root_rot_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute height error |e_{p,z}|, tilt error e_tilt, and hard termination d_t in {0, 1} (Appendix A).
  Returns:
    err_z: (B,) |z_ref - z_robot|
    err_tilt: (B,) |g_z^ref - g_z^robot|
    terminated: (B,) boolean tensor (err_z > 0.25 or err_tilt > 0.8)
  """
  err_z = torch.abs(ref_root_pos_w[:, 2] - state.root_pos_w[:, 2])
  # z-component of unit gravity [0, 0, -1] in root frame is -R_root[:, 2, 2]
  gz_ref = -ref_root_rot_w[:, 2, 2]
  gz_rob = -state.root_rot_w[:, 2, 2]
  err_tilt = torch.abs(gz_ref - gz_rob)
  terminated = (err_z > 0.25) | (err_tilt > 0.8)
  return err_z, err_tilt, terminated

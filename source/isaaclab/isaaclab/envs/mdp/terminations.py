# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

from isaaclab.assets import RigidObjectCollection
"""
MDP terminations.
"""


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


def command_resample(env: ManagerBasedRLEnv, command_name: str, num_resamples: int = 1) -> torch.Tensor:
    """Terminate the episode based on the total number of times commands have been re-sampled.

    This makes the maximum episode length fluid in nature as it depends on how the commands are
    sampled. It is useful in situations where delayed rewards are used :cite:`rudin2022advanced`.
    """
    command: CommandTerm = env.command_manager.get_term(command_name)
    return torch.logical_and((command.time_left <= env.step_dt), (command.command_counter == num_resamples))


"""
Root terminations.
"""


def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle


def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def kuavoV4Pouring_cup_tilted_sideways(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("pouring_cup"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    max_tilt_angle_rad: float = 1.2,
    grace_period_steps: int = 15,
    max_cup_ang_speed_for_failure: float = 0.1,
    lift_height_above_table: float = 1.15,
) -> torch.Tensor:
    cup: RigidObject = env.scene[asset_cfg.name]

    # Cache table Z — XformPrimView get_world_poses() is expensive, only call once
    if not hasattr(env, "_table_z_cache"):
        table_entity = env.scene[table_cfg.name]
        table_positions, _ = table_entity.get_world_poses()  # (num_envs, 3)
        env._table_z_cache = table_positions[:, 2].clone()   # (num_envs,)

    # Cache spawn quaternion — avoids torch.tensor() allocation every step
    if not hasattr(env, "_cup_qref_cache"):
        env._cup_qref_cache = cup.data.default_root_state[:, 3:7].clone()  # (num_envs, 4)

    table_z = env._table_z_cache
    q_ref = env._cup_qref_cache

    # Height gate
    cup_z = cup.data.root_pos_w[:, 2]
    cup_height_above_table = cup_z - table_z
    cup_is_low = cup_height_above_table < lift_height_above_table

    # Grace period
    past_grace = env.episode_length_buf >= grace_period_steps
    if not hasattr(env, "_quat_conj_sign"):
        env._quat_conj_sign = torch.tensor(
            [1, -1, -1, -1], device=cup.data.root_quat_w.device, dtype=cup.data.root_quat_w.dtype
        )
    # Tilt from cached spawn orientation
    q_current = cup.data.root_quat_w
    q_ref_inv = q_ref * env._quat_conj_sign
    w0, x0, y0, z0 = q_ref_inv[:, 0], q_ref_inv[:, 1], q_ref_inv[:, 2], q_ref_inv[:, 3]
    w1, x1, y1, z1 = q_current[:, 0], q_current[:, 1], q_current[:, 2], q_current[:, 3]
    rel_w = torch.clamp(w0*w1 - x0*x1 - y0*y1 - z0*z1, -1.0, 1.0)
    tilt_angle = 2.0 * torch.acos(torch.abs(rel_w))
    cup_is_fallen = tilt_angle > max_tilt_angle_rad

    # Angular velocity gate
    ang_speed = torch.norm(cup.data.root_ang_vel_w, dim=-1)

    # print(f"[{asset_cfg.name}] cup_rel_z: {cup_height_above_table[0]:.4f}, tilt_deg: {torch.rad2deg(tilt_angle[0]):.2f}, "
    #     f"cup_is_low: {cup_is_low[0]}, cup_is_fallen: {cup_is_fallen[0]}, "
    #     f"cup_is_still: {cup_is_still[0]}, past_grace: {past_grace[0]}")

    return past_grace & cup_is_low & cup_is_fallen


def liquid_particle_spilled(
    env: ManagerBasedRLEnv,
    particle_cfg: SceneEntityCfg = SceneEntityCfg("liquid_particles"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("pouring_cup_2"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    min_spilled_count: int = 5,
    cup_xy_radius: float = 0.05,
    bowl_xy_radius: float = 0.08,   # tune to your bowl size
    fell_off_z: float = 0.50,
    grace_period_steps: int = 30,
) -> torch.Tensor:

    particles: RigidObjectCollection = env.scene[particle_cfg.name]
    cup: RigidObject = env.scene[cup_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]

    past_grace = env.episode_length_buf >= grace_period_steps

    p_pos = particles.data.object_pos_w       # (num_envs, 24, 3)

    # XY distance from cup
    cup_xy = cup.data.root_pos_w[:, :2].unsqueeze(1)
    xy_dist_cup = torch.norm(p_pos[..., :2] - cup_xy, dim=-1)
    outside_cup = xy_dist_cup > cup_xy_radius

    # XY distance from bowl
    bowl_xy = bowl.data.root_pos_w[:, :2].unsqueeze(1)
    xy_dist_bowl = torch.norm(p_pos[..., :2] - bowl_xy, dim=-1)
    outside_bowl = xy_dist_bowl > bowl_xy_radius

    # Fell off table entirely
    p_z = p_pos[..., 2]
    fell_off = p_z < fell_off_z

    # Spilled = outside cup AND outside bowl, or fell off table
    spill_count = ((outside_cup & outside_bowl) | fell_off).sum(dim=1)

    # print(f"[spill] count: {spill_count[0]}, "
    #       f"xy_cup min: {xy_dist_cup[0].min():.4f} max: {xy_dist_cup[0].max():.4f}, "
    #       f"xy_bowl min: {xy_dist_bowl[0].min():.4f} max: {xy_dist_bowl[0].max():.4f}, "
    #       f"past_grace: {past_grace[0]}")

    return past_grace & (spill_count >= min_spilled_count)




"""
Joint terminations.
"""


def joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)

    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > limits[..., 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < limits[..., 0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_vel_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    limits = asset.data.soft_joint_vel_limits
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > limits[:, asset_cfg.joint_ids], dim=1)


def joint_vel_out_of_manual_limit(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity, dim=1)


def joint_effort_out_of_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques. If they are not, it means that clipping has occurred.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = ~torch.isclose(
        asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
    )
    return torch.any(out_of_limits, dim=1)


"""
Contact sensor.
"""


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )

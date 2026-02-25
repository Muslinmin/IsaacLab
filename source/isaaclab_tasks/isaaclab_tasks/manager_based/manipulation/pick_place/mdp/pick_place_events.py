# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject, RigidObjectCollection
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# def reset_object_poses_nut_pour(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     pose_range: dict[str, tuple[float, float]],
#     sorting_beaker_cfg: SceneEntityCfg = SceneEntityCfg("sorting_beaker"),
#     factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
#     sorting_bowl_cfg: SceneEntityCfg = SceneEntityCfg("sorting_bowl"),
#     sorting_scale_cfg: SceneEntityCfg = SceneEntityCfg("sorting_scale"),
# ):
#     """Reset the asset root states to a random position and orientation uniformly within the given ranges.

#     Args:
#         env: The RL environment instance.
#         env_ids: The environment IDs to reset the object poses for.
#         sorting_beaker_cfg: The configuration for the sorting beaker asset.
#         factory_nut_cfg: The configuration for the factory nut asset.
#         sorting_bowl_cfg: The configuration for the sorting bowl asset.
#         sorting_scale_cfg: The configuration for the sorting scale asset.
#         pose_range: The dictionary of pose ranges for the objects. Keys are
#                     ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.
#     """
#     # extract the used quantities (to enable type-hinting)
#     sorting_beaker = env.scene[sorting_beaker_cfg.name]
#     factory_nut = env.scene[factory_nut_cfg.name]
#     sorting_bowl = env.scene[sorting_bowl_cfg.name]
#     sorting_scale = env.scene[sorting_scale_cfg.name]

#     # get default root state
#     sorting_beaker_root_states = sorting_beaker.data.default_root_state[env_ids].clone()
#     factory_nut_root_states = factory_nut.data.default_root_state[env_ids].clone()
#     sorting_bowl_root_states = sorting_bowl.data.default_root_state[env_ids].clone()
#     sorting_scale_root_states = sorting_scale.data.default_root_state[env_ids].clone()

#     # get pose ranges
#     range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
#     ranges = torch.tensor(range_list, device=sorting_beaker.device)

#     # randomize sorting beaker and factory nut together
#     rand_samples = math_utils.sample_uniform(
#         ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
#     )
#     orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
#     positions_sorting_beaker = (
#         sorting_beaker_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
#     )
#     positions_factory_nut = factory_nut_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
#     orientations_sorting_beaker = math_utils.quat_mul(sorting_beaker_root_states[:, 3:7], orientations_delta)
#     orientations_factory_nut = math_utils.quat_mul(factory_nut_root_states[:, 3:7], orientations_delta)

#     # randomize sorting bowl
#     rand_samples = math_utils.sample_uniform(
#         ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
#     )
#     orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
#     positions_sorting_bowl = sorting_bowl_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
#     orientations_sorting_bowl = math_utils.quat_mul(sorting_bowl_root_states[:, 3:7], orientations_delta)

#     # randomize scorting scale
#     rand_samples = math_utils.sample_uniform(
#         ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
#     )
#     orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
#     positions_sorting_scale = sorting_scale_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
#     orientations_sorting_scale = math_utils.quat_mul(sorting_scale_root_states[:, 3:7], orientations_delta)

#     # set into the physics simulation
#     sorting_beaker.write_root_pose_to_sim(
#         torch.cat([positions_sorting_beaker, orientations_sorting_beaker], dim=-1), env_ids=env_ids
#     )
#     factory_nut.write_root_pose_to_sim(
#         torch.cat([positions_factory_nut, orientations_factory_nut], dim=-1), env_ids=env_ids
#     )
#     sorting_bowl.write_root_pose_to_sim(
#         torch.cat([positions_sorting_bowl, orientations_sorting_bowl], dim=-1), env_ids=env_ids
#     )
#     sorting_scale.write_root_pose_to_sim(
#         torch.cat([positions_sorting_scale, orientations_sorting_scale], dim=-1), env_ids=env_ids
#     )



# def reset_object_poses_nut_pour(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     pose_range: dict[str, tuple[float, float]],
#     pouring_cup_cfg: SceneEntityCfg = SceneEntityCfg("pouring_cup"),
#     factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
#     bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
# ):
#     """Reset object root poses with XY randomization.
#     - pouring_cup and factory_nut move together (same XY offset, same orientation delta)
#     - bowl randomized independently (XY offset, orientation delta)
#     """
#     # extract assets
#     pouring_cup = env.scene[pouring_cup_cfg.name]
#     factory_nut = env.scene[factory_nut_cfg.name]
#     bowl = env.scene[bowl_cfg.name]

#     # default root states (pos xyz + quat wxyz)
#     cup_root_states = pouring_cup.data.default_root_state[env_ids].clone()
#     nut_root_states = factory_nut.data.default_root_state[env_ids].clone()
#     bowl_root_states = bowl.data.default_root_state[env_ids].clone()

#     # pose ranges: only x,y provided -> z/roll/pitch/yaw become (0,0) => not randomized
#     range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
#     ranges = torch.tensor(range_list, device=pouring_cup.device)

#     # -------------------------
#     # Randomize cup + nut together
#     # -------------------------
#     rand_samples = math_utils.sample_uniform(
#         ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=pouring_cup.device
#     )
#     orientations_delta = math_utils.quat_from_euler_xyz(
#         rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
#     )

#     pos_offset = rand_samples[:, 0:3]
#     env_origins = env.scene.env_origins[env_ids]

#     cup_pos = cup_root_states[:, 0:3] + env_origins + pos_offset
#     nut_pos = nut_root_states[:, 0:3] + env_origins + pos_offset

#     cup_quat = math_utils.quat_mul(cup_root_states[:, 3:7], orientations_delta)
#     nut_quat = math_utils.quat_mul(nut_root_states[:, 3:7], orientations_delta)

#     # -------------------------
#     # Randomize bowl independently
#     # -------------------------
#     rand_samples = math_utils.sample_uniform(
#         ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=pouring_cup.device
#     )
#     orientations_delta = math_utils.quat_from_euler_xyz(
#         rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
#     )

#     bowl_pos = bowl_root_states[:, 0:3] + env_origins + rand_samples[:, 0:3]
#     bowl_quat = math_utils.quat_mul(bowl_root_states[:, 3:7], orientations_delta)

#     # write to sim
#     pouring_cup.write_root_pose_to_sim(torch.cat([cup_pos, cup_quat], dim=-1), env_ids=env_ids)
#     factory_nut.write_root_pose_to_sim(torch.cat([nut_pos, nut_quat], dim=-1), env_ids=env_ids)
#     bowl.write_root_pose_to_sim(torch.cat([bowl_pos, bowl_quat], dim=-1), env_ids=env_ids)


def reset_object_poses_nut_pour(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    pouring_cup_cfg: SceneEntityCfg = SceneEntityCfg("pouring_cup"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
):
    """Reset object root poses with XY randomization.
    - pouring_cup randomized in XY
    - factory_nut placed at cup center (+z offset) every reset
    - bowl randomized independently in XY
    """
    # extract assets
    pouring_cup = env.scene[pouring_cup_cfg.name]
    factory_nut = env.scene[factory_nut_cfg.name]
    bowl = env.scene[bowl_cfg.name]

    # default root states (pos xyz + quat wxyz)
    cup_root_states = pouring_cup.data.default_root_state[env_ids].clone()
    nut_root_states = factory_nut.data.default_root_state[env_ids].clone()
    bowl_root_states = bowl.data.default_root_state[env_ids].clone()

    # pose ranges: only x,y provided -> z/roll/pitch/yaw become (0,0) => not randomized
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=pouring_cup.device)

    # -------------------------
    # Randomize cup + nut together
    # -------------------------
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=pouring_cup.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )

    pos_offset = rand_samples[:, 0:3]
    env_origins = env.scene.env_origins[env_ids]

    cup_pos = cup_root_states[:, 0:3] + env_origins + pos_offset
    nut_pos = nut_root_states[:, 0:3] + env_origins + pos_offset

    cup_quat = math_utils.quat_mul(cup_root_states[:, 3:7], orientations_delta)
    nut_quat = math_utils.quat_mul(nut_root_states[:, 3:7], orientations_delta)

    # -------------------------
    # Randomize bowl independently
    # -------------------------
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=pouring_cup.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )

    bowl_pos = bowl_root_states[:, 0:3] + env_origins + rand_samples[:, 0:3]
    bowl_quat = math_utils.quat_mul(bowl_root_states[:, 3:7], orientations_delta)

    # write to sim
    pouring_cup.write_root_pose_to_sim(torch.cat([cup_pos, cup_quat], dim=-1), env_ids=env_ids)
    factory_nut.write_root_pose_to_sim(torch.cat([nut_pos, nut_quat], dim=-1), env_ids=env_ids)
    bowl.write_root_pose_to_sim(torch.cat([bowl_pos, bowl_quat], dim=-1), env_ids=env_ids)

def reset_liquid_pour_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # --- Ranges ---
    xy_range: dict[str, tuple[float, float]] | None = None,
    table_z_range: tuple[float, float] = (0.0, 0.0),
    yaw_range: tuple[float, float] = (0.0, 0.0),
    # --- Proximity ---
    min_distance: float = 0.12,
    max_rejection_attempts: int = 50,
    # --- Scene entity configs ---
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    cup_1_cfg: SceneEntityCfg = SceneEntityCfg("pouring_cup"),
    cup_2_cfg: SceneEntityCfg = SceneEntityCfg("pouring_cup_2"),
    particles_1_cfg: SceneEntityCfg = SceneEntityCfg("liquid_particles"),
    particles_2_cfg: SceneEntityCfg = SceneEntityCfg("liquid_particles_2"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
):
    """Reset object poses for liquid pouring with domain randomization.

    Algorithm:
        1. Sample a single table Z offset (height variation).
        2. Randomize bowl X/Y on table surface.
        3. Randomize cup 1 X/Y with rejection sampling (> min_distance from bowl).
        4. Randomize cup 2 X/Y with rejection sampling (> min_distance from bowl AND cup 1).
        5. Co-randomize particles with their parent cup (same X/Y/Z offset).
        6. Co-randomize factory nut with cup 1.
        7. Write all poses to sim.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to reset.
        xy_range: Dict with "x" and "y" keys, each a (min, max) tuple.
                  Each object (bowl, cup1, cup2) gets its own independent sample.
        table_z_range: (min, max) tuple for table height Z offset in meters.
        yaw_range: (min, max) tuple for yaw randomization in radians.
        min_distance: Minimum XY Euclidean distance between any pair of
                      (bowl, cup1, cup2) after randomization.
        max_rejection_attempts: Max attempts before accepting last sample.
        table_cfg:      SceneEntityCfg for the table (AssetBase).
        bowl_cfg:       SceneEntityCfg for the bowl.
        cup_1_cfg:      SceneEntityCfg for pouring cup 1.
        cup_2_cfg:      SceneEntityCfg for pouring cup 2.
        particles_1_cfg: SceneEntityCfg for liquid particles in cup 1.
        particles_2_cfg: SceneEntityCfg for liquid particles in cup 2.
        factory_nut_cfg: SceneEntityCfg for the factory nut (co-moves with cup 1).
    """
    if xy_range is None:
        xy_range = {"x": (-0.03, 0.03), "y": (-0.03, 0.03)}

    # ── Resolve scene entities ──────────────────────────────────────────
    bowl: RigidObject = env.scene[bowl_cfg.name]
    cup_1: RigidObject = env.scene[cup_1_cfg.name]
    cup_2: RigidObject = env.scene[cup_2_cfg.name]
    particles_1: RigidObjectCollection = env.scene[particles_1_cfg.name]
    particles_2: RigidObjectCollection = env.scene[particles_2_cfg.name]
    factory_nut: RigidObject = env.scene[factory_nut_cfg.name]

    device = bowl.device
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]  # (n, 3)

    # ── Get default root states ─────────────────────────────────────────
    bowl_default = bowl.data.default_root_state[env_ids].clone()       # (n, 13)
    cup_1_default = cup_1.data.default_root_state[env_ids].clone()
    cup_2_default = cup_2.data.default_root_state[env_ids].clone()
    nut_default = factory_nut.data.default_root_state[env_ids].clone()

    # ── Range values ────────────────────────────────────────────────────
    x_lo, x_hi = xy_range.get("x", (0.0, 0.0))
    y_lo, y_hi = xy_range.get("y", (0.0, 0.0))
    yaw_lo, yaw_hi = yaw_range

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Table Z randomization
    # ══════════════════════════════════════════════════════════════════════
    # Single Z offset shared by table and everything sitting on it.
    table_dz = torch.zeros(n, device=device)
    if table_z_range[0] != 0.0 or table_z_range[1] != 0.0:
        table_dz = torch.empty(n, device=device).uniform_(table_z_range[0], table_z_range[1])

    # Table is an AssetBaseCfg → appears as XformPrimView in scene.extras
    # Move it via set_world_poses(). We track previous offset to undo it on re-reset.
    if table_z_range[0] != 0.0 or table_z_range[1] != 0.0:
        table_entity = env.scene[table_cfg.name]  # XformPrimView
        table_pos, table_quat = table_entity.get_world_poses()  # (total_envs, 3), (total_envs, 4)

        # Initialize offset tracker on first call
        if not hasattr(env, "_prev_table_dz"):
            env._prev_table_dz = torch.zeros(env.scene.num_envs, device=device)

        # Undo previous offset, apply new one (only for envs being reset)
        table_pos[env_ids, 2] += table_dz - env._prev_table_dz[env_ids]
        table_entity.set_world_poses(positions=table_pos, orientations=table_quat)
        env._prev_table_dz[env_ids] = table_dz

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Randomize bowl X/Y
    # ══════════════════════════════════════════════════════════════════════
    bowl_dx = torch.empty(n, device=device).uniform_(x_lo, x_hi)
    bowl_dy = torch.empty(n, device=device).uniform_(y_lo, y_hi)
    bowl_dyaw = torch.empty(n, device=device).uniform_(yaw_lo, yaw_hi)

    # Absolute XY of bowl in local env frame (for proximity checks)
    bowl_abs_xy = torch.stack([
        bowl_default[:, 0] + bowl_dx,
        bowl_default[:, 1] + bowl_dy,
    ], dim=-1)  # (n, 2)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Randomize cup 1 X/Y — proximity check vs bowl
    # ══════════════════════════════════════════════════════════════════════
    cup_1_dx, cup_1_dy = _rejection_sample_xy(
        n, device, x_lo, x_hi, y_lo, y_hi,
        base_xy=cup_1_default[:, 0:2],
        occupied=[bowl_abs_xy],
        min_distance=min_distance,
        max_attempts=max_rejection_attempts,
    )
    cup_1_dyaw = torch.empty(n, device=device).uniform_(yaw_lo, yaw_hi)

    cup_1_abs_xy = torch.stack([
        cup_1_default[:, 0] + cup_1_dx,
        cup_1_default[:, 1] + cup_1_dy,
    ], dim=-1)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Randomize cup 2 X/Y — proximity check vs bowl AND cup 1
    # ══════════════════════════════════════════════════════════════════════
    cup_2_dx, cup_2_dy = _rejection_sample_xy(
        n, device, x_lo, x_hi, y_lo, y_hi,
        base_xy=cup_2_default[:, 0:2],
        occupied=[bowl_abs_xy, cup_1_abs_xy],
        min_distance=min_distance,
        max_attempts=max_rejection_attempts,
    )
    cup_2_dyaw = torch.empty(n, device=device).uniform_(yaw_lo, yaw_hi)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: Write all poses to sim
    # ══════════════════════════════════════════════════════════════════════

    # --- Bowl ---
    _write_rigid_object_pose(
        bowl, bowl_default, origins, env_ids,
        dx=bowl_dx, dy=bowl_dy, dz=table_dz, dyaw=bowl_dyaw,
    )

    # --- Cup 1 ---
    _write_rigid_object_pose(
        cup_1, cup_1_default, origins, env_ids,
        dx=cup_1_dx, dy=cup_1_dy, dz=table_dz, dyaw=cup_1_dyaw,
    )

    # --- Cup 2 ---
    _write_rigid_object_pose(
        cup_2, cup_2_default, origins, env_ids,
        dx=cup_2_dx, dy=cup_2_dy, dz=table_dz, dyaw=cup_2_dyaw,
    )

    # --- Factory nut (co-moves with cup 1) ---
    _write_rigid_object_pose(
        factory_nut, nut_default, origins, env_ids,
        dx=cup_1_dx, dy=cup_1_dy, dz=table_dz, dyaw=cup_1_dyaw,
    )

    # --- Particles 1 (co-move with cup 1) ---
    _write_particle_collection_pose(
        particles_1, env_ids, origins,
        dx=cup_1_dx, dy=cup_1_dy, dz=table_dz,
    )

    # --- Particles 2 (co-move with cup 2) ---
    _write_particle_collection_pose(
        particles_2, env_ids, origins,
        dx=cup_2_dx, dy=cup_2_dy, dz=table_dz,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Rejection sampling for XY with proximity constraint
# ═══════════════════════════════════════════════════════════════════════════

def _rejection_sample_xy(
    n: int,
    device: torch.device,
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    base_xy: torch.Tensor,
    occupied: list[torch.Tensor],
    min_distance: float,
    max_attempts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample XY offsets ensuring minimum distance from all occupied positions.

    For each environment, samples (dx, dy) offsets. The absolute position
    (base_xy + offset) is checked against all occupied positions. Environments
    that violate min_distance are re-sampled up to max_attempts times.

    Args:
        n: Number of environments.
        device: Torch device.
        x_lo, x_hi: X offset range.
        y_lo, y_hi: Y offset range.
        base_xy: Default XY positions, shape (n, 2).
        occupied: List of absolute XY positions to avoid, each shape (n, 2).
        min_distance: Minimum allowed XY Euclidean distance.
        max_attempts: Max rejection iterations.

    Returns:
        (dx, dy) offset tensors, each shape (n,).
    """
    best_dx = torch.empty(n, device=device).uniform_(x_lo, x_hi)
    best_dy = torch.empty(n, device=device).uniform_(y_lo, y_hi)
    needs_resample = torch.ones(n, dtype=torch.bool, device=device)

    for attempt in range(max_attempts):
        if not needs_resample.any():
            break

        # Only resample environments that still violate
        count = needs_resample.sum().item()
        best_dx[needs_resample] = torch.empty(count, device=device).uniform_(x_lo, x_hi)
        best_dy[needs_resample] = torch.empty(count, device=device).uniform_(y_lo, y_hi)

        # Compute candidate absolute XY
        candidate_xy = torch.stack([
            base_xy[:, 0] + best_dx,
            base_xy[:, 1] + best_dy,
        ], dim=-1)  # (n, 2)

        # Check distance against every occupied position
        valid = torch.ones(n, dtype=torch.bool, device=device)
        for occ_xy in occupied:
            dist = torch.norm(candidate_xy - occ_xy, dim=-1)
            valid &= dist > min_distance

        needs_resample = ~valid

    if needs_resample.any():
        num_failed = needs_resample.sum().item()
        print(
            f"[reset_liquid_pour_poses] WARNING: {num_failed}/{n} envs could not satisfy "
            f"min_distance={min_distance:.3f}m after {max_attempts} attempts. "
            f"Consider reducing xy_range or min_distance."
        )

    return best_dx, best_dy


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Write randomized pose for a single RigidObject
# ═══════════════════════════════════════════════════════════════════════════

def _write_rigid_object_pose(
    entity: RigidObject,
    default_state: torch.Tensor,
    origins: torch.Tensor,
    env_ids: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    dz: torch.Tensor,
    dyaw: torch.Tensor,
):
    """Write a randomized pose for a single RigidObject.

    Position = default_pos + env_origin + (dx, dy, dz)
    Orientation = default_quat * delta_yaw_quat

    Args:
        entity: The RigidObject to write to.
        default_state: Default root state, shape (n, 13).
        origins: Environment origins, shape (n, 3).
        env_ids: Environment IDs tensor.
        dx, dy, dz: Position offsets, each shape (n,).
        dyaw: Yaw rotation offset in radians, shape (n,).
    """
    n = len(env_ids)
    device = entity.device

    # Position
    pos = default_state[:, 0:3] + origins
    pos[:, 0] += dx
    pos[:, 1] += dy
    pos[:, 2] += dz

    # Orientation: apply yaw delta to default quaternion
    zeros = torch.zeros(n, device=device)
    quat_delta = math_utils.quat_from_euler_xyz(zeros, zeros, dyaw)
    quat = math_utils.quat_mul(default_state[:, 3:7], quat_delta)

    entity.write_root_pose_to_sim(
        torch.cat([pos, quat], dim=-1), env_ids=env_ids
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Write co-randomized poses for a RigidObjectCollection (particles)
# ═══════════════════════════════════════════════════════════════════════════

def _write_particle_collection_pose(
    particles: RigidObjectCollection,
    env_ids: torch.Tensor,
    origins: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    dz: torch.Tensor,
):
    """Co-randomize a RigidObjectCollection with its parent cup.

    Each particle gets the same (dx, dy, dz) offset as its parent cup,
    preserving the relative arrangement of particles inside the cup.

    Uses the official Isaac Lab RigidObjectCollection API:
        - data.default_object_state  → (num_envs, num_objects, 13)
        - write_object_link_pose_to_sim(poses, env_ids)
        - write_object_com_velocity_to_sim(velocities, env_ids)

    Args:
        particles: The RigidObjectCollection entity.
        env_ids: Environment IDs tensor.
        origins: Environment origins, shape (n, 3).
        dx, dy, dz: Position offsets from the parent cup, each shape (n,).
    """
    # default_object_state: (num_envs, num_objects, 13)
    default_states = particles.data.default_object_state[env_ids].clone()

    # Build offset: (n, 1, 3) → broadcasts over all particles
    offset = torch.stack([dx, dy, dz], dim=-1).unsqueeze(1)  # (n, 1, 3)

    # Apply env origins + randomization offset to positions
    default_states[..., 0:3] += origins.unsqueeze(1) + offset

    # Write poses (pos + quat) and zero velocities
    particles.write_object_link_pose_to_sim(default_states[..., 0:7], env_ids=env_ids)
    particles.write_object_com_velocity_to_sim(default_states[..., 7:13], env_ids=env_ids)



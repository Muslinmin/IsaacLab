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
#     bowl.write_root_pose_to_sim(torch.cat([bowl_pos, bowl_quat], dim=-1), env_ids=env_ids



DEBUG = False


def _dbg(msg: str):
    """Print only when DEBUG is enabled."""
    if DEBUG:
        print(f"[reset_pour_dbg] {msg}")


def reset_liquid_pour_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # --- Ranges ---
    xy_range: dict[str, tuple[float, float]] | None = None,
    table_z_range: tuple[float, float] = (0.0, 0.0),
    yaw_range: tuple[float, float] = (0.0, 0.0),
    # --- Anti-tunneling ---
    spawn_z_buffer: float = 0.005,
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
    if xy_range is None:
        xy_range = {"x": (-0.03, 0.03), "y": (-0.03, 0.03)}

    _dbg("=" * 70)
    _dbg("RESET TRIGGERED")
    _dbg(f"  env_ids: {env_ids.tolist()}")

    # ── Skip randomization during replay/annotation ─────────────────────
    # When annotate_demos.py (or generate_dataset.py) calls env.reset_to(),
    # the scene is restored from the HDF5 recording. We must NOT randomize
    # on top of that, or objects won't match the recorded trajectory.
    #
    # The calling script sets env._skip_pose_randomization = True before
    # calling reset_to(), and clears it after.
    if getattr(env, "_skip_pose_randomization", False):
        _dbg("  ⏭️  SKIPPING randomization (_skip_pose_randomization=True)")
        _dbg("=" * 70)
        return

    _dbg(f"  xy_range: {xy_range}")
    _dbg(f"  table_z_range: {table_z_range}")
    _dbg(f"  yaw_range: {yaw_range}")
    _dbg(f"  spawn_z_buffer: {spawn_z_buffer}")
    _dbg(f"  min_distance: {min_distance}")

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
    bowl_default = bowl.data.default_root_state[env_ids].clone()
    cup_1_default = cup_1.data.default_root_state[env_ids].clone()
    cup_2_default = cup_2.data.default_root_state[env_ids].clone()
    nut_default = factory_nut.data.default_root_state[env_ids].clone()

    # ── Debug: print default positions (local frame, before env_origin) ─
    _dbg("-" * 50)
    _dbg("DEFAULT POSITIONS (local frame, from init_state in cfg):")
    for i in range(min(n, 3)):  # print first 3 envs max
        _dbg(f"  env[{env_ids[i].item()}]:")
        _dbg(f"    env_origin:    ({origins[i, 0]:.5f}, {origins[i, 1]:.5f}, {origins[i, 2]:.5f})")
        _dbg(f"    bowl default:  ({bowl_default[i, 0]:.5f}, {bowl_default[i, 1]:.5f}, {bowl_default[i, 2]:.5f})")
        _dbg(f"    cup_1 default: ({cup_1_default[i, 0]:.5f}, {cup_1_default[i, 1]:.5f}, {cup_1_default[i, 2]:.5f})")
        _dbg(f"    cup_2 default: ({cup_2_default[i, 0]:.5f}, {cup_2_default[i, 1]:.5f}, {cup_2_default[i, 2]:.5f})")
        _dbg(f"    nut default:   ({nut_default[i, 0]:.5f}, {nut_default[i, 1]:.5f}, {nut_default[i, 2]:.5f})")

    # ── Debug: print current LIVE positions (world frame) ───────────────
    _dbg("-" * 50)
    _dbg("CURRENT LIVE POSITIONS (world frame, before this reset):")
    for i in range(min(n, 3)):
        eid = env_ids[i].item()
        _dbg(f"  env[{eid}]:")
        _dbg(f"    bowl LIVE:  ({bowl.data.root_pos_w[eid, 0]:.5f}, {bowl.data.root_pos_w[eid, 1]:.5f}, {bowl.data.root_pos_w[eid, 2]:.5f})")
        _dbg(f"    cup_1 LIVE: ({cup_1.data.root_pos_w[eid, 0]:.5f}, {cup_1.data.root_pos_w[eid, 1]:.5f}, {cup_1.data.root_pos_w[eid, 2]:.5f})")
        _dbg(f"    cup_2 LIVE: ({cup_2.data.root_pos_w[eid, 0]:.5f}, {cup_2.data.root_pos_w[eid, 1]:.5f}, {cup_2.data.root_pos_w[eid, 2]:.5f})")

    # ── Range values ────────────────────────────────────────────────────
    x_lo, x_hi = xy_range.get("x", (0.0, 0.0))
    y_lo, y_hi = xy_range.get("y", (0.0, 0.0))
    yaw_lo, yaw_hi = yaw_range

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Table Z randomization
    # ══════════════════════════════════════════════════════════════════════
    table_dz = torch.zeros(n, device=device)
    if table_z_range[0] != 0.0 or table_z_range[1] != 0.0:
        table_dz = torch.empty(n, device=device).uniform_(table_z_range[0], table_z_range[1])

    # Read table position BEFORE moving it
    table_entity = env.scene[table_cfg.name]
    table_pos_before, table_quat = table_entity.get_world_poses()

    _dbg("-" * 50)
    _dbg("TABLE Z RANDOMIZATION:")
    for i in range(min(n, 3)):
        eid = env_ids[i].item()
        _dbg(f"  env[{eid}]: table_z_before={table_pos_before[eid, 2]:.5f}, table_dz={table_dz[i]:.5f}")

    if table_z_range[0] != 0.0 or table_z_range[1] != 0.0:
        if not hasattr(env, "_prev_table_dz"):
            env._prev_table_dz = torch.zeros(env.scene.num_envs, device=device)

        table_pos_before[env_ids, 2] += table_dz - env._prev_table_dz[env_ids]
        table_entity.set_world_poses(positions=table_pos_before, orientations=table_quat)
        env._prev_table_dz[env_ids] = table_dz

        # Read back to confirm
        table_pos_after, _ = table_entity.get_world_poses()
        for i in range(min(n, 3)):
            eid = env_ids[i].item()
            _dbg(f"  env[{eid}]: table_z_after={table_pos_after[eid, 2]:.5f}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Compute effective object dz
    # ══════════════════════════════════════════════════════════════════════
    object_dz = table_dz + spawn_z_buffer

    _dbg("-" * 50)
    _dbg("EFFECTIVE OBJECT DZ (table_dz + spawn_z_buffer):")
    for i in range(min(n, 3)):
        _dbg(f"  env[{env_ids[i].item()}]: object_dz={object_dz[i]:.5f} "
             f"(table_dz={table_dz[i]:.5f} + buffer={spawn_z_buffer:.4f})")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Randomize bowl X/Y
    # ══════════════════════════════════════════════════════════════════════
    bowl_dx = torch.empty(n, device=device).uniform_(x_lo, x_hi)
    bowl_dy = torch.empty(n, device=device).uniform_(y_lo, y_hi)
    bowl_dyaw = torch.empty(n, device=device).uniform_(yaw_lo, yaw_hi)

    bowl_abs_xy = torch.stack([
        bowl_default[:, 0] + bowl_dx,
        bowl_default[:, 1] + bowl_dy,
    ], dim=-1)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Randomize cup 1 X/Y — proximity check vs bowl
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
    # STEP 5: Randomize cup 2 X/Y — proximity check vs bowl AND cup 1
    # ══════════════════════════════════════════════════════════════════════
    cup_2_dx, cup_2_dy = _rejection_sample_xy(
        n, device, x_lo, x_hi, y_lo, y_hi,
        base_xy=cup_2_default[:, 0:2],
        occupied=[bowl_abs_xy, cup_1_abs_xy],
        min_distance=min_distance,
        max_attempts=max_rejection_attempts,
    )
    cup_2_dyaw = torch.empty(n, device=device).uniform_(yaw_lo, yaw_hi)

    # ── Debug: XY offsets and proximity ──────────────────────────────────
    _dbg("-" * 50)
    _dbg("XY OFFSETS & PROXIMITY CHECK:")
    for i in range(min(n, 3)):
        eid = env_ids[i].item()
        b_xy = bowl_abs_xy[i]
        c1_xy = cup_1_abs_xy[i]
        c2_xy = torch.stack([cup_2_default[i, 0] + cup_2_dx[i], cup_2_default[i, 1] + cup_2_dy[i]])
        d_bc1 = torch.norm(b_xy - c1_xy).item()
        d_bc2 = torch.norm(b_xy - c2_xy).item()
        d_c1c2 = torch.norm(c1_xy - c2_xy).item()
        _dbg(f"  env[{eid}]:")
        _dbg(f"    bowl  offset: dx={bowl_dx[i]:.4f}, dy={bowl_dy[i]:.4f}, abs_xy=({b_xy[0]:.4f}, {b_xy[1]:.4f})")
        _dbg(f"    cup_1 offset: dx={cup_1_dx[i]:.4f}, dy={cup_1_dy[i]:.4f}, abs_xy=({c1_xy[0]:.4f}, {c1_xy[1]:.4f})")
        _dbg(f"    cup_2 offset: dx={cup_2_dx[i]:.4f}, dy={cup_2_dy[i]:.4f}, abs_xy=({c2_xy[0]:.4f}, {c2_xy[1]:.4f})")
        _dbg(f"    distances: bowl↔cup1={d_bc1:.4f}, bowl↔cup2={d_bc2:.4f}, cup1↔cup2={d_c1c2:.4f}  "
             f"(min_distance={min_distance:.3f}) "
             f"{'✅ OK' if min(d_bc1, d_bc2, d_c1c2) > min_distance else '⚠️ VIOLATION'}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: Write all poses to sim
    # ══════════════════════════════════════════════════════════════════════
    _dbg("-" * 50)
    _dbg("WRITING FINAL WORLD POSITIONS:")

    # --- Bowl ---
    bowl_final_pos = _write_rigid_object_pose(
        bowl, bowl_default, origins, env_ids,
        dx=bowl_dx, dy=bowl_dy, dz=object_dz, dyaw=bowl_dyaw,
    )

    # --- Cup 1 ---
    cup1_final_pos = _write_rigid_object_pose(
        cup_1, cup_1_default, origins, env_ids,
        dx=cup_1_dx, dy=cup_1_dy, dz=object_dz, dyaw=cup_1_dyaw,
    )

    # --- Cup 2 ---
    cup2_final_pos = _write_rigid_object_pose(
        cup_2, cup_2_default, origins, env_ids,
        dx=cup_2_dx, dy=cup_2_dy, dz=object_dz, dyaw=cup_2_dyaw,
    )

    # --- Factory nut (co-moves with cup 1) ---
    nut_final_pos = _write_rigid_object_pose(
        factory_nut, nut_default, origins, env_ids,
        dx=cup_1_dx, dy=cup_1_dy, dz=object_dz, dyaw=cup_1_dyaw,
    )

    # --- Particles 1 (co-move with cup 1) ---
    _write_particle_collection_pose(
        particles_1, env_ids, origins,
        dx=cup_1_dx, dy=cup_1_dy, dz=object_dz,
    )

    # --- Particles 2 (co-move with cup 2) ---
    _write_particle_collection_pose(
        particles_2, env_ids, origins,
        dx=cup_2_dx, dy=cup_2_dy, dz=object_dz,
    )

    # ── Debug: FINAL SUMMARY — the numbers that matter ──────────────────
    # Re-read table Z after all writes
    table_pos_final, _ = table_entity.get_world_poses()

    _dbg("-" * 50)
    _dbg("🔍 SPAWN HEIGHT DIAGNOSTIC (object_z vs table_surface_z):")
    _dbg("   If gap is NEGATIVE → object is INSIDE the table!")
    _dbg("   You need to raise the default Z in your env_cfg.py init_state")
    for i in range(min(n, 3)):
        eid = env_ids[i].item()
        table_surface_z = table_pos_final[eid, 2].item()
        bowl_z = bowl_final_pos[i, 2].item()
        cup1_z = cup1_final_pos[i, 2].item()
        cup2_z = cup2_final_pos[i, 2].item()
        nut_z = nut_final_pos[i, 2].item()

        _dbg(f"  env[{eid}]:")
        _dbg(f"    table_surface_z: {table_surface_z:.5f}")
        _dbg(f"    bowl_z:  {bowl_z:.5f}  gap={bowl_z - table_surface_z:+.5f}m  "
             f"{'✅' if bowl_z > table_surface_z else '❌ INSIDE TABLE'}")
        _dbg(f"    cup_1_z: {cup1_z:.5f}  gap={cup1_z - table_surface_z:+.5f}m  "
             f"{'✅' if cup1_z > table_surface_z else '❌ INSIDE TABLE'}")
        _dbg(f"    cup_2_z: {cup2_z:.5f}  gap={cup2_z - table_surface_z:+.5f}m  "
             f"{'✅' if cup2_z > table_surface_z else '❌ INSIDE TABLE'}")
        _dbg(f"    nut_z:   {nut_z:.5f}  gap={nut_z - table_surface_z:+.5f}m  "
             f"{'✅' if nut_z > table_surface_z else '❌ INSIDE TABLE'}")

    _dbg("=" * 70)


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
    """Sample XY offsets ensuring minimum distance from all occupied positions."""
    best_dx = torch.empty(n, device=device).uniform_(x_lo, x_hi)
    best_dy = torch.empty(n, device=device).uniform_(y_lo, y_hi)
    needs_resample = torch.ones(n, dtype=torch.bool, device=device)

    for attempt in range(max_attempts):
        if not needs_resample.any():
            break

        count = needs_resample.sum().item()
        best_dx[needs_resample] = torch.empty(count, device=device).uniform_(x_lo, x_hi)
        best_dy[needs_resample] = torch.empty(count, device=device).uniform_(y_lo, y_hi)

        candidate_xy = torch.stack([
            base_xy[:, 0] + best_dx,
            base_xy[:, 1] + best_dy,
        ], dim=-1)

        valid = torch.ones(n, dtype=torch.bool, device=device)
        for occ_xy in occupied:
            dist = torch.norm(candidate_xy - occ_xy, dim=-1)
            valid &= dist > min_distance

        needs_resample = ~valid

    if needs_resample.any():
        num_failed = needs_resample.sum().item()
        print(
            f"[reset_liquid_pour_poses] ⚠️  WARNING: {num_failed}/{n} envs could not satisfy "
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
) -> torch.Tensor:
    """Write a randomized pose for a single RigidObject.

    Returns:
        The final world-frame positions (n, 3) for debug logging.
    """
    n = len(env_ids)
    device = entity.device

    pos = default_state[:, 0:3] + origins
    pos[:, 0] += dx
    pos[:, 1] += dy
    pos[:, 2] += dz

    zeros = torch.zeros(n, device=device)
    quat_delta = math_utils.quat_from_euler_xyz(zeros, zeros, dyaw)
    quat = math_utils.quat_mul(default_state[:, 3:7], quat_delta)

    pose = torch.cat([pos, quat], dim=-1)
    entity.write_root_pose_to_sim(pose, env_ids=env_ids)

    return pos.clone()


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
    """Co-randomize a RigidObjectCollection with its parent cup."""
    default_states = particles.data.default_object_state[env_ids].clone()
    num_particles = default_states.shape[1]

    offset = torch.stack([dx, dy, dz], dim=-1).unsqueeze(1)
    default_states[..., 0:3] += origins.unsqueeze(1) + offset

    particles.write_object_link_pose_to_sim(default_states[..., 0:7], env_ids=env_ids)
    particles.write_object_com_velocity_to_sim(default_states[..., 7:13], env_ids=env_ids)

    # Debug: print particle Z range
    if DEBUG:
        for i in range(min(len(env_ids), 3)):
            p_z = default_states[i, :, 2]
            _dbg(f"    particles env[{env_ids[i].item()}]: "
                 f"{num_particles} spheres, z_min={p_z.min():.5f}, z_max={p_z.max():.5f}")


# ═══════════════════════════════════════════════════════════════════════════
# Example EventCfg wiring
# ═══════════════════════════════════════════════════════════════════════════
#
# @configclass
# class EventCfg:
#     reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
#
#     set_factory_nut_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("factory_nut"),
#             "mass_distribution_params": (0.2, 0.2),
#             "operation": "abs",
#         },
#     )
#
#     reset_object = EventTerm(
#         func=mdp.reset_liquid_pour_poses,
#         mode="reset",
#         params={
#             "xy_range": {
#                 "x": [-0.04, 0.04],
#                 "y": [-0.04, 0.04],
#             },
#             "table_z_range": (-0.02, 0.03),
#             "yaw_range": (-0.15, 0.15),
#             "min_distance": 0.12,
#             "spawn_z_buffer": 0.005,
#         },
#     )
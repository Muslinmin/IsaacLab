# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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


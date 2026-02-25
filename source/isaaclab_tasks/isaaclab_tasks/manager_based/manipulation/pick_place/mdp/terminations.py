# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject, AssetBase
from isaaclab.managers import SceneEntityCfg

from isaaclab.assets import RigidObjectCollection


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done_pick_place(
    env: ManagerBasedRLEnv,
    task_link_name: str = "",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_wrist_max_x: float = 0.26,
    min_x: float = 0.40,
    max_x: float = 0.85,
    min_y: float = 0.35,
    max_y: float = 0.60,
    max_height: float = 1.10,
    min_vel: float = 0.20,
) -> torch.Tensor:
    """Determine if the object placement task is complete.

    This function checks whether all success conditions for the task have been met:
    1. object is within the target x/y range
    2. object is below a minimum height
    3. object velocity is below threshold
    4. Right robot wrist is retracted back towards body (past a given x pos threshold)

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity.
        right_wrist_max_x: Maximum x position of the right wrist for task completion.
        min_x: Minimum x position of the object for task completion.
        max_x: Maximum x position of the object for task completion.
        min_y: Minimum y position of the object for task completion.
        max_y: Maximum y position of the object for task completion.
        max_height: Maximum height (z position) of the object for task completion.
        min_vel: Minimum velocity magnitude of the object for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    if task_link_name == "":
        raise ValueError("task_link_name must be provided to task_done_pick_place")

    # Get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]

    # Extract wheel position relative to environment origin
    object_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    object_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    object_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    object_vel = torch.abs(object.data.root_vel_w)

    # Get right wrist position relative to environment origin
    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index(task_link_name)
    right_wrist_x = robot_body_pos_w[:, right_eef_idx, 0] - env.scene.env_origins[:, 0]

    # Check all success conditions and combine with logical AND
    done = object_x < max_x
    done = torch.logical_and(done, object_x > min_x)
    done = torch.logical_and(done, object_y < max_y)
    done = torch.logical_and(done, object_y > min_y)
    done = torch.logical_and(done, object_height < max_height)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, object_vel[:, 0] < min_vel)
    done = torch.logical_and(done, object_vel[:, 1] < min_vel)
    done = torch.logical_and(done, object_vel[:, 2] < min_vel)

    return done


def task_done_nut_pour(
    env: ManagerBasedRLEnv,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    pouring_cup_cfg: SceneEntityCfg = SceneEntityCfg("pouring_cup"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
    # nut in bowl
    max_nut_to_bowl_xy: float = 0.05,
    min_nut_above_bowl_bottom_z: float = 0.00,
    max_nut_below_bowl_rim_z: float = 0.12,
    # cup on table
    z_threshold: float = 1.02,
    vz_threshold: float = 0.02,
) -> torch.Tensor:

    bowl: RigidObject = env.scene[bowl_cfg.name]
    cup: RigidObject = env.scene[pouring_cup_cfg.name]
    nut: RigidObject = env.scene[factory_nut_cfg.name]

    bowl_pos = bowl.data.root_pos_w - env.scene.env_origins
    cup_pos = cup.data.root_pos_w - env.scene.env_origins
    nut_pos = nut.data.root_pos_w - env.scene.env_origins

    # nut inside bowl
    dx = nut_pos[:, 0] - bowl_pos[:, 0]
    dy = nut_pos[:, 1] - bowl_pos[:, 1]
    nut_to_bowl_xy = torch.sqrt(dx * dx + dy * dy)
    nut_rel_z = nut_pos[:, 2] - bowl_pos[:, 2]

    nut_inside_bowl = nut_to_bowl_xy < max_nut_to_bowl_xy
    nut_inside_bowl = torch.logical_and(nut_inside_bowl, nut_rel_z > min_nut_above_bowl_bottom_z)
    nut_inside_bowl = torch.logical_and(nut_inside_bowl, nut_rel_z < max_nut_below_bowl_rim_z)


    # cup on table (vectorized, scalable)
    cup_z = cup_pos[:, 2]
    cup_vz = cup.data.root_lin_vel_w[:, 2]
    cup_on_table = (cup_z < z_threshold) & (torch.abs(cup_vz) < vz_threshold)

    # if env.num_envs == 1:
    #     print("---- DEBUG SUCCESS CHECK ----")
    #     print("nut_to_bowl_xy:", nut_to_bowl_xy.item())
    #     print("nut_rel_z:", nut_rel_z.item())
    #     print("cup_z:", cup_pos[:, 2].item())
    #     print("nut_inside_bowl:", nut_inside_bowl.item())
    #     print("cup_on_table:", cup_on_table.item())


    return torch.logical_and(nut_inside_bowl, cup_on_table)



def liquid_particle_pour_success(
    env: ManagerBasedRLEnv,
    particle_cfg: SceneEntityCfg = SceneEntityCfg("liquid_particles"),
    particle_cfg_2: SceneEntityCfg = SceneEntityCfg("liquid_particles_2"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    pouring_cup_cfg: SceneEntityCfg = SceneEntityCfg("pouring_cup"),
    pouring_cup_cfg_2: SceneEntityCfg = SceneEntityCfg("pouring_cup_2"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    success_ratio: float = 0.6,
    bowl_xy_radius: float = 0.06,
    bowl_rim_offset: float = 0.10,
    bowl_bottom_offset: float = -0.02,
    cup_lift_threshold: float = 1.05,
    cup_vz_threshold: float = 0.05,
    particle_vel_threshold: float = 0.05,
    required_both: bool = False
) -> torch.Tensor:

    bowl: RigidObject = env.scene[bowl_cfg.name]

    # Recompute each episode
    cache_key = "_bowl_z_success_cache"
    if env.episode_length_buf[0] == 0 or not hasattr(env, cache_key):
        bowl_z = bowl.data.root_pos_w[:, 2]
        setattr(env, cache_key, {
            "bottom": (bowl_z + bowl_bottom_offset).clone(),
            "rim":    (bowl_z + bowl_rim_offset).clone(),
        })
    if env.episode_length_buf[0] == 0 or not hasattr(env, "_table_z_success_cache"):
        table_entity = env.scene[table_cfg.name]
        table_positions, _ = table_entity.get_world_poses()
        env._table_z_success_cache = table_positions[:, 2].clone()

    bowl_cache = getattr(env, cache_key)
    bowl_z_bottom = bowl_cache["bottom"].unsqueeze(1)
    bowl_z_rim    = bowl_cache["rim"].unsqueeze(1)
    table_z       = env._table_z_success_cache
    bowl_xy       = bowl.data.root_pos_w[:, :2].unsqueeze(1)

    def count_in_bowl(particle_cfg_local, cup_cfg_local):
        particles: RigidObjectCollection = env.scene[particle_cfg_local.name]
        cup: RigidObject = env.scene[cup_cfg_local.name]

        p_pos = particles.data.object_pos_w
        p_vel = particles.data.object_lin_vel_w
        num_particles = p_pos.shape[1]

        xy_dist = torch.norm(p_pos[..., :2] - bowl_xy, dim=-1)
        p_z = p_pos[..., 2]

        inside_bowl = (
            (xy_dist < bowl_xy_radius)
            & (p_z > bowl_z_bottom)
            & (p_z < bowl_z_rim)
        )
        settled = torch.norm(p_vel, dim=-1) < particle_vel_threshold
        in_bowl_count = (inside_bowl & settled).sum(dim=1)

        cup_height = cup.data.root_pos_w[:, 2] - table_z
        cup_vz = cup.data.root_lin_vel_w[:, 2]
        cup_on_table = (cup_height < cup_lift_threshold) & (torch.abs(cup_vz) < cup_vz_threshold)

        min_count = int(num_particles * success_ratio)
        success = (in_bowl_count >= min_count) & cup_on_table

        print(f"[success:{particle_cfg_local.name}] in_bowl: {in_bowl_count[0]}/{num_particles} "
              f"(need {min_count}), cup_on_table: {cup_on_table[0]}")

        return success

    success_cup1 = count_in_bowl(particle_cfg, pouring_cup_cfg)
    success_cup2 = count_in_bowl(particle_cfg_2, pouring_cup_cfg_2)
    if required_both:
        return success_cup1 & success_cup2
    else:
        # Either cup successfully poured = success
        return success_cup1 | success_cup2





def task_done_exhaust_pipe(
    env: ManagerBasedRLEnv,
    blue_exhaust_pipe_cfg: SceneEntityCfg = SceneEntityCfg("blue_exhaust_pipe"),
    blue_sorting_bin_cfg: SceneEntityCfg = SceneEntityCfg("blue_sorting_bin"),
    max_blue_exhaust_to_bin_x: float = 0.085,
    max_blue_exhaust_to_bin_y: float = 0.200,
    min_blue_exhaust_to_bin_y: float = -0.090,
    max_blue_exhaust_to_bin_z: float = 0.070,
) -> torch.Tensor:
    """Determine if the exhaust pipe task is complete.

    This function checks whether all success conditions for the task have been met:
    1. The blue exhaust pipe is placed in the correct position

    Args:
        env: The RL environment instance.
        blue_exhaust_pipe_cfg: Configuration for the blue exhaust pipe entity.
        blue_sorting_bin_cfg: Configuration for the blue sorting bin entity.
        max_blue_exhaust_to_bin_x: Maximum x position of the blue exhaust pipe
            relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_y: Maximum y position of the blue exhaust pipe
            relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_z: Maximum z position of the blue exhaust pipe
            relative to the blue sorting bin for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entities from the scene
    blue_exhaust_pipe: RigidObject = env.scene[blue_exhaust_pipe_cfg.name]
    blue_sorting_bin: RigidObject = env.scene[blue_sorting_bin_cfg.name]

    # Get positions relative to environment origin
    blue_exhaust_pipe_pos = blue_exhaust_pipe.data.root_pos_w - env.scene.env_origins
    blue_sorting_bin_pos = blue_sorting_bin.data.root_pos_w - env.scene.env_origins

    # blue exhaust to bin
    blue_exhaust_to_bin_x = torch.abs(blue_exhaust_pipe_pos[:, 0] - blue_sorting_bin_pos[:, 0])
    blue_exhaust_to_bin_y = blue_exhaust_pipe_pos[:, 1] - blue_sorting_bin_pos[:, 1]
    blue_exhaust_to_bin_z = blue_exhaust_pipe_pos[:, 2] - blue_sorting_bin_pos[:, 2]

    done = blue_exhaust_to_bin_x < max_blue_exhaust_to_bin_x
    done = torch.logical_and(done, blue_exhaust_to_bin_y < max_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_y > min_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_z < max_blue_exhaust_to_bin_z)

    return done

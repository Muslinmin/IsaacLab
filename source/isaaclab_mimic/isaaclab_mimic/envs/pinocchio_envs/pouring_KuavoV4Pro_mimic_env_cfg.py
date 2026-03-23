# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.kuavoV4ProPouring_pink_ik_env_cfg import KuavoV4ProPouringPinkIKEnvCfg, KuavoV4ProPouringPinkIKCosmosEnvCfg
from isaaclab_tasks.manager_based.manipulation.pick_place.kuavoV4ProPouring_env_cfg import PourKuavoV4ProTeleopEnvCfg




@configclass
class PouringKuavoV4ProMimic_GenCfg(KuavoV4ProPouringPinkIKEnvCfg, MimicEnvCfg):

    def __post_init__(self):
        # Calling post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "KuavoV4Pro_pouring_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = False
        self.datagen_config.max_num_failures = 25
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 25
        self.datagen_config.seed = 10

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
            object_ref="pouring_cup_2",          
            subtask_term_signal=None,          # no boundary to annotate
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.0,
            num_interpolation_steps=0,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            )
        )
        
        self.subtask_configs["right"] = subtask_configs

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="pouring_cup",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grab_lift_cup",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generatio
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )

        # dummy final subtask so "place_done" isn't the last one
        subtask_configs.append(SubTaskConfig(
            object_ref="pouring_cup",
            subtask_term_signal="episode_done_idle",
            first_subtask_start_offset_range=(0, 0),
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.0,
            num_interpolation_steps=0,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ))

        self.subtask_configs["left"] = subtask_configs



@configclass
class PouringKuavoV4ProMimic_CosmosGenCfg(KuavoV4ProPouringPinkIKCosmosEnvCfg, MimicEnvCfg):

    def __post_init__(self):
        # Calling post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "KuavoV4Pro_pouring_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = False
        self.datagen_config.max_num_failures = 25
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 25
        self.datagen_config.seed = 10

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
            object_ref="pouring_cup",          
            subtask_term_signal=None,          # no boundary to annotate
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.0,
            num_interpolation_steps=0,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            )
        )
        
        self.subtask_configs["left"] = subtask_configs

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="pouring_cup_2",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grab_lift_cup",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generatio
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )

        # dummy final subtask so "place_done" isn't the last one
        subtask_configs.append(SubTaskConfig(
            object_ref="pouring_cup_2",
            subtask_term_signal="episode_done_idle",
            first_subtask_start_offset_range=(0, 0),
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.0,
            num_interpolation_steps=0,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ))

        self.subtask_configs["right"] = subtask_configs



@configclass
class PouringKuavoV4ProMimic_AnnotateEnvCfg(PourKuavoV4ProTeleopEnvCfg, MimicEnvCfg):

    def __post_init__(self):
        # Calling post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "KuavoV4Pro_pouring_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = False
        self.datagen_config.max_num_failures = 25
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 25
        self.datagen_config.seed = 10

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
            object_ref="pouring_cup_2",          # idle_right
            subtask_term_signal=None,          # no boundary to annotate
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.0,
            num_interpolation_steps=0,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            )
        )
        
        self.subtask_configs["right"] = subtask_configs

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="pouring_cup",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grab_lift_cup",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generatio
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )

        # dummy final subtask so "place_done" isn't the last one
        subtask_configs.append(SubTaskConfig(
            object_ref="pouring_cup",
            subtask_term_signal="episode_done_idle",
            first_subtask_start_offset_range=(0, 0),
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.0,
            num_interpolation_steps=0,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ))

        self.subtask_configs["left"] = subtask_configs

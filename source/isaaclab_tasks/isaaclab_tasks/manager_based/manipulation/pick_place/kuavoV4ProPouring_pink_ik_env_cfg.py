# kuavoV4ProPouring_pink_ik_env_cfg.py

from pink.tasks import DampingTask, FrameTask

import isaaclab.controllers.utils as ControllerUtils
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.kuavoV4ProPouring_env_cfg import (
    PourKuavoV4ProBaseEnvCfg, PourKuavoV4ProGenerateBaseEnvCfg
)


@configclass
class KuavoPouringPinkActionsCfg:
    """Pink IK action term (34D: 14 EEF pose + 20 fingers)."""
    
    pink: PinkInverseKinematicsActionCfg = PinkInverseKinematicsActionCfg(
        # Joints controlled by Pink IK (arms only)
        pink_controlled_joint_names=[
            *[f"zarm_l{i}_joint" for i in range(1, 8)],
            *[f"zarm_r{i}_joint" for i in range(1, 8)],
        ],
        # Finger joints (passed through directly)
        hand_joint_names=[
            # Left hand (10)
            "l_thumbCMC", "l_thumbMCP",
            "l_indexMCP", "l_indexPIP",
            "l_middleMCP", "l_middlePIP",
            "l_ringMCP", "l_ringPIP",
            "l_littleMCP", "l_littlePIP",
            # Right hand (10)
            "r_thumbCMC", "r_thumbMCP",
            "r_indexMCP", "r_indexPIP",
            "r_middleMCP", "r_middlePIP",
            "r_ringMCP", "r_ringPIP",
            "r_littleMCP", "r_littlePIP",
        ],
        target_eef_link_names={
            "left": "l_palm",
            "right": "r_palm",
        },
        # NOTE: target_eef_link_names is INSIDE controller config, NOT here!
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=20,
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "l_palm",  # Pinocchio frame name
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=8,
                    gain=0.9,
                ),
                FrameTask(
                    "r_palm",  # Pinocchio frame name
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=8,
                    gain=0.9,
                ),
                DampingTask(cost=0.1)
            ],
            fixed_input_tasks=[
                NullSpacePostureTask(
                    cost=5.0,
                    lm_damping=0.5,
                    controlled_frames=["l_palm", "r_palm"],
                    controlled_joints=[
                        *[f"zarm_l{i}_joint" for i in range(1, 8)],
                        *[f"zarm_r{i}_joint" for i in range(1, 8)],
                    ],
                ),],
            xr_enabled=False,
        ),
    )


@configclass
class KuavoV4ProPouringPinkIKEnvCfg(PourKuavoV4ProBaseEnvCfg):
    """Environment config for Mimic data generation with Pink IK."""
    
    # CRITICAL: Start with no actions (base has actions=None)
    actions = None
    
    def __post_init__(self):
        # MUST call super().__post_init__() FIRST to set decimation, episode_length_s, etc.
        super().__post_init__()
        
        # THEN replace actions with Pink IK
        self.actions = KuavoPouringPinkActionsCfg()
        temp_urdf_output_path = "/workspace/biped_s48/biped_s48.urdf"
        temp_urdf_meshes_output_path = "/workspace/biped_s48/meshes"
        # Convert USD to URDF for Pink/Pinocchio
        # temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
        #     self.scene.robot.spawn.usd_path,
        #     "/tmp",
        #     exclude_joints=["joints_camera"],
        #     force_conversion=True,
        # )
        
        # Set URDF paths for the IK controller
        self.actions.pink.controller.urdf_path = temp_urdf_output_path
        self.actions.pink.controller.mesh_path = temp_urdf_meshes_output_path


@configclass
class KuavoV4ProPouringPinkIKCosmosEnvCfg(PourKuavoV4ProGenerateBaseEnvCfg):
    """Different Parent for COSMOS generation, but still uses the same Pink IK action space."""
    
    # CRITICAL: Start with no actions (base has actions=None)
    actions = None
    
    def __post_init__(self):
        # MUST call super().__post_init__() FIRST to set decimation, episode_length_s, etc.
        super().__post_init__()
        
        # THEN replace actions with Pink IK
        self.actions = KuavoPouringPinkActionsCfg()
        temp_urdf_output_path = "/workspace/biped_s48/biped_s48.urdf"
        temp_urdf_meshes_output_path = "/workspace/biped_s48/meshes"
        # Convert USD to URDF for Pink/Pinocchio
        # temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
        #     self.scene.robot.spawn.usd_path,
        #     "/tmp",
        #     force_conversion=True,
        # )
        
        # Set URDF paths for the IK controller
        self.actions.pink.controller.urdf_path = temp_urdf_output_path
        self.actions.pink.controller.mesh_path = temp_urdf_meshes_output_path
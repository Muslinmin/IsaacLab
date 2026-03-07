# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv



def set_fourth_joints_to_90(env, angle_rad: float = -2.1):
    """Force zarm_l4_joint and zarm_r4_joint to a given angle (in radians)."""
    robot = env.scene["robot"]

    # Current joint state after env.reset()
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()

    # Find indices for the 4th joints by name
    l4_ids, _ = robot.find_joints(["zarm_l4_joint"])
    r4_ids, _ = robot.find_joints(["zarm_r4_joint"])
    l4_id = l4_ids[0]
    r4_id = r4_ids[0]

    # Set both 4th joints to desired angle, zero velocity
    joint_pos[:, l4_id] = angle_rad
    joint_pos[:, r4_id] = angle_rad
    joint_vel[:, l4_id] = 0.0
    joint_vel[:, r4_id] = 0.0

    # Push into the sim and targets
    robot.set_joint_position_target(joint_pos)
    robot.set_joint_velocity_target(joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)






class PouringKuavoV4ProMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic wrapper for KuavoV4Pro.
    
    This env handles TWO action formats:
    1. RECORDED actions (from teleoperation): Joint positions [14 arm + 20 finger]
    2. GENERATED actions (for Mimic): Pink IK format [7 left pose + 7 right pose + 20 finger]

    


    The key is that action_to_target_eef_pose() interprets recorded joint actions
    by reading the CURRENT EEF pose from observations (after the action was applied).
    """

  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pink_finger_perm = None

    def _compute_pink_finger_perm(self, device: torch.device):
        pink = self.action_manager._terms.get("pink", None)
        if pink is None:
            return None

        robot = self.scene["robot"]
        cfg_order = list(pink.cfg.hand_joint_names)
        internal_names = [robot.joint_names[i] for i in pink._hand_joint_ids]

        name_to_idx = {n: i for i, n in enumerate(cfg_order)}
        perm_list = [name_to_idx[n] for n in internal_names]

        return torch.tensor(perm_list, device=device, dtype=torch.long)

    def step(self, action: torch.Tensor):
        # Debug: print first few steps
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        if self._step_count < 50:
            robot = self.scene["robot"]
            jp = robot.data.joint_pos[0]
            print(f"[STEP {self._step_count}] action shape: {action.shape}")
            print(f"[STEP {self._step_count}] l4: {jp[16].item():.4f}, r4: {jp[17].item():.4f}")
            print(f"  action[0, :14]: {action[0, :14].tolist()}")
            self._step_count += 1
            
            
            print(f"  joint_pos[16] (l4): {jp[16].item():.4f}")
            print(f"  joint_pos[17] (r4): {jp[17].item():.4f}")
            print(f"  left_z target: {action[0, 2].item():.4f}, right_z target: {action[0, 9].item():.4f}")
        pink = self.action_manager._terms.get("pink", None)

        # Keep a copy of the original finger vector (cfg/teleop order) for debugging
        orig_fingers = None
        if pink is not None and action.shape[1] >= 34:
            orig_fingers = action[:, 14:34].detach().clone()  # shape (N, 20), in cfg order

            # Ensure perm exists AND is on the same device as action
            if (
                self._pink_finger_perm is None
                or self._pink_finger_perm.device != action.device
            ):
                self._pink_finger_perm = self._compute_pink_finger_perm(device=action.device)

            # Apply permutation to match pink._hand_joint_ids orders
            action = action.clone()
            action[:, 14:34] = action[:, 14:34].index_select(1, self._pink_finger_perm)

        # Run the real step (this is where joint targets get set)
        obs, reward, terminated, truncated, info = super().step(action)

        # ---- DEBUG AFTER super().step ----
        if pink is not None and action.shape[1] >= 34:
            robot = self.scene["robot"]

            left_joint_names = [
                "l_thumbCMC", "l_thumbMCP",
                "l_indexMCP", "l_indexPIP",
                "l_middleMCP", "l_middlePIP",
                "l_ringMCP", "l_ringPIP",
                "l_littleMCP", "l_littlePIP",
            ]
            right_joint_names = [
                "r_thumbCMC", "r_thumbMCP",
                "r_indexMCP", "r_indexPIP",
                "r_middleMCP", "r_middlePIP",
                "r_ringMCP", "r_ringPIP",
                "r_littleMCP", "r_littlePIP",
            ]

            left_ids, _ = robot.find_joints(left_joint_names)
            right_ids, _ = robot.find_joints(right_joint_names)

            # IMPORTANT: use orig_fingers for L/R action prints (cfg order),
            # because action[:,14:34] is now in Pink internal order (mixed).
            left_action_fingers = orig_fingers[0, :10]
            right_action_fingers = orig_fingers[0, 10:]

            # print("[DEBUG] L action (cfg order):", left_action_fingers.tolist())
            # print("[DEBUG] L target:", robot.data.joint_pos_target[0, left_ids].tolist())
            # print("[DEBUG] L actual:", robot.data.joint_pos[0, left_ids].tolist())
            # print("[DEBUG] R action (cfg order):", right_action_fingers.tolist())
            # print("[DEBUG] R target:", robot.data.joint_pos_target[0, right_ids].tolist())
            # print("---")

        return obs, reward, terminated, truncated, info

    # def step(self, action: torch.Tensor):
    #     # --- FIX: reorder finger dims to match pink._hand_joint_ids order ---
    #     pink = self.action_manager._terms.get("pink", None)
    #     if pink is not None:
    #         robot = self.scene["robot"]
    #         left_joint_names = [
    #             "l_thumbCMC", "l_thumbMCP",
    #             "l_indexMCP", "l_indexPIP",
    #             "l_middleMCP", "l_middlePIP",
    #             "l_ringMCP", "l_ringPIP",
    #             "l_littleMCP", "l_littlePIP",
    #         ]

    #         right_joint_names = [
    #             "r_thumbCMC", "r_thumbMCP",
    #             "r_indexMCP", "r_indexPIP",
    #             "r_middleMCP", "r_middlePIP",
    #             "r_ringMCP", "r_ringPIP",
    #             "r_littleMCP", "r_littlePIP",
    #         ]
    #         cfg_order = pink.cfg.hand_joint_names
    #         actual_order = [robot.joint_names[i] for i in pink._hand_joint_ids]

    #         name_to_idx = {n: i for i, n in enumerate(cfg_order)}
    #         perm = torch.tensor([name_to_idx[n] for n in actual_order], device=action.device)

    #         # action layout: [14 eef + 20 fingers]
    #         fingers = action[:, 14:34]
    #         action = action.clone()
    #         action[:, 14:34] = fingers.index_select(1, perm)
     

    #         left_ids, _ = robot.find_joints(left_joint_names)
    #         right_ids, _ = robot.find_joints(right_joint_names)

    #         print("LEFT IDS:", left_ids, [robot.joint_names[i] for i in left_ids])
    #         print("RIGHT IDS:", right_ids, [robot.joint_names[i] for i in right_ids])

    #         left_actual = robot.data.joint_pos[0, left_ids]
    #         left_target = robot.data.joint_pos_target[0, left_ids]

    #         # If this is PinkIK-format action: fingers live at action[0, 14:24] and [0, 24:34]
    #         left_action_fingers = action[0, 14:24]
    #         right_action_fingers = action[0, 24:34]

    #         print("[DEBUG] L action:", left_action_fingers.tolist())
    #         print("[DEBUG] L target:", left_target.tolist())
    #         print("[DEBUG] L actual:", left_actual.tolist())
    #         print("[DEBUG] R action:", right_action_fingers.tolist())
    #         print("[DEBUG] R target:", robot.data.joint_pos_target[0, right_ids].tolist())
        # ---------------------------------------------------------------

        return super().step(action)


        
    #     return obs, reward, terminated, truncated, info
    # def reset(self, *args, **kwargs):
    #     obs, info = super().reset(*args, **kwargs)

    #     # FORCE ARM POSE AFTER EVERY RESET
    #     set_fourth_joints_to_90(self)

    #     return obs, info
    # def reset(self, *args, **kwargs):
    #     obs, info = super().reset(*args, **kwargs)

    #     # Force-teleport ALL joints to the correct pose
    #     robot = self.scene["robot"]
    #     joint_pos = robot.data.default_joint_pos.clone()
    #     joint_vel = torch.zeros_like(joint_pos)
        
    #     robot.write_joint_state_to_sim(joint_pos, joint_vel)
    #     robot.set_joint_position_target(joint_pos)
    #     robot.set_joint_velocity_target(joint_vel)

    #     return obs, info
        
    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return current EEF pose from observations as 4x4 matrices."""
        if env_ids is None:
            env_ids = slice(None)

        eef_pos_name = f"{eef_name}_eef_pos"
        eef_quat_name = f"{eef_name}_eef_quat"

        pos = self.obs_buf["policy"][eef_pos_name][env_ids]
        rot = PoseUtils.matrix_from_quat(self.obs_buf["policy"][eef_quat_name][env_ids])
        return PoseUtils.make_pose(pos, rot)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF pose -> Pink IK action format.
        
        Pink IK expects: [left_pos(3), left_quat(4), right_pos(3), right_quat(4), fingers(20)]
        """

        # Extract from 4x4 pose matrices
        left_pos, left_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"])
        right_pos, right_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"])
        
        left_quat = PoseUtils.quat_from_matrix(left_rot)
        right_quat = PoseUtils.quat_from_matrix(right_rot)
        
        # Get gripper actions (finger joints)
        left_gripper = gripper_action_dict.get("left")
        right_gripper = gripper_action_dict.get("right")
        # DEBUG: Check values BEFORE any processing
        # if left_gripper is not None:
        #     print(f"[DEBUG INPUT] left_gripper min/max: {left_gripper.min():.4f} / {left_gripper.max():.4f}")
        # if right_gripper is not None:
        #     print(f"[DEBUG INPUT] right_gripper min/max: {right_gripper.min():.4f} / {right_gripper.max():.4f}")
    
        if left_gripper is None:
            left_gripper = torch.zeros(10, device=self.device)
        else:
            left_gripper = left_gripper.to(device=self.device)
            
        if right_gripper is None:
            right_gripper = torch.zeros(10, device=self.device)
        else:
            right_gripper = right_gripper.to(device=self.device)
        
        # Ensure correct shapes (squeeze batch dim if present)
        if left_pos.dim() > 1:
            left_pos = left_pos.squeeze(0)
        if left_quat.dim() > 1:
            left_quat = left_quat.squeeze(0)
        if right_pos.dim() > 1:
            right_pos = right_pos.squeeze(0)
        if right_quat.dim() > 1:
            right_quat = right_quat.squeeze(0)
        if left_gripper.dim() > 1:
            left_gripper = left_gripper.squeeze(0)
        if right_gripper.dim() > 1:
            right_gripper = right_gripper.squeeze(0)
        # print(f"[DEBUG AFTER SQUEEZE] left min/max: {left_gripper.min():.4f} / {left_gripper.max():.4f}")
        # print(f"[DEBUG AFTER SQUEEZE] right min/max: {right_gripper.min():.4f} / {right_gripper.max():.4f}")
    
        # Apply noise if specified
        if action_noise_dict is not None:
            if action_noise_dict.get("left") is not None:
                noise = float(action_noise_dict["left"])
                left_pos = left_pos + noise * torch.randn_like(left_pos)
                left_quat = left_quat + noise * torch.randn_like(left_quat)
                left_quat = left_quat / left_quat.norm()
            if action_noise_dict.get("right") is not None:
                noise = float(action_noise_dict["right"])
                right_pos = right_pos + noise * torch.randn_like(right_pos)
                right_quat = right_quat + noise * torch.randn_like(right_quat)
                right_quat = right_quat / right_quat.norm()
        
        # Concatenate into Pink IK action format (34D total)
        action = torch.cat([
            left_pos,       # 3D
            left_quat,      # 4D  
            right_pos,      # 3D
            right_quat,     # 4D
            left_gripper,   # 10D
            right_gripper,  # 10D
        ], dim=0)
        
        # DEBUG: Check final action
        # print(f"[DEBUG FINAL ACTION] fingers [14:24] min/max: {action[14:24].min():.4f} / {action[14:24].max():.4f}")
        # print(f"[DEBUG FINAL ACTION] fingers [24:34] min/max: {action[24:34].min():.4f} / {action[24:34].max():.4f}")
    
        return action

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert action -> target EEF poses.
        
        For RECORDED demos (joint position format), we return the CURRENT observed EEF pose.
        For GENERATED demos (Pink IK format), we extract poses from the action directly.
        
        We detect the format by checking if the action looks like quaternions (normalized).
        """
        # Check if this is Pink IK format by looking at quaternion normalization
        # Pink IK: [pos(3), quat(4), pos(3), quat(4), fingers(20)]
        # Joint pos: [arm_joints(14), fingers(20)]
        
        if action.dim() == 2:
            # Check if indices 3:7 look like a quaternion (norm ≈ 1)
            potential_quat = action[:, 3:7]
            quat_norms = potential_quat.norm(dim=-1)
            is_pink_ik_format = torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=0.1)
            
            if is_pink_ik_format:
                # Pink IK format - extract poses directly
                left_pos = action[:, 0:3]
                left_quat = action[:, 3:7]
                right_pos = action[:, 7:10]
                right_quat = action[:, 10:14]
                
                left_rot = PoseUtils.matrix_from_quat(left_quat)
                right_rot = PoseUtils.matrix_from_quat(right_quat)
                
                return {
                    "left": PoseUtils.make_pose(left_pos, left_rot),
                    "right": PoseUtils.make_pose(right_pos, right_rot),
                }
            else:
                # Joint position format - return current observed poses
                return {
                    "left": self.get_robot_eef_pose("left"),
                    "right": self.get_robot_eef_pose("right"),
                }
        else:
            # For 3D tensors (batched trajectories), assume we need current pose
            return {
                "left": self.get_robot_eef_pose("left"),
                "right": self.get_robot_eef_pose("right"),
            }

    # def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
    #     """
    #     Recorded finger actions (per hand) are in teleop order from ActionsCfg:
    #     [thumbCMC, thumbMCP, indexMCP, indexPIP, middleMCP, middlePIP, ringMCP, ringPIP, littleMCP, littlePIP]

    #     Pink IK expects (per hand) order from your printed hand_joint_names:
    #     [indexMCP, littleMCP, middleMCP, ringMCP, thumbCMC, indexPIP, littlePIP, middlePIP, ringPIP, thumbMCP]

    #     So we remap teleop -> pink before returning.
    #     """
    #     TELEOP_TO_PINK = torch.tensor([2, 8, 4, 6, 0, 3, 9, 5, 7, 1], device=actions.device)

    #     def remap(x: torch.Tensor, dim: int) -> torch.Tensor:
    #         return torch.index_select(x, dim, TELEOP_TO_PINK)

    #     if actions.dim() == 3:
    #         # (N, T, D)
    #         left_teleop  = actions[:, :, -20:-10]  # last 20..10: left hand (10)
    #         right_teleop = actions[:, :, -10:]     # last 10: right hand (10)
    #         left_pink  = remap(left_teleop, dim=2)
    #         right_pink = remap(right_teleop, dim=2)

    #     elif actions.dim() == 2:
    #         # (N, D) or (T, D)
    #         left_teleop  = actions[:, -20:-10]
    #         right_teleop = actions[:, -10:]
    #         left_pink  = remap(left_teleop, dim=1)
    #         right_pink = remap(right_teleop, dim=1)

    #     else:
    #         raise ValueError(f"Unexpected actions shape: {tuple(actions.shape)}")

    #     return {"left": left_pink, "right": right_pink}
    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        # teleop order: [thumbCMC, thumbMCP, indexMCP, indexPIP, middleMCP, middlePIP, ringMCP, ringPIP, littleMCP, littlePIP]
        if actions.dim() == 3:
            left  = actions[:, :, -20:-10]
            right = actions[:, :, -10:]
        elif actions.dim() == 2:
            left  = actions[:, -20:-10]
            right = actions[:, -10:]
        else:
            raise ValueError(f"Unexpected actions shape: {tuple(actions.shape)}")

        return {"left": left, "right": right}
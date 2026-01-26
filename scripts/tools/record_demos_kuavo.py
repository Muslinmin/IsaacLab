# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")



parser.add_argument(
    "--reset_delay",
    type=float,
    default=2.0,
    help="Delay in seconds after reset before recording starts. Default is 2.0 seconds.",
)


parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=20, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=5, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=True,
    help="Enable Pinocchio.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


print("Recording for kuavov4 pro pouring to kuavo's v4 pro")
kuavo_task = "kuavoV4Pro-Pouring-Base"
print("Cameras enabled by default")
args_cli.enable_cameras = True


app_launcher_args = vars(args_cli)


import pinocchio  # noqa: F401


# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


# Third-party imports
import gymnasium as gym
import os
import time
import torch
import math

# Omniverse logger
import omni.log
import omni.ui as ui

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
#touched - for teleoperation of the kuavo robot
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.devices import kuavoXR
#touched - for keyboard handling to stop/start or reset env - if needed
from isaaclab.devices.keyboard import KuavoKeyboardDevice



import isaaclab_mimic.envs  # noqa: F401

from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions


#touched - kuavoV4ProBreakfast_env_cfg.py
# if args_cli.enable_pinocchio:
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    # import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

from collections.abc import Callable

#touched - used for kuavoV4ProBreakfast_env_cfg.py
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg



from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode



#touched - kuavoV4ProBreakfast_env_cfg.py is within the following package
import isaaclab_tasks  # noqa: F401


from pxr import UsdPhysics
import omni.usd



from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def debug_validate_cfg(env_cfg):
    print("\n========== ENV CFG VALIDATION ==========")

    required_sections = [
        "scene", "sim", "observations", "actions",
        "rewards", "terminations", "curricula",
        "randomization", "recorders"
    ]

    for sec in required_sections:
        if not hasattr(env_cfg, sec):
            print(f"[ERROR] Missing section: env_cfg.{sec}")
        else:
            sec_val = getattr(env_cfg, sec)
            if sec_val is None:
                print(f"[ERROR] Section {sec} is None!")
            else:
                # Print only high level summary to avoid spam
                try:
                    print(f"[OK] {sec}: {type(sec_val)} with {len(sec_val.__dict__)} attributes")
                except:
                    print(f"[OK] {sec}: {type(sec_val)}")

    print("========================================\n")


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    """Create and configure the environment configuration.

    Parses the environment configuration and makes necessary adjustments for demo recording.
    Extracts the success termination function and configures the recorder manager.

    Args:
        output_dir: Directory where recorded demonstrations will be saved
        output_file_name: Name of the file to store the demonstrations

    Returns:
        tuple[isaaclab_tasks.utils.parse_cfg.EnvCfg, Optional[object]]: A tuple containing:
            - env_cfg: The configured environment configuration
            - success_term: The success termination object or None if not available

    Raises:
        Exception: If parsing the environment configuration fails
    """
    # parse configuration
    try:
        env_cfg = parse_env_cfg(kuavo_task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = kuavo_task.split(":")[-1]
    except Exception as e:
        omni.log.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # if args_cli.xr:
    #     # If cameras are not enabled and XR is enabled, remove camera configs
    #     if not args_cli.enable_cameras:
    #         env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """Create the environment from the configuration.

    Args:
        env_cfg: The environment configuration object that defines the environment properties.
            This should be an instance of EnvCfg created by parse_env_cfg().

    Returns:
        gym.Env: A Gymnasium environment instance for the specified task.

    Raises:
        Exception: If environment creation fails for any reason.
    """
    try:
        # debug_validate_cfg(env_cfg)
        env = gym.make(kuavo_task, cfg=env_cfg).unwrapped
        return env
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        exit(1)



def print_applied_arm_actuator_gains(env):
    """Print the ACTUAL stiffness/damping values for arm joints from Isaac Lab actuators."""
    robot = env.scene["robot"]
    
    LEFT_ARM_JOINTS = [f"zarm_l{i}_joint" for i in range(1, 8)]
    RIGHT_ARM_JOINTS = [f"zarm_r{i}_joint" for i in range(1, 8)]
    
    print("\n" + "="*70)
    print("ARM ACTUATOR VALUES (from Isaac Lab)")
    print("="*70)
    
    # Try to find arm actuators - they might be grouped or per-joint
    print("\n=== ALL AVAILABLE ACTUATORS ===")
    for name in robot.actuators.keys():
        print(f"  {name}")
    
    print("\n=== LEFT ARM JOINTS ===")
    for jname in LEFT_ARM_JOINTS:
        found = False
        # Check if there's a per-joint actuator (like "LA_zarm_l1_joint")
        for prefix in ["LA_", "L_", ""]:
            actuator_key = f"{prefix}{jname}"
            if actuator_key in robot.actuators:
                act = robot.actuators[actuator_key]
                stiff = act.stiffness[0, 0].item()
                damp = act.damping[0, 0].item()
                print(f"  {jname}: stiffness={stiff:.6f}, damping={damp:.10f}")
                found = True
                break
        
        # Check if it's part of a grouped actuator (like "left_arm")
        if not found:
            for act_name, act in robot.actuators.items():
                if jname in act.joint_names:
                    idx = act.joint_names.index(jname)
                    stiff = act.stiffness[0, idx].item()
                    damp = act.damping[0, idx].item()
                    print(f"  {jname}: stiffness={stiff:.6f}, damping={damp:.10f}  (from '{act_name}')")
                    found = True
                    break
        
        if not found:
            print(f"  {jname}: NOT FOUND in any actuator")
    
    print("\n=== RIGHT ARM JOINTS ===")
    for jname in RIGHT_ARM_JOINTS:
        found = False
        for prefix in ["RA_", "R_", ""]:
            actuator_key = f"{prefix}{jname}"
            if actuator_key in robot.actuators:
                act = robot.actuators[actuator_key]
                stiff = act.stiffness[0, 0].item()
                damp = act.damping[0, 0].item()
                print(f"  {jname}: stiffness={stiff:.6f}, damping={damp:.10f}")
                found = True
                break
        
        if not found:
            for act_name, act in robot.actuators.items():
                if jname in act.joint_names:
                    idx = act.joint_names.index(jname)
                    stiff = act.stiffness[0, idx].item()
                    damp = act.damping[0, idx].item()
                    print(f"  {jname}: stiffness={stiff:.6f}, damping={damp:.10f}  (from '{act_name}')")
                    found = True
                    break
        
        if not found:
            print(f"  {jname}: NOT FOUND in any actuator")
    
    print("="*70 + "\n")



def setup_kuavo_teleop_interface():
    teleop_interface = None
    device_cfg = kuavoXR.KuavoRosJointDeviceCfg(
        cmd_sub_address="tcp://192.168.8.102:5555",
        state_pub_bind="tcp://*:5556",
        num_dofs=34,  # arms + fingers
    )
    #no callbacks added here
    try:
        teleop_interface = kuavoXR.KuavoRosJointDevice(device_cfg)
        omni.log.warn(f"Created KUAVO TELEOPERATION INTERFACE~~~~~~~~~~~~~~~~~~~~~~~~")
    except Exception as e:
        omni.log.error(f"Failed to create teleop device: {e}")
        exit(1)
    return teleop_interface

def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """Set up the user interface elements.

    Creates instruction display and UI window with labels for showing information
    to the user during demonstration recording.

    Args:
        label_text: Text to display showing current recording status
        env: The environment instance for which UI is being created

    Returns:
        InstructionDisplay: The configured instruction display object
    """
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False

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


# def handle_reset(
#     env: gym.Env, success_step_count: int, instruction_display: InstructionDisplay, label_text: str
# ) -> int:
#     """Handle resetting the environment.

#     Resets the environment, recorder manager, and related state variables.
#     Updates the instruction display with current status.

#     Args:
#         env: The environment instance to reset
#         success_step_count: Current count of consecutive successful steps
#         instruction_display: The display object to update
#         label_text: Text to display showing current recording status

#     Returns:
#         int: Reset success step count (0)
#     """
#     print("Resetting environment...")
#     env.sim.reset()
#     env.recorder_manager.reset()
#     env.reset()
#     set_fourth_joints_to_90(env, -2.1)
#     success_step_count = 0
#     instruction_display.show_demo(label_text)
#     return success_step_count

def handle_reset(
    env: gym.Env, 
    success_step_count: int, 
    instruction_display: InstructionDisplay, 
    label_text: str,
    teleop_interface: object,
    should_have_delay: bool
) -> tuple[int, bool, float]:
    """Handle resetting the environment with optional countdown delay.

    Resets the environment, recorder manager, and related state variables.
    Updates the instruction display with current status.
    Optionally starts a countdown before recording begins.

    Args:
        env: The environment instance to reset
        success_step_count: Current count of consecutive successful steps
        instruction_display: The display object to update
        label_text: Text to display showing current recording status
        teleop_interface: Teleoperation interface for sending camera frames during delay
        should_have_delay: Whether to activate countdown delay after reset

    Returns:
        tuple[int, bool, float]: (success_step_count=0, delay_active, delay_start_time)
    """
    print("\n" + "="*60)
    print("RESETTING ENVIRONMENT")
    print("="*60)
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    set_fourth_joints_to_90(env, -2.1)
    success_step_count = 0
    
    if should_have_delay:
        # Update UI with countdown message
        delay_message = f"{label_text}\n\n🔄 Reset complete!\n⏱️  Recording starts in {args_cli.reset_delay:.1f} seconds...\n👋 Move your arms to starting position now!"
        instruction_display.show_demo(delay_message)
        
        print(f"\n✓ Reset complete!")
        print(f"⏱️  Countdown: Recording starts in {args_cli.reset_delay:.1f} seconds...")
        print(f"👋 Move your arms to the starting position now!\n")
        
        # Return delay state (will be handled in main loop)
        return success_step_count, True, time.time()
    else:
        # No delay - just show ready message
        instruction_display.show_demo(label_text + "\n\n✓ Reset complete! Press W to start recording.")
        print(f"\n✓ Reset complete! Ready to record.")
        print(f"Press W to start recording.\n")
        
        # No delay
        return success_step_count, False, 0.0




def compute_joint_ids(env):
    robot = env.scene["robot"]
    joint_ids = []

    # Arms
    arm_joint_names = [f"zarm_l{i}_joint" for i in range(1, 8)] + \
                      [f"zarm_r{i}_joint" for i in range(1, 8)]
    for name in arm_joint_names:
        ids, _ = robot.find_joints([name])
        print(f"Joint names {ids} {name}")
        joint_ids.append(ids[0])

    # Left hand
    left_hand_names = [
        "l_thumbCMC", "l_thumbMCP",
        "l_indexMCP", "l_indexPIP",
        "l_middleMCP", "l_middlePIP",
        "l_ringMCP", "l_ringPIP",
        "l_littleMCP", "l_littlePIP",
    ]
    for name in left_hand_names:
        ids, _ = robot.find_joints([name])
        joint_ids.append(ids[0])

    # Right hand
    right_hand_names = [
        "r_thumbCMC", "r_thumbMCP",
        "r_indexMCP", "r_indexPIP",
        "r_middleMCP", "r_middlePIP",
        "r_ringMCP", "r_ringPIP",
        "r_littleMCP", "r_littlePIP",
    ]
    for name in right_hand_names:
        ids, _ = robot.find_joints([name])
        joint_ids.append(ids[0])

    return joint_ids



def read_joint_drive_gains_from_usd(joint_names, drive_token_candidates=("angular", "rotX", "rotY", "rotZ")):
    """
    Reads stiffness/damping/maxForce from USD joint drives using UsdPhysics.DriveAPI.

    drive_token_candidates:
      - common instance names used for drives. 'angular' is typical for revolute joints.
      - D6 joints may use rotX/rotY/rotZ.
    Returns:
      dict: joint_name -> {"stiffness": val|None, "damping": val|None, "max_force": val|None, "drive_token": str|None}
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage not available. Call after env.reset() / stage load.")

    target = set(joint_names)
    gains = {}

    for prim in stage.Traverse():
        name = prim.GetName()
        if name not in target:
            continue

        found = False
        for tok in drive_token_candidates:
            drive_api = UsdPhysics.DriveAPI.Get(prim, tok)
            if not drive_api:
                continue

            # Attributes may or may not exist; Get() returns None if unset
            stiffness = drive_api.GetStiffnessAttr().Get() if drive_api.GetStiffnessAttr() else None
            damping = drive_api.GetDampingAttr().Get() if drive_api.GetDampingAttr() else None
            max_force = drive_api.GetMaxForceAttr().Get() if drive_api.GetMaxForceAttr() else None

            gains[name] = {
                "stiffness": stiffness,
                "damping": damping,
                "max_force": max_force,
                "drive_token": tok,
            }
            found = True
            break

        if not found:
            gains[name] = {"stiffness": None, "damping": None, "max_force": None, "drive_token": None}

    # ensure all requested joints exist in dict
    for j in joint_names:
        gains.setdefault(j, {"stiffness": None, "damping": None, "max_force": None, "drive_token": None})

    return gains
def print_joint_drive_gains_from_usd():
    ## SIM JOINT READINGS ( NOTE: THIS DOES NOT APPLY WHEN YOU HAVE A CODE THAT OVVERIDES THE STIFFNESS AND DAMPING VALUES; INSTEAD, USE THE ONE ABOVE)
    LEFT_FINGERS = ["l_thumbCMC","l_thumbMCP","l_indexMCP","l_indexPIP","l_middleMCP","l_middlePIP","l_ringMCP","l_ringPIP","l_littleMCP","l_littlePIP"]
    RIGHT_FINGERS = ["r_thumbCMC","r_thumbMCP","r_indexMCP","r_indexPIP","r_middleMCP","r_middlePIP","r_ringMCP","r_ringPIP","r_littleMCP","r_littlePIP"]

    usd_left = read_joint_drive_gains_from_usd(LEFT_FINGERS)
    usd_right = read_joint_drive_gains_from_usd(RIGHT_FINGERS)

    print("=== LEFT finger drives ===")
    for j in LEFT_FINGERS:
        print(j, usd_left[j])

    print("=== RIGHT finger drives ===")
    for j in RIGHT_FINGERS:
        print(j, usd_right[j])

    left_stiffness  = [usd_left[j]["stiffness"] for j in LEFT_FINGERS]
    left_damping    = [usd_left[j]["damping"] for j in LEFT_FINGERS]
    right_stiffness = [usd_right[j]["stiffness"] for j in RIGHT_FINGERS]
    right_damping   = [usd_right[j]["damping"] for j in RIGHT_FINGERS]

    print("LEFT stiffness:", left_stiffness)
    print("LEFT damping  :", left_damping)
    print("RIGHT stiffness:", right_stiffness)
    print("RIGHT damping  :", right_damping)
def print_applied_finger_actuator_gains(env):
    """Print the ACTUAL stiffness/damping values being used in simulation."""
    robot = env.scene["robot"]
    
    LEFT_FINGERS = ["l_thumbCMC","l_thumbMCP","l_indexMCP","l_indexPIP","l_middleMCP","l_middlePIP","l_ringMCP","l_ringPIP","l_littleMCP","l_littlePIP"]
    RIGHT_FINGERS = ["r_thumbCMC","r_thumbMCP","r_indexMCP","r_indexPIP","r_middleMCP","r_middlePIP","r_ringMCP","r_ringPIP","r_littleMCP","r_littlePIP"]
    
    print("\n" + "="*60)
    print("ACTUALLY APPLIED VALUES (from Isaac Lab actuators)")
    print("="*60)
    
    print("\n=== LEFT FINGER ACTUATORS ===")
    for jname in LEFT_FINGERS:
        actuator_key = f"L_{jname}"  # Your naming convention
        if actuator_key in robot.actuators:
            act = robot.actuators[actuator_key]
            # Stiffness/damping are tensors of shape (num_envs, num_joints_in_actuator)
            # Since each actuator has 1 joint, index [0, 0]
            stiff = act.stiffness[0, 0].item()
            damp = act.damping[0, 0].item()
            print(f"  {jname}: stiffness={stiff:.6f}, damping={damp:.10f}")
        else:
            print(f"  {jname}: actuator '{actuator_key}' not found")
    
    print("\n=== RIGHT FINGER ACTUATORS ===")
    for jname in RIGHT_FINGERS:
        actuator_key = f"R_{jname}"  # Your naming convention
        if actuator_key in robot.actuators:
            act = robot.actuators[actuator_key]
            stiff = act.stiffness[0, 0].item()
            damp = act.damping[0, 0].item()
            print(f"  {jname}: stiffness={stiff:.6f}, damping={damp:.10f}")
        else:
            print(f"  {jname}: actuator '{actuator_key}' not found")
    
    # Also print all available actuators for debugging
    print("\n=== ALL AVAILABLE ACTUATORS ===")
    for name in robot.actuators.keys():
        print(f"  {name}")
    
    print("="*60 + "\n")


def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object | None,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Sets up callback functions for teleop device, initializes the UI,
    and runs the main loop that processes user inputs and environment steps.
    Records demonstrations when success conditions are met.

    Args:
        env: The environment instance
        teleop_interface: Optional teleop interface (will be created if None)
        success_term: The success termination object or None if not available
        rate_limiter: Optional rate limiter to control simulation speed

    Returns:
        int: Number of successful demonstrations recorded
    """
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    running_recording_instance = False

    reset_delay_active = False
    reset_delay_start_time = 0.0
    should_start_with_delay = False  # Track if we want countdown when startin
    # Callback closures for the teleop device

    def reset_recording_instance():
        nonlocal should_reset_recording_instance, running_recording_instance, should_start_with_delay
        was_recording = running_recording_instance  # Remember if we were recording
        should_reset_recording_instance = True
        running_recording_instance = False  # Stop recording during reset
        should_start_with_delay = was_recording  # Only delay if we were recording
        
        if was_recording:
            print("Recording instance reset requested (was recording → will have delay)")
        else:
            print("Recording instance reset requested (was NOT recording → NO delay)")

    def start_recording_instance():
        nonlocal reset_delay_active, reset_delay_start_time, should_start_with_delay
        # Trigger countdown delay
        should_start_with_delay = True
        reset_delay_active = True
        reset_delay_start_time = time.time()
        print("Start recording requested → countdown begins")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("Recording paused")
    # Set up teleoperation callbacks
    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }
    #retrieve joint ids
    joint_ids = compute_joint_ids(env)
    print(f"JOINT IDS: {joint_ids}")
    teleop_interface = setup_kuavo_teleop_interface()
    # Create keyboard device
    kb_cfg = KuavoKeyboardDevice.KuavoKeyboardDeviceCfg()
    kb_device = KuavoKeyboardDevice.KuavoKeyboardDevice(kb_cfg)


    kb_device.add_callback("START", start_recording_instance)   # W
    kb_device.add_callback("STOP", stop_recording_instance)     # S
    kb_device.add_callback("RESET", reset_recording_instance)   # R or T

    # Reset before starting
    # env.sim.reset()
    env.reset()
    print_applied_arm_actuator_gains(env)




    joint_ids = compute_joint_ids(env)
    robot = env.scene["robot"]
    print_applied_finger_actuator_gains(env)
    print("=== TELEOP INDEX → (articulation_index, joint_name) ===")
    for i, jid in enumerate(joint_ids):
        print(f"{i:2d}: {jid:2d}  {robot.joint_names[jid]}")
    print(f"Recording initial status: {running_recording_instance}")
    

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)

    subtasks = {}

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # Check for any key pressed
            # Get XR command
            action = teleop_interface.advance()
            # Expand to batch dimension
            actions = action.repeat(env.num_envs, 1)
            ego_rgb = env.scene["cam_egoview"].data.output["rgb"][0]  # [H,W,3]
            ego_rgb_np = ego_rgb.cpu().numpy()
            #send images
            teleop_interface.send_camera_frame("cam_egoview", ego_rgb_np)
            if reset_delay_active:
                elapsed = time.time() - reset_delay_start_time
                remaining = args_cli.reset_delay - elapsed
                
                if remaining > 0:
                    # Always print (carriage return overwrites previous line)
                    print(f"\r⏱️  Recording starts in {remaining:.1f} seconds...   ", end='', flush=True)
                    
                    # Allow teleop to run but don't record - just step the simulation
                    env.step(actions)
                    
                    # Send joint state during countdown
                    joint_pos_tensor = env.scene["robot"].data.joint_pos[0, joint_ids]
                    joint_pos_np = joint_pos_tensor.cpu().numpy()
                    teleop_interface.send_joint_state(joint_pos_np)
                else:
                    # Delay expired - start recording
                    print("\n\n✅ RECORDING STARTED!")
                    print("="*60 + "\n")
                    reset_delay_active = False
                    running_recording_instance = True
                    instruction_display.show_demo(label_text)
                
                # Skip rest of loop during countdown
                if rate_limiter:
                    rate_limiter.sleep(env)
                continue
            # Perform action on environment
            if running_recording_instance:
                # Compute actions based on environment
                obv = env.step(actions)
                joint_pos_tensor = env.scene["robot"].data.joint_pos[0, joint_ids]  # shape [num_sim_joints]
                joint_pos_np = joint_pos_tensor.cpu().numpy()
                teleop_interface.send_joint_state(joint_pos_np)
                if subtasks is not None:
                    if subtasks == {}:
                        subtasks = obv[0].get("subtask_terms")
                    elif subtasks:
                        show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
            else:
                env.sim.render()

            # Check for success condition
            success_step_count, success_reset_needed = process_success_condition(env, success_term, success_step_count)
            if success_reset_needed:
                should_reset_recording_instance = True

            # Update demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            # Check if we've reached the desired number of demos
            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            # Handle reset if requested
            # if should_reset_recording_instance:
            #     success_step_count = handle_reset(env, success_step_count, instruction_display, label_text)
            #     should_reset_recording_instance = False
            if should_reset_recording_instance:
                success_step_count, reset_delay_active, reset_delay_start_time = handle_reset(
                    env, success_step_count, instruction_display, label_text, teleop_interface, should_start_with_delay
                )
                should_reset_recording_instance = False
                should_start_with_delay = False  # Reset the flag

            # Check if simulation is stopped
            if env.sim.is_stopped():
                break

            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)

    return current_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using teleop interfaces.

    Main function that orchestrates the entire process:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Runs the simulation loop to collect demonstrations
    5. Cleans up resources when done

    Raises:
        Exception: Propagates exceptions from any of the called functions
    """
    # if handtracking is selected, rate limiting is achieved via OpenXR
    if args_cli.xr:
        rate_limiter = None
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        # Assign the teleop visualization manager to the visualization system
        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # Set up output directories
    output_dir, output_file_name = setup_output_directories()

    # # Create and configure environment
    global env_cfg  
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    # # Create environment
    env = create_environment(env_cfg)

    # # Run simulation loop
    current_recorded_demo_count = run_simulation_loop(env, None, success_term, rate_limiter)

    # Clean up
    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

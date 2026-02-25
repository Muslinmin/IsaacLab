# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg

# from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from . import mdp

from isaaclab_assets.robots.kuavoV4Pro import KUAVO_V4PRO_CFG 

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.actuators import ImplicitActuatorCfg


from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


from pxr import Usd, UsdGeom, Gf
import omni.usd

# PhysX USD schema modules (names can vary slightly across Isaac Sim versions)
from pxr import PhysxSchema, UsdPhysics

# ============================================================================
# LIQUID PARTICLE SPHERES — Using RigidObjectCollectionCfg
# ============================================================================
#
# Single scene entity "liquid_particles" containing 24 rigid spheres.
# Accessed as: env.scene["liquid_particles"]
# Data shape:  env.scene["liquid_particles"].data.object_pos_w → (num_envs, 24, 3)
#
# This is the performant approach for multi-env (96 envs × 24 spheres).
# One tensor lookup, one vectorized check — no per-sphere Python loops.
# ============================================================================
from isaaclab.assets import RigidObjectCollectionCfg

_PX, _PY, _PZ = 0.20093, 0.43066, 1.0694

# 2×2 grid spacing and z-layer spacing (meters)
_GRID_SP = 0.015
_LAYER_SP = 0.02

def _build_liquid_particle_collection(
    color: tuple[float, float, float] = (0.2, 0.8, 0.2),
    origin: tuple[float, float, float] = (_PX, _PY, _PZ),
    prefix: str = "liq",
) -> RigidObjectCollectionCfg:
    px, py, pz = origin
    grid = [
        (-_GRID_SP / 2, -_GRID_SP / 2),
        ( _GRID_SP / 2, -_GRID_SP / 2),
        (-_GRID_SP / 2,  _GRID_SP / 2),
        ( _GRID_SP / 2,  _GRID_SP / 2),
    ]
    rigid_objects = {}
    idx = 0
    for layer in range(6):
        z_off = layer * _LAYER_SP
        for dx, dy in grid:
            name = f"{prefix}_sphere_{idx:02d}"
            rigid_objects[name] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/liquid_particle_{name}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=[px + dx, py + dy, pz + z_off],
                    rot=[1.0, 0.0, 0.0, 0.0],
                ),
                spawn=sim_utils.SphereCfg(
                    radius=0.005,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.1,
                        angular_damping=0.2,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            )
            idx += 1
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)
# Left cup (green)
_LIQUID_PARTICLES_CFG = _build_liquid_particle_collection(
    color=(0.2, 0.8, 0.2),
    origin=(-0.19789, 0.46173, 1.03684),
    prefix="left",
)

# Right cup (pink)
_LIQUID_PARTICLES_CFG_2 = _build_liquid_particle_collection(
    color=(1.0, 0.753, 0.796),
    origin=(0.20093, 0.43066, 1.0694),
    prefix="right",
)


# ---------------------------
# Finger joint gain tuning code here
# ---------------------------
LEFT_FINGERS = [
    "l_thumbCMC", "l_thumbMCP",
    "l_indexMCP", "l_indexPIP",
    "l_middleMCP", "l_middlePIP",
    "l_ringMCP", "l_ringPIP",
    "l_littleMCP", "l_littlePIP",
]
RIGHT_FINGERS = [
    "r_thumbCMC", "r_thumbMCP",
    "r_indexMCP", "r_indexPIP",
    "r_middleMCP", "r_middlePIP",
    "r_ringMCP", "r_ringPIP",
    "r_littleMCP", "r_littlePIP",
]

# Baseline values (copied from USD prints)
LEFT_FINGER_STIFFNESS_BASE = [
    2.3181025981903076, 1.1247035264968872,
    2.387361526489258,  1.1263597011566162,
    2.426273822784424,  1.1305006742477417,
    2.3825979232788086, 1.1235071420669556,
    1.2559813261032104, 1.1207642555236816,
]
LEFT_FINGER_DAMPING_BASE = [
    0.0009272410534322262, 0.00044988139416091144,
    0.0009549445821903646, 0.0004505438555497676,
    0.0009705095435492694, 0.0004522002418525517,
    0.0009530391544103622, 0.00044940283987671137,
    0.0005023925332352519, 0.0004483057127799839,
]
RIGHT_FINGER_STIFFNESS_BASE = [
    0.2776448428630829,   0.09061048179864883,
    0.30723679065704346,  0.035638388246297836,
    0.3706650733947754,   0.0442572720348835,
    0.28632596135139465,  0.02746584080159664,
    0.2208922803401947,   0.019619882106781006,
]
RIGHT_FINGER_DAMPING_BASE = [
    0.00011105793964816257, 0.00003624419332481922,
    0.00012289472215343267, 0.00001425535538146505,
    0.00014826603000983596, 0.00001770290873537306,
    0.00011453039041953161, 0.00001098633674700977,
    0.00008835690823616460, 0.00000784795247454895,
]



# GUIDE
# l_thumbCMC, l_thumbMCP,    (index 0, 1)
# l_indexMCP, l_indexPIP,    (index 2, 3)
# l_middleMCP, l_middlePIP,  (index 4, 5)
# l_ringMCP, l_ringPIP,      (index 6, 7)
# l_littleMCP, l_littlePIP,  (index 8, 9)  <-- PINKY

LEFT_FINGER_STIFFNESS_OVERRIDE: list[float] = [
    2.3131025981903076, 1.1197035264968872,
    2.382361526489258,  1.1213597011566162,
    2.421273822784424,  1.1255006742477417,
    2.3775979232788086, 1.1185071420669556,
    1.2559813261032104, 1.1207642555236816, 
] #-0.005 for all except for pinky

LEFT_FINGER_DAMPING_OVERRIDE: list[float] = [
    0.12593324105343223, 0.1254558813941609,
    0.12596094458219036, 0.12545654385554977,
    0.12597650954354927, 0.12545820024185256,
    0.12595903915441037, 0.12545540283987672,
    0.09425239253323525, 0.09419830571277998,
]
RIGHT_FINGER_STIFFNESS_OVERRIDE: list[float] = [
    0.3276448428630829,  0.14061048179864883,
    0.3572367906570435,  0.08563838824629784,
    0.4206650733947754,  0.0942572720348835,
    0.33632596135139465, 0.07746584080159664,
    0.2708922803401947,  0.06961988210678101,
]
RIGHT_FINGER_DAMPING_OVERRIDE: list[float] = [
    0.012611057939648163, 0.01253624419332482,
    0.012622894722153433, 0.012514255355381466,
    0.012648266030009836, 0.012517702908735373,
    0.012614530390419532, 0.01251098633674701,
    0.012588356908236165, 0.01250784795247455,
]


# Optional: easy scalar deltas you can play with (set to 0.0 by default)
# NOT USED
LEFT_STIFFNESS_DELTA = 0 
LEFT_DAMPING_DELTA = 0
RIGHT_STIFFNESS_DELTA = 0
RIGHT_DAMPING_DELTA = 0



def _build_per_joint_finger_actuators(prefix: str, joint_names: list[str],
                                      stiffness_list: list[float], damping_list: list[float]) -> dict[str, ImplicitActuatorCfg]:
    actuators: dict[str, ImplicitActuatorCfg] = {}
    for i, jname in enumerate(joint_names):
        key = f"{prefix}_{jname}"
        actuators[key] = ImplicitActuatorCfg(
            joint_names_expr=[jname],
            stiffness=float(stiffness_list[i]),
            damping=float(damping_list[i]),
        )
    return actuators




def _select_finger_gains():
    def _pick(base, override, n, name):
        if len(override) == 0:
            return base
        if len(override) != n:
            raise ValueError(f"{name} override must be length {n}, got {len(override)}.")
        return override

    left_kp = _pick(LEFT_FINGER_STIFFNESS_BASE, LEFT_FINGER_STIFFNESS_OVERRIDE, len(LEFT_FINGERS), "Left stiffness")
    left_kd = _pick(LEFT_FINGER_DAMPING_BASE,   LEFT_FINGER_DAMPING_OVERRIDE,   len(LEFT_FINGERS), "Left damping")
    right_kp = _pick(RIGHT_FINGER_STIFFNESS_BASE, RIGHT_FINGER_STIFFNESS_OVERRIDE, len(RIGHT_FINGERS), "Right stiffness")
    right_kd = _pick(RIGHT_FINGER_DAMPING_BASE,   RIGHT_FINGER_DAMPING_OVERRIDE,   len(RIGHT_FINGERS), "Right damping")

    # apply scalar deltas (optional)
    left_kp = [v + LEFT_STIFFNESS_DELTA for v in left_kp]
    left_kd = [v + LEFT_DAMPING_DELTA for v in left_kd]
    right_kp = [v + RIGHT_STIFFNESS_DELTA for v in right_kp]
    right_kd = [v + RIGHT_DAMPING_DELTA for v in right_kd]
    return left_kp, left_kd, right_kp, right_kd


def _maybe_override_finger_actuators(base_cfg: ArticulationCfg) -> ArticulationCfg:
    """
    If override arrays are provided (non-empty), replace broad hand actuators with per-joint ones.
    Otherwise: keep USD-inherited behavior (your current setup).
    """
    wants_override = (
        len(LEFT_FINGER_STIFFNESS_OVERRIDE) > 0 or len(LEFT_FINGER_DAMPING_OVERRIDE) > 0 or
        len(RIGHT_FINGER_STIFFNESS_OVERRIDE) > 0 or len(RIGHT_FINGER_DAMPING_OVERRIDE) > 0
    )
    if not wants_override:
        # No overrides -> keep original (hands inherit USD because stiffness/damping are None in KUAVO_V4PRO_CFG)
        return base_cfg

    left_kp, left_kd, right_kp, right_kd = _select_finger_gains()
    print("~~~~~~~~~~~~~~~~~~[FingerOverride] Using LEFT damping:", left_kd)
    # Build one actuator per joint
    finger_actuators: dict[str, ImplicitActuatorCfg] = {}
    # finger_actuators.update(_build_per_joint_finger_actuators("L", LEFT_FINGERS, left_kp, left_kd))
    # finger_actuators.update(_build_per_joint_finger_actuators("R", RIGHT_FINGERS, right_kp, right_kd))
    finger_actuators.update(_build_per_joint_finger_actuators("L", LEFT_FINGERS, left_kp, left_kd))
    finger_actuators.update(_build_per_joint_finger_actuators("R", RIGHT_FINGERS, right_kp, right_kd))

    # IMPORTANT: remove broad hand actuators to avoid "multiple matches" errors
    new_actuators = dict(base_cfg.actuators)
    new_actuators.pop("left_hand", None)
    new_actuators.pop("right_hand", None)

    # Merge per-joint actuators
    new_actuators.update(finger_actuators)

    return base_cfg.replace(actuators=new_actuators)



# ---------------------------
# Finger joint gain tuning code end
# ---------------------------

# ---------------------------
# ARM joint gain tuning code here
# ---------------------------
LEFT_ARM_JOINTS = [f"zarm_l{i}_joint" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"zarm_r{i}_joint" for i in range(1, 8)]

# Baseline values - FILL THESE IN after running print_applied_arm_actuator_gains()
# Format: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
LEFT_ARM_STIFFNESS_BASE: list[float] = [
    37181.226562, 29939.884766, 29791.384766, 25459.476562, 6427.223145, 2443.445068, 1117.259644,
]
LEFT_ARM_DAMPING_BASE: list[float] = [
    14.8724908829, 11.9759540558, 11.9165534973, 10.1837902069, 2.5708892345, 0.9773780107, 0.4469038844,
]
RIGHT_ARM_STIFFNESS_BASE: list[float] = [
    36566.347656, 30312.841797, 30164.755859, 25822.480469, 6366.853516, 2183.256836, 678.476257,
]
RIGHT_ARM_DAMPING_BASE: list[float] = [
    14.6265392303, 12.1251363754, 12.0659036636, 10.3289928436, 2.5467414856, 0.8733027577, 0.2713904977,
]

# Override arrays - Set to empty [] to use BASE values
# To add 25% more damping, we'll compute it automatically below
LEFT_ARM_STIFFNESS_OVERRIDE: list[float] = []
LEFT_ARM_DAMPING_OVERRIDE: list[float] = []  # Will be auto-computed if empty
RIGHT_ARM_STIFFNESS_OVERRIDE: list[float] = []
RIGHT_ARM_DAMPING_OVERRIDE: list[float] = []  # Will be auto-computed if empty

# Multiplier for damping (1.25 = 25% increase)
ARM_DAMPING_MULTIPLIER = 15


def _build_per_joint_arm_actuators(prefix: str, joint_names: list[str],
                                   stiffness_list: list[float], damping_list: list[float]) -> dict[str, ImplicitActuatorCfg]:
    """Build per-joint actuators for arm joints."""
    actuators: dict[str, ImplicitActuatorCfg] = {}
    for i, jname in enumerate(joint_names):
        key = f"{prefix}_{jname}"
        actuators[key] = ImplicitActuatorCfg(
            joint_names_expr=[jname],
            stiffness=float(stiffness_list[i]),
            damping=float(damping_list[i]),
        )
    return actuators


def _select_arm_gains():
    """Choose either override arrays or baseline with multiplier."""
    def _pick_or_multiply(base, override, multiplier, n, name):
        if len(override) > 0:
            if len(override) != n:
                raise ValueError(f"{name} override must be length {n}, got {len(override)}.")
            return override
        # Apply multiplier to base (for damping with 25% increase)
        return [v * multiplier for v in base]
    
    def _pick(base, override, n, name):
        if len(override) > 0:
            if len(override) != n:
                raise ValueError(f"{name} override must be length {n}, got {len(override)}.")
            return override
        return base
    
    # Stiffness: use override or base (no multiplier)
    left_kp = _pick(LEFT_ARM_STIFFNESS_BASE, LEFT_ARM_STIFFNESS_OVERRIDE, len(LEFT_ARM_JOINTS), "Left arm stiffness")
    right_kp = _pick(RIGHT_ARM_STIFFNESS_BASE, RIGHT_ARM_STIFFNESS_OVERRIDE, len(RIGHT_ARM_JOINTS), "Right arm stiffness")
    
    # Damping: use override, or base * multiplier (25% increase)
    left_kd = _pick_or_multiply(LEFT_ARM_DAMPING_BASE, LEFT_ARM_DAMPING_OVERRIDE, ARM_DAMPING_MULTIPLIER, len(LEFT_ARM_JOINTS), "Left arm damping")
    right_kd = _pick_or_multiply(RIGHT_ARM_DAMPING_BASE, RIGHT_ARM_DAMPING_OVERRIDE, ARM_DAMPING_MULTIPLIER, len(RIGHT_ARM_JOINTS), "Right arm damping")
    
    return left_kp, left_kd, right_kp, right_kd


def _maybe_override_arm_actuators(base_cfg: ArticulationCfg) -> ArticulationCfg:
    """
    If arm base values are provided (non-zero), replace grouped arm actuators with per-joint ones.
    """
    # Check if we have base values to work with
    has_base_values = (
        any(v != 0.0 for v in LEFT_ARM_STIFFNESS_BASE) or
        any(v != 0.0 for v in LEFT_ARM_DAMPING_BASE) or
        any(v != 0.0 for v in RIGHT_ARM_STIFFNESS_BASE) or
        any(v != 0.0 for v in RIGHT_ARM_DAMPING_BASE)
    )
    
    if not has_base_values:
        print("~~~~~~~~~~~~~~~~~~[ArmOverride] No base values set, skipping arm override")
        return base_cfg
    
    left_kp, left_kd, right_kp, right_kd = _select_arm_gains()
    
    print("~~~~~~~~~~~~~~~~~~[ArmOverride] Using LEFT ARM damping:", left_kd)
    print("~~~~~~~~~~~~~~~~~~[ArmOverride] Using RIGHT ARM damping:", right_kd)
    
    # Build per-joint actuators
    arm_actuators: dict[str, ImplicitActuatorCfg] = {}
    arm_actuators.update(_build_per_joint_arm_actuators("LA", LEFT_ARM_JOINTS, left_kp, left_kd))
    arm_actuators.update(_build_per_joint_arm_actuators("RA", RIGHT_ARM_JOINTS, right_kp, right_kd))
    
    # Remove existing arm actuators to avoid conflicts
    new_actuators = dict(base_cfg.actuators)
    new_actuators.pop("left_arm", None)
    new_actuators.pop("right_arm", None)
    
    # Merge per-joint actuators
    new_actuators.update(arm_actuators)
    
    return base_cfg.replace(actuators=new_actuators)



ROBOT_CFG = _maybe_override_arm_actuators(_maybe_override_finger_actuators(KUAVO_V4PRO_CFG))
# ---------------------------
# ARM joint gain tuning code end
# ---------------------------
##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_vel={".*": 0.0},
        ),
        spawn=UsdFileCfg(
            usd_path=ROBOT_CFG.spawn.usd_path,   # or your robot usd path
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
        ),

    )


    ################################################## ROBOTIC ASSETS ##################################################


    cam_egoview = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera/cam_egoview",
        data_types=["rgb"],
        width=640,
        height=480,
        spawn=None,   # use camera from the USD
    )

    cam_leftwristview = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/l_palm/cam_leftwristview",
        data_types=["rgb"],
        width=640,
        height=480,
        spawn=None,
    )

    cam_rightwristview = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/r_palm/cam_rightwristview",
        data_types=["rgb"],
        width=640,
        height=480,
        spawn=None,
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    ################################################## NON-ROBOTIC ASSETS ##################################################
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/table.usd",
            scale=(1.0, 1.0, 1.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )



    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.00492, 0.4432, 0.99078],
        ),
        spawn=UsdFileCfg(
            usd_path="/home/sensethreat/lab_mimic/IsaacLab/source/isaaclab_assets/data/bowl.usd",
            scale=(0.001, 0.001, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        ),
    )

    pouring_cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PouringCup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.19842, 0.45059, 0.98921],
            rot=[0.7071068, 0.7071068, 0.0, 0.0],  # 90 deg around X
        ),
        spawn=UsdFileCfg(
            usd_path="/home/sensethreat/lab_mimic/IsaacLab/source/isaaclab_assets/data/cup_1.usd",
            scale=(0.0007, 0.0007, 0.0007),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )
    pouring_cup_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PouringCup_2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.19994, 0.42043, 0.99124],
            rot=[0.7071068, 0.7071068, 0.0, 0.0],  # 90 deg around X
        ),
        spawn=UsdFileCfg(
            usd_path="/home/sensethreat/lab_mimic/IsaacLab/source/isaaclab_assets/data/cup_2.usd",
            scale=(0.0007, 0.0007, 0.0007),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    liquid_particles: RigidObjectCollectionCfg = _LIQUID_PARTICLES_CFG
    liquid_particles_2: RigidObjectCollectionCfg = _LIQUID_PARTICLES_CFG_2





    factory_nut = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FactoryNut",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.19789, 0.46173, 1.03684], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/factory_m16_nut_green.usd",
            scale=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.2,
                angular_damping=0.4
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        ),
    )


@configclass
class ObjectTableSceneGenerateCfg(ObjectTableSceneCfg):
    """Scene for Cosmos data generation — 720p RGB + depth + seg on all cameras."""

    # Override ego camera: 720p + depth + seg
    cam_egoview = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera/cam_egoview",
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        width=1280,
        height=720,
        spawn=None,
    )
    cam_leftwristview = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/l_palm/cam_leftwristview",
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        width=1280,
        height=720,
        spawn=None,
    )
    cam_rightwristview = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/r_palm/cam_rightwristview",
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        width=1280,
        height=720,
        spawn=None,
    )



##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for Kuavo (arms + fingers)."""
    # kuavo_action: ActionTermCfg = MISSING
    # 14 DOFs: 7 left arm joints + 7 right arm joints
    """Action specifications for Kuavo — single unified 34D term.

    Using ONE JointPositionActionCfg for all 34 DOFs (14 arm + 20 finger) ensures
    that env.action_manager.action is a single contiguous tensor in a known order:
        [0:7]   left arm  (zarm_l1..l7)
        [7:14]  right arm (zarm_r1..r7)
        [14:24] left hand (l_thumbCMC, l_thumbMCP, l_indexMCP, l_indexPIP,
                           l_middleMCP, l_middlePIP, l_ringMCP, l_ringPIP,
                           l_littleMCP, l_littlePIP)
        [24:34] right hand (r_thumbCMC, r_thumbMCP, r_indexMCP, r_indexPIP,
                            r_middleMCP, r_middlePIP, r_ringMCP, r_ringPIP,
                            r_littleMCP, r_littlePIP)

    Previously three separate terms (arms / left_hand / right_hand) caused
    data/actions in the HDF5 to record garbled hand data because Isaac Lab's
    ActionManager concatenates multi-term actions in ways that do not faithfully
    reflect the per-hand split. A single term eliminates this ambiguity entirely.
    """

    kuavo_all: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            # --- Left arm (7 DOF) ---
            *[f"zarm_l{i}_joint" for i in range(1, 8)],
            # --- Right arm (7 DOF) ---
            *[f"zarm_r{i}_joint" for i in range(1, 8)],
            # --- Left hand (10 DOF) ---
            "l_thumbCMC", "l_thumbMCP",
            "l_indexMCP",  "l_indexPIP",
            "l_middleMCP", "l_middlePIP",
            "l_ringMCP",   "l_ringPIP",
            "l_littleMCP", "l_littlePIP",
            # --- Right hand (10 DOF) ---
            "r_thumbCMC", "r_thumbMCP",
            "r_indexMCP",  "r_indexPIP",
            "r_middleMCP", "r_middlePIP",
            "r_ringMCP",   "r_ringPIP",
            "r_littleMCP", "r_littlePIP",
        ],
        preserve_order=True,
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # --- End-effector poses (computed via FK in sim) ---
        left_eef_pos = ObsTerm(
            func=mdp.get_eef_pos,
            params={"link_name": "l_palm"},
        )
        left_eef_quat = ObsTerm(
            func=mdp.get_eef_quat,
            params={"link_name": "l_palm"},
        )
        right_eef_pos = ObsTerm(
            func=mdp.get_eef_pos,
            params={"link_name": "r_palm"},
        )
        right_eef_quat = ObsTerm(
            func=mdp.get_eef_quat,
            params={"link_name": "r_palm"},
        )

        # --- Hand / finger joints (regex for all l_* / r_* finger joints) ---
        hand_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["l_.*", "r_.*"]},
        )

        # --- Cameras: ego + wrist views (RGB) ---
        cam_egoview_rgb = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_egoview"),
                "data_type": "rgb",          # normalized + mean-centered by default
                "normalize": False,
            },
        )
        cam_leftwrist_rgb = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_leftwristview"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        cam_rightwrist_rgb = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("cam_rightwristview"),
                "data_type": "rgb",
                "normalize": False,
            },
        )


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bowl_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("bowl"),
        },
    )

    cup_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("pouring_cup"),
        },
    )
    cup_2_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("pouring_cup_2"),
        },
    )

    cup_tilted_sideways=DoneTerm(
        func=mdp.kuavoV4Pouring_cup_tilted_sideways,
        params={
            "asset_cfg": SceneEntityCfg("pouring_cup"),
            "max_tilt_angle_rad": 1.4,
            "grace_period_steps": 15,
            "max_cup_ang_speed_for_failure": 0.17,
            "lift_height_above_table": 1.08,
        }
    )

    cup_2_tilted_sideways = DoneTerm(
        func=mdp.kuavoV4Pouring_cup_tilted_sideways,
        params={
            "asset_cfg": SceneEntityCfg("pouring_cup_2"),
            "max_tilt_angle_rad": 1.4,
            "grace_period_steps": 15,
            "max_cup_ang_speed_for_failure": 0.17,
            "lift_height_above_table": 1.08,
        },
    )


    particle_spilled = DoneTerm(
        func=mdp.liquid_particle_spilled,
        params={
            "particle_cfg": SceneEntityCfg("liquid_particles_2"),
            "cup_cfg": SceneEntityCfg("pouring_cup_2"),
            "min_spilled_count": 5,
            "cup_xy_radius": 0.05,
            "fell_off_z": 0.50,
            "grace_period_steps": 30,
        },
    )


    particle_spilled_2 = DoneTerm(
        func=mdp.liquid_particle_spilled,
        params={
            "particle_cfg": SceneEntityCfg("liquid_particles_2"),
            "cup_cfg": SceneEntityCfg("pouring_cup_2"),
            "min_spilled_count": 5,
            "cup_xy_radius": 0.05,
            "fell_off_z": 0.50,
            "grace_period_steps": 30,
        },
    )

    # success = DoneTerm(
    #     func=mdp.liquid_particle_pour_success,
    #     params={
    #         "particle_cfg": SceneEntityCfg("liquid_particles_2"),
    #         "bowl_cfg": SceneEntityCfg("bowl"),
    #         "pouring_cup_cfg": SceneEntityCfg("pouring_cup_2"),
    #         "min_in_bowl_count": 10,      # adjust: how many in bowl = success
    #         "bowl_xy_radius": 0.06,
    #         "bowl_z_below_rim": 0.12,
    #         "cup_z_threshold": 1.05,
    #         "particle_vel_threshold": 0.05,
    #     },
    # )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    set_factory_nut_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_nut"),
            "mass_distribution_params": (0.2, 0.2),
            "operation": "abs",
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_object_poses_nut_pour,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
        },
    )


@configclass
class PourKuavoV4ProBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for the KuavoV4Pro environment. 
    Noticed that the actions are left as None here - this base config is meant to be extended by specific env variants (e.g. teleoperation with joint position control) that will define the action space. 
    This allows us to reuse common settings (scene, observations, terminations, events) while customizing the action space as needed for different experiments.

    During Mimic generation ,this environment will be extended with a variant that defines the PINK IK.
    """
    
    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    # In base environment, actions are omitted - subclasses define them
    actions = None
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        """Post initialization - COMMON settings for ALL subclasses."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 1 / 100
        self.sim.render_interval = 2
        
        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        # List of image observations in policy observations
        self.image_obs_list = ["cam_egoview", "cam_rightwristview", "cam_leftwristview"]


@configclass
class PourKuavoV4ProGenerateBaseEnvCfg(PourKuavoV4ProBaseEnvCfg):
    """Base config for Cosmos generation — swaps scene to 720p multi-modal cameras."""
    scene: ObjectTableSceneGenerateCfg = ObjectTableSceneGenerateCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )

    def __post_init__(self):
        super().__post_init__()
        # All 3 cameras active for Cosmos export
        self.image_obs_list = ["cam_egoview", "cam_rightwristview", "cam_leftwristview"]


@configclass
class PourKuavoV4ProTeleopEnvCfg(PourKuavoV4ProBaseEnvCfg):
    """For teleoperation - uses joint position actions."""
    
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()  # Gets decimation, episode_length_s, etc. from base
        
        # Teleoperation-specific settings
        left_arm = [0.17, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
        right_arm = [0.17, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
        hand_idle = [
            -1.5, -1.5,
            -1.5, -3.0,
            -1.5, -3.0,
            -1.5, -3.0,
            -1.5, -3.0,
        ]
        self.idle_action = torch.tensor(
            [
                *left_arm,
                *right_arm,
                *hand_idle,
                *hand_idle,
            ], device=self.sim.device)
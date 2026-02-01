from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

KUAVO_V4PRO_USD_PATH = (
    "/home/sensethreat/lab_mimic/biped_s48/biped_s48/biped_s48.usd"
)


KUAVO_V4PRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KUAVO_V4PRO_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=None,
        articulation_props=None,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos = {
            ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # both arms
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "zarm_l.*",   # all left arm joints
                "zarm_r.*",   # all right arm joints
            ],
            stiffness=None,
            damping=None,
        ),
        # left hand
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=["l_.*"],
            stiffness=None,
            damping=None,
        ),
        # right hand
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=["r_.*"],
            stiffness=None,
            damping=None,
        ),
    },
)

#!/usr/bin/env python3
"""
Spawn a table (from Nucleus) + local USD props (from isaaclab_assets/data)
so you can eyeball pose/scale.

Run:
  ./isaaclab.sh -p scripts/tools/spawn_debug_assets.py
"""

import argparse
import os
from typing import List

# IMPORTANT: AppLauncher is safe to import before SimulationApp starts
from isaaclab.app import AppLauncher


def _resolve_local_usd(assets_data_dir: str, name: str, candidates: List[str]) -> str:
    for rel in candidates:
        p = os.path.join(assets_data_dir, rel)
        if os.path.isfile(p):
            return p
    tried = "\n".join([os.path.join(assets_data_dir, rel) for rel in candidates])
    raise FileNotFoundError(
        f"Could not find {name}. Tried:\n{tried}\n\n"
        f"Tip: if you followed the recommended structure, put them under Props/<Type>/<Name>/... ."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--barrel-scale", type=float, default=1.0)
    parser.add_argument("--yarn-scale", type=float, default=1.0)
    args = parser.parse_args()

    # 1) Start SimulationApp FIRST (this makes pxr / USD available)
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    # 2) Now it is safe to import Isaac/Omni/pxr-dependent modules
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

    @configclass
    class DebugSpawnSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=GroundPlaneCfg(),
        )

        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )

        table = AssetBaseCfg(
            prim_path="/World/Table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[0.0, 0.55, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/table.usd",
                scale=(1.0, 1.0, 1.3),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ),
        )

        barrel_cup = RigidObjectCfg(
            prim_path="/World/BarrelCup",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.10, 0.55, 1.05],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path="",  # injected at runtime
                scale=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
            ),
        )

        bowl = RigidObjectCfg(
            prim_path="/World/YarnBowl",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[-0.10, 0.55, 1.05],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path="",  # injected at runtime
                scale=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
            ),
        )

    # Resolve local USDs (you said they’re directly under isaaclab_assets/data)
    barrel_usd = _resolve_local_usd(
        ISAACLAB_ASSETS_DATA_DIR,
        "barrel_cup.usd",
        candidates=[
            "barrel_cup.usd",
            "Props/Containers/BarrelCup/barrel_cup.usd",
            "Props/Containers/barrel_cup.usd",
            "Props/barrel_cup.usd",
        ],
    )
    yarn_usd = _resolve_local_usd(
        ISAACLAB_ASSETS_DATA_DIR,
        "bowl.usd",
        candidates=[
            "bowl.usd",
            "Props/Containers/YarnBowl/bowl.usd",
            "Props/Containers/bowl.usd",
            "Props/bowl.usd",
        ],
    )

    print(f"[INFO] ISAACLAB_ASSETS_DATA_DIR = {ISAACLAB_ASSETS_DATA_DIR}")
    print(f"[INFO] barrel_cup USD = {barrel_usd}")
    print(f"[INFO] bowl USD  = {yarn_usd}")

    # Simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=args.dt, device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.6, 1.6, 1.3], target=[0.0, 0.55, 0.9])

    scene_cfg = DebugSpawnSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    scene_cfg.barrel_cup.spawn.usd_path = barrel_usd
    scene_cfg.bowl.spawn.usd_path = yarn_usd
    scene_cfg.barrel_cup.spawn.scale = (args.barrel_scale,) * 3
    scene_cfg.bowl.spawn.scale = (args.yarn_scale,) * 3

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    print("[INFO] Scene spawned. Close the window or Ctrl+C to exit.")
    try:
        while simulation_app.is_running():
            sim.step()
            scene.update(sim.get_physics_dt())
            stage = sim.stage
            # print(stage.GetPrimAtPath("/World/Table").IsValid())
            # print(stage.GetPrimAtPath("/World/YarnBowl").IsValid())
            # print(stage.GetPrimAtPath("/World/BarrelCup").IsValid())

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

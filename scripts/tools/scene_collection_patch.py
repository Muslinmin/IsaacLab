"""
Patch for InteractiveScene.get_state() and reset_to() to include RigidObjectCollection.

IsaacLab's InteractiveScene does NOT save/restore RigidObjectCollection state in
get_state() / reset_to(). This means particles (or any entity using RigidObjectCollectionCfg)
are silently dropped during demo recording and replay.

HOW TO APPLY:
    Option A (monkey-patch): Call patch_scene_for_collections(env.scene) once after env creation.
    Option B (manual):       Copy the two methods below into your InteractiveScene subclass.

This patch adds a "rigid_object_collection" key to the state dict, storing per-object
poses and velocities using the write_object_link_pose_to_sim / write_object_link_velocity_to_sim API.
"""

import torch
from collections.abc import Sequence


def get_state_with_collections(self, is_relative: bool = False):
    """Patched get_state that also captures RigidObjectCollection states."""
    # Call the original to get articulation + rigid_object + deformable_object
    state = self._original_get_state(is_relative=is_relative)

    # ---- ADD: rigid_object_collections ----
    state["rigid_object_collection"] = dict()
    for asset_name, collection in self._rigid_object_collections.items():
        asset_state = dict()
        # object_pos_w: (num_envs, num_objects, 3)
        # object_quat_w: (num_envs, num_objects, 4)
        # object_vel_w: (num_envs, num_objects, 6)
        asset_state["object_pose"] = torch.cat([
            collection.data.object_pos_w.clone(),
            collection.data.object_quat_w.clone(),
        ], dim=-1)  # (num_envs, num_objects, 7)

        if is_relative:
            # Subtract env origins from position (first 3 of the 7)
            asset_state["object_pose"][:, :, :3] -= self.env_origins.unsqueeze(1)

        asset_state["object_velocity"] = collection.data.object_vel_w.clone()  # (num_envs, num_objects, 6)
        state["rigid_object_collection"][asset_name] = asset_state

    return state


def reset_to_with_collections(
    self,
    state: dict,
    env_ids: Sequence[int] | None = None,
    is_relative: bool = False,
):
    """Patched reset_to that also restores RigidObjectCollection states."""
    # Call the original to restore articulation + rigid_object + deformable_object
    self._original_reset_to(state, env_ids=env_ids, is_relative=is_relative)

    if env_ids is None:
        env_ids = slice(None)

    # ---- ADD: rigid_object_collections ----
    if "rigid_object_collection" in state:
        for asset_name, collection in self._rigid_object_collections.items():
            if asset_name not in state["rigid_object_collection"]:
                continue
            asset_state = state["rigid_object_collection"][asset_name]

            object_pose = asset_state["object_pose"].clone()  # (N, num_objects, 7)
            if is_relative:
                if isinstance(env_ids, slice):
                    origins = self.env_origins[env_ids]
                else:
                    origins = self.env_origins[env_ids]
                object_pose[:, :, :3] += origins.unsqueeze(1)

            object_velocity = asset_state["object_velocity"].clone()  # (N, num_objects, 6)

            # Write each object's pose and velocity
            collection.write_object_link_pose_to_sim(object_pose, env_ids=env_ids)
            collection.write_object_link_velocity_to_sim(object_velocity, env_ids=env_ids)

        # Need to write data to sim again after our additions
        for collection in self._rigid_object_collections.values():
            collection.write_data_to_sim()


def patch_scene_for_collections(scene):
    """
    Monkey-patch an InteractiveScene instance to support RigidObjectCollection
    in get_state() and reset_to().

    Usage:
        env = MyEnv(cfg)
        patch_scene_for_collections(env.scene)
    """
    import types

    # Save originals
    scene._original_get_state = scene.get_state
    scene._original_reset_to = scene.reset_to

    # Replace with patched versions
    scene.get_state = types.MethodType(get_state_with_collections, scene)
    scene.reset_to = types.MethodType(reset_to_with_collections, scene)

    print("[PATCH] InteractiveScene patched to save/restore RigidObjectCollection state")
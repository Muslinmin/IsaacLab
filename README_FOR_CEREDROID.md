# IsaacLab — Bimanual Pouring Synthetic Data Generation Pipeline

> Modified fork of [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) for the RSE4018 capstone project:  
> **"Scaling VLA Fine-Tuning with Synthetic Data Generation"**

This repository extends Isaac Lab with a complete pipeline for generating synthetic bimanual pouring demonstrations on the **KuavoV4Pro 34-DOF humanoid robot**, using **MimicGen** for data augmentation and **LeRobot v2.1** as the output format for downstream VLA fine-tuning (NVIDIA GR00T N1.6-3B).

---

## Architecture Overview

The pipeline has three stages: teleoperation-based demonstration collection, MimicGen synthetic data generation, and data conversion for VLA training.

### 1. Teleoperation & Isaac Lab Environment

```
Operator (Meta Quest 3)
  │  WiFi / USB
  ▼
ROS1 Docker (Noetic)
  Drake IK solver: EEF targets → 34D joint commands
  ZMQ PUB :5555 (34D joint cmds)    ZMQ SUB :5556 (48D joint state)
  │                                          ▲
  ▼                                          │
Isaac Lab (conda env)
  KuavoRosJointDevice
    ZMQ SUB cmd :5555
    ZMQ PUB state :5556, img :5557
    advance() → torch.Tensor [1, 34]
  │
  ▼
Base Environment Config
  ├── KuavoV4Pro ArticulationCfg   48 DOF, per-joint actuators
  ├── ObjectTableSceneCfg          Robot, table, cups, particles
  ├── ObservationsCfg              Joint pos, EEF, 3× RGB
  ├── TerminationsCfg              Timeout, drop, spill, success
  ├── ActionsCfg                   34D: 14 arm + 20 finger
  ├── EventCfg                     Reset scene, liquid poses
  ├── Domain Randomisation         3 object variants × 3 table materials × 3 light presets
  └── Cameras                      ego 640×480 + 2 wrist (D405)
```

### 2. Pink IK Layer & MimicGen Generation

```
Pink IK Layer
  ├── PinkInverseKinematicsActionCfg   14 arm joints → Pink IK, 20 finger → passthrough
  └── PinkIKControllerCfg             Pinocchio URDF (l_palm + r_palm)
                                       pos 8.0, orient 1.0, damp 0.5
                                       NullSpacePostureTask cost=0.2
                                       DampingTask cost=0.5

MimicGen Config (PouringKuavoV4ProMimic_GenCfg)
  ├── generation_guarantee = True
  ├── generation_joint_pos = True
  ├── num_trials = 100, max_fail = 25
  └── Subtask definitions
        object_ref per subtask
        subtask_term_signal for annotation
        NN selection (k=3), action_noise = 0.003
        interpolation steps, idle subtask boundary

MimicGen Env (PouringKuavoV4ProMimicEnv)
  ├── action_to_target_eef         Joint pos vs Pink IK
  ├── target_eef_to_action         EEF → Pink IK 34D
  ├── actions_to_gripper           Extract 20D → L/R 10D
  ├── step()                       finger permutation fix
  └── reset()                      4th joints → 90°

Pipeline Scripts
  ├── record_demos.py              Teleoperation recording → raw HDF5
  ├── annotate_demos.py            Subtask boundary labels → annotated HDF5
  └── generate_dataset.py          Synthetic generation → output.hdf5
```

### 3. Data Conversion (HDF5 → LeRobot)

```
output_dataset.hdf5
  Per episode: RGB frames + 34D joint actions + joint states
             + EEF positions/quaternions + hand joint states
  Auto-detected format: teleoperation vs MimicGen
  │
  ▼
hdf5_to_lerobot.py
  ├── Joint layout mapping         34D sim → 44D real layout
  ├── Hand joint reduction         10D sim → 6D real (MCP only)
  ├── Articulation indexing        48D tensor → arm + hand indices
  └── Format auto-detect           Teleop vs MimicGen source
  │
  ▼
LeRobot v2.1 dataset
  Parquet (actions, states) + MP4 (camera views)
  → HuggingFace upload
```

---

## Repository Structure (Capstone Additions)

Files and directories added or modified relative to upstream `isaac-sim/IsaacLab`:

```
IsaacLab/
├── source/
│   ├── isaaclab/isaaclab/
│   │   ├── controllers/pink_ik/          # Pink IK controller (Pinocchio-based)
│   │   ├── devices/metaquestkuavoXR/     # Meta Quest 3 teleoperation device
│   │   └── envs/
│   │       ├── manager_based_rl_mimic_env.py
│   │       └── mimic_env_cfg.py
│   ├── isaaclab_assets/isaaclab_assets/
│   │   └── robots/kuavoV4Pro.py          # KuavoV4Pro articulation config
│   ├── isaaclab_mimic/isaaclab_mimic/
│   │   ├── datagen/                      # MimicGen data generation core
│   │   └── envs/pinocchio_envs/
│   │       ├── pouring_KuavoV4Pro_mimic_env.py      # Mimic env with IK fixes
│   │       └── pouring_KuavoV4Pro_mimic_env_cfg.py   # Gen/Annotate/Cosmos configs
│   └── isaaclab_tasks/.../manipulation/pick_place/
│       ├── kuavoV4ProPouring_env_cfg.py              # Base pouring env + domain rand
│       ├── kuavoV4ProPouring_pink_ik_env_cfg.py      # Pink IK env config
│       └── mdp/
│           ├── pick_place_events.py                  # Scene reset events
│           └── terminations.py                       # Liquid particle pour success
│
├── scripts/
│   ├── imitation_learning/isaaclab_mimic/
│   │   ├── annotate_demos.py             # Subtask boundary annotation
│   │   ├── generate_dataset.py           # MimicGen synthetic generation
│   │   └── scene_collection_patch.py     # RigidObjectCollection state patch
│   └── tools/
│       ├── record_demos.py               # Teleoperation demo recording
│       ├── hdf5_to_mp4.py                # Video extraction from HDF5
│       └── merge_hdf5_datasets.py        # Dataset merging utility
│
├── upload_sd.py                          # Upload datasets to HuggingFace
├── download_dataset.py                   # Download datasets from HuggingFace
├── downloadDatasetSpecific.py            # Download specific dataset splits
│
├── # Cloud infrastructure (RunPod)
├── setup_all.sh                          # Master env bootstrap
├── setup_runpod_vnc.sh                   # VNC server installation
├── setup_and_start_display.sh            # Display + VNC orchestrator
├── setup_isaac_env.sh                    # Isaac Sim env vars
├── activate_conda.sh                     # Conda activation helper
├── init_conda.sh                         # Conda initialisation
├── start_vnc.sh                          # VNC server launcher
├── start_ssh.sh                          # SSH server launcher
│
└── test_*_report.txt                     # MimicGen generation test logs (×5)
```

---

## Prerequisites

- **NVIDIA Isaac Sim 4.5 / 5.0 / 5.1**
- **Python 3.11** (conda environment)
- **Pinocchio** (for Pink IK controller)
- **ROS1 Noetic** (Docker, for teleoperation only)
- **Meta Quest 3** (for teleoperation only)
- GPU with ≥24 GB VRAM recommended for generation

---

## Setup

### Local Installation

Follow the standard [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), then:

```bash
git clone https://github.com/Muslinmin/IsaacLab.git
cd IsaacLab
git checkout capstone

# Install Isaac Lab extensions
./isaaclab.sh --install
pip install pin pink  # Pinocchio + Pink IK
```

### Cloud Installation (RunPod)

```bash
cd /workspace
git clone https://github.com/Muslinmin/IsaacLab.git
cd IsaacLab && git checkout capstone

# One-command setup
bash setup_all.sh

# If VNC display is needed for headed generation
bash setup_and_start_display.sh
```

---

## Usage

### 1. Record Demonstrations (Teleoperation)

Requires Meta Quest 3 + ROS1 Docker running the Drake IK solver.

```bash
python scripts/tools/record_demos.py \
    --task Isaac-KuavoV4Pro-Pouring-PinkIK-Teleop-v0 \
    --num_demos 10 \
    --output_dir datasets/real_demos
```

### 2. Annotate Subtask Boundaries

```bash
python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-KuavoV4Pro-Pouring-PinkIK-Annotate-v0 \
    --input datasets/real_demos/demo.hdf5
```

### 3. Generate Synthetic Data (MimicGen)

```bash
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-KuavoV4Pro-Pouring-PinkIK-Generate-v0 \
    --input datasets/real_demos/demo_annotated.hdf5 \
    --output datasets/synthetic/output.hdf5 \
    --num_envs 4
```

### 4. Convert to LeRobot Format & Upload

```bash
# Convert HDF5 → LeRobot v2.1
python hdf5_to_lerobot.py \
    --input datasets/synthetic/output.hdf5 \
    --output datasets/lerobot_output

# Upload to HuggingFace
python upload_sd.py --repo Lusmse/syn_realDataset
```

---

## Key Engineering Contributions

- **Pink IK Controller**: Pinocchio-based inverse kinematics integrated into Isaac Lab's action manager for 14-DOF bimanual arm control with null-space posture regularisation.
- **RigidObjectCollection State Patch**: Monkey-patch for `InteractiveScene.get_state()`/`reset_to()` to serialise rigid-body particle states, enabling liquid simulation reset during MimicGen generation.
- **Liquid Particle Pour Success Metric**: Custom termination function using per-particle position/velocity thresholds relative to bowl geometry for automated success detection.
- **Visual Domain Randomisation**: Three object mesh variants × three table materials × three lighting presets, selectable via `split_id` for controlled ablation.
- **Left/Right Arm Subtask Fix**: Corrected arm-to-object reference mapping in MimicGen subtask configs (swapped `pouring_cup` ↔ `pouring_cup_2` assignment).

---

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Upstream isaac-sim/IsaacLab (synced) |
| `capstone` | Stable capstone project branch |
| `cloud_progress` | Cloud development snapshots (merged into capstone) |

---

## Related Repositories

- **[Lusmse/syn_realDataset](https://huggingface.co/datasets/Lusmse/syn_realDataset)** — Real and synthetic LeRobot datasets
- **[Lusmse/capstone-vla-checkpoints](https://huggingface.co/Lusmse/capstone-vla-checkpoints)** — GR00T N1.6-3B fine-tuned checkpoints

---

## License

This fork inherits the [BSD-3-Clause](LICENSE) license from Isaac Lab. The `isaaclab_mimic` extension is licensed under [Apache 2.0](LICENSE-mimic). Isaac Sim itself is under [proprietary NVIDIA licensing](docs/licenses/dependencies/isaacsim-license.txt).

---

## Acknowledgements

- **Isaac Lab** — [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) (Mittal et al., 2025)
- **MimicGen** — [NVlabs/mimicgen](https://github.com/NVlabs/mimicgen) (Mandlekar et al., 2023)
- **Pink IK** — [stephane-caron/pink](https://github.com/stephane-caron/pink) (Pinocchio-based)

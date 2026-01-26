# kuavo_ros_joint_device.py
# Extended version with camera streaming support

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import zmq
import torch

from isaaclab.devices import DeviceBase
from isaaclab.utils import configclass

from typing import Callable, Dict, List


@configclass
@dataclass
class KuavoRosJointDeviceCfg:
    """Config for KuavoRosJointDevice."""
    device: str = "cpu"
    # ZMQ endpoint where ROS publishes joint commands (Isaac SUBscribes here)
    cmd_sub_address: str = "tcp://192.168.8.103:5555"  # <- ROS machine IP:5555

    # ZMQ endpoint where Isaac publishes joint states (Isaac PUBlishes here)
    state_pub_bind: str = "tcp://*:5556"               # <- ROS connects/subscribes to this
    
    # NEW: ZMQ endpoint where Isaac publishes camera images
    image_pub_bind: str = "tcp://*:5557"               # <- ROS connects/subscribes to this

    # Number of DOFs expected from ROS (arms + fingers)
    # 14 arm joints (7 left, 7 right) + 20 joints (left hand, right hand)
    num_dofs: int = 34

    # How long advance() waits for first command before giving zeros (sec)
    init_timeout_sec: float = 2.0

    # Whether to print basic debug info
    verbose: bool = True
    
    # Image compression quality (1-100, higher = better quality)
    jpeg_quality: int = 85


class KuavoRosJointDevice(DeviceBase):
    """Device that receives Kuavo joint commands from ROS via ZeroMQ.

    - .advance() returns latest joint command as a torch tensor [num_envs, num_dofs]
    - .send_joint_state() publishes sim joint state back to ROS
    - .send_camera_frame() publishes camera images to ROS (NEW)
    """

    def __init__(self, cfg: KuavoRosJointDeviceCfg):
        super().__init__(cfg)
        self.device = cfg.device
        self.cfg: KuavoRosJointDeviceCfg = cfg
        self._num_dofs = cfg.num_dofs

        # ZMQ context and sockets
        self._context = zmq.Context.instance()

        # Subscribe to ROS joint commands
        self._cmd_sub = self._context.socket(zmq.SUB)
        self._cmd_sub.connect(cfg.cmd_sub_address)
        self._cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # Publish joint states back to ROS
        self._state_pub = self._context.socket(zmq.PUB)
        self._state_pub.bind(cfg.state_pub_bind)
        
        # NEW: Publish camera images to ROS
        self._img_pub = self._context.socket(zmq.PUB)
        self._img_pub.bind(cfg.image_pub_bind)
        self._img_pub.setsockopt(zmq.SNDHWM, 1)  # Keep only 1 frame in buffer (low latency)

        # Latest received action (numpy array shape [num_dofs])
        self._latest_action: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = True

        # Background thread to receive commands
        self._recv_thread = threading.Thread(
            target=self._recv_cmd_loop, daemon=True
        )
        self._recv_thread.start()

        if self.cfg.verbose:
            print(
                f"[KuavoRosJointDevice] Listening for commands on {cfg.cmd_sub_address}, "
                f"publishing joint states on {cfg.state_pub_bind}, "
                f"publishing images on {cfg.image_pub_bind}, num_dofs={cfg.num_dofs}"
            )

        # Optionally wait for first message, so advance() isn't all zeros forever
        self._wait_for_first_command()

    # ------------------------------------------------------------------
    # Background receive loop (existing)

    # == TELEOP INDEX → (articulation_index, joint_name) ===
    # 0:  2  zarm_l1_joint
    # 1:  7  zarm_l2_joint
    # 2: 12  zarm_l3_joint
    # 3: 16  zarm_l4_joint
    # 4: 20  zarm_l5_joint
    # 5: 24  zarm_l6_joint
    # 6: 26  zarm_l7_joint
    # 7:  3  zarm_r1_joint
    # 8:  8  zarm_r2_joint
    # 9: 13  zarm_r3_joint
    # 10: 17  zarm_r4_joint
    # 11: 21  zarm_r5_joint
    # 12: 25  zarm_r6_joint
    # 13: 27  zarm_r7_joint
    # 14: 32  l_thumbCMC
    # 15: 42  l_thumbMCP
    # 16: 28  l_indexMCP
    # 17: 38  l_indexPIP
    # 18: 30  l_middleMCP
    # 19: 40  l_middlePIP
    # 20: 31  l_ringMCP
    # 21: 41  l_ringPIP
    # 22: 29  l_littleMCP
    # 23: 39  l_littlePIP
    # 24: 37  r_thumbCMC
    # 25: 47  r_thumbMCP
    # 26: 33  r_indexMCP
    # 27: 43  r_indexPIP
    # 28: 35  r_middleMCP
    # 29: 45  r_middlePIP
    # 30: 36  r_ringMCP
    # 31: 46  r_ringPIP
    # 32: 34  r_littleMCP
    # 33: 44  r_littlePIP
    # Recording initial status: False



    # ------------------------------------------------------------------
    def _recv_cmd_loop(self):
        """Blocking loop to receive joint command vectors from ROS."""
        while self._running:
            try:
                msg = self._cmd_sub.recv(flags=0)  # blocking

                # Assume ROS sends a simple space-separated float string, e.g. "0.0 1.0 ..."
                arr = np.fromstring(msg.decode("utf-8"), sep=" ")

                if arr.shape[0] != self._num_dofs:
                    if self.cfg.verbose:
                        print(
                            f"[KuavoRosJointDevice] Warning: expected {self._num_dofs} DOFs, "
                            f"got {arr.shape[0]}"
                        )
                    continue
                # ---- DEBUG: print thumb & ring commands (rate-limited) ----
                if self.cfg.verbose:
                    now = time.time()
                    # create a timer attribute the first time
                    if not hasattr(self, "_dbg_last_print"):
                        self._dbg_last_print = 0.0

                    # print at most 10x per second
                    if now - self._dbg_last_print > 0.1:
                        self._dbg_last_print = now

                        # indices based on your TELEOP INDEX mapping
                        l_thumb = arr[14:16]   # [l_thumbCMC, l_thumbMCP]
                        l_ring  = arr[20:22]   # [l_ringMCP,  l_ringPIP]
                        r_thumb = arr[24:26]   # [r_thumbCMC, r_thumbMCP]
                        r_ring  = arr[30:32]   # [r_ringMCP,  r_ringPIP]

                        # print(
                        #     "[KuavoRosJointDevice][DBG] "
                        #     f"L_thumb(CMC,MCP)={l_thumb}  "
                        #     f"L_ring(MCP,PIP)={l_ring}  "
                        #     f"R_thumb(CMC,MCP)={r_thumb}  "
                        #     f"R_ring(MCP,PIP)={r_ring}"
                        # )
                # -----------------------------------------------------------
                with self._lock:
                    # keep the internal action vector in radians (what Isaac expects)
                    self._latest_action = arr

            except zmq.ZMQError:
                if not self._running:
                    break
                time.sleep(0.001)

    def _wait_for_first_command(self):
        """Wait up to init_timeout_sec for first command, else use default pose."""
        t0 = time.time()
        while self._latest_action is None and (time.time() - t0) < self.cfg.init_timeout_sec:
            time.sleep(0.01)

        if self._latest_action is None:
            if self.cfg.verbose:
                print(
                    f"[KuavoRosJointDevice] No command received within {self.cfg.init_timeout_sec}s. "
                    "Using DEFAULT joint values until something arrives."
                )

            with self._lock:
                default = np.zeros(self._num_dofs, dtype=np.float32)
                default[0] = 0.17      # first joint
                default[3] = -2.1      # fourth joint
                default[7] = 0.17
                default[10] = -2.1
                self._latest_action = default

    # ------------------------------------------------------------------
    # Public API used by teleop interface
    # ------------------------------------------------------------------
    def advance(self, dt: float = 0.0) -> torch.Tensor:
        """Return latest joint command as a torch tensor [num_envs, num_dofs].

        For teleop, num_envs is typically 1.
        """
        with self._lock:
            if self._latest_action is None:
                action_np = np.zeros(self._num_dofs, dtype=np.float32)
                action_np[0] = 0.17
                action_np[3] = -2.1
                action_np[7] = 0.17
                action_np[10] = -2.1
            else:
                action_np = self._latest_action.astype(np.float32, copy=False)

        # Shape: [1, num_dofs] for a single environment
        action = torch.from_numpy(action_np).unsqueeze(0)
        return action.to(self.device)  # DeviceBase has self.device

    # ------------------------------------------------------------------
    # Publishing sim joint state back to ROS (existing)
    # ------------------------------------------------------------------
    def send_joint_state(self, joint_pos: np.ndarray):
        """Publish current sim joint positions back to ROS via ZMQ.

        joint_pos: shape [num_dofs] numpy array in the same ordering ROS expects.
        """
        # Serialize as space-separated floats
        msg = " ".join(f"{x:.6f}" for x in joint_pos)
        self._state_pub.send_string(msg)
        # print(f"Sending joint states {msg}")

    # ------------------------------------------------------------------
    # NEW: Publishing camera frames to ROS
    # ------------------------------------------------------------------
    def send_camera_frame(self, camera_id: str, image_data: np.ndarray):
        """Publish camera frame to ROS via ZMQ.
        
        Args:
            camera_id: Camera identifier (e.g., "ego", "left_wrist", "right_wrist")
            image_data: numpy array [H, W, 3] in RGB format (uint8, range 0-255)
        """
        try:
            import cv2
            import pickle
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
            
            # Encode as JPEG for compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality]
            success, encoded = cv2.imencode('.jpg', image_bgr, encode_param)
            
            if not success:
                print(f"[KuavoRosJointDevice] Failed to encode image for {camera_id}")
                return
            
            # Create message: [camera_id, width, height, encoding, data]
            msg = {
                'camera_id': camera_id,
                'width': image_data.shape[1],
                'height': image_data.shape[0],
                'encoding': 'jpeg',
                'data': encoded.tobytes()
            }
            
            # Send via ZMQ (non-blocking to avoid slowing down sim)
            # if self.cfg.verbose:
            #     print(f"Sending imagges...")
            self._img_pub.send(pickle.dumps(msg), flags=zmq.NOBLOCK)
            
        except zmq.Again:
            # Buffer full, skip this frame (prevents blocking)
            if self.cfg.verbose:
                print(f"[KuavoRosJointDevice] Skipped frame for {camera_id} (buffer full)")
        except Exception as e:
            print(f"[KuavoRosJointDevice] Error sending camera frame: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        self._running = False
        try:
            self._cmd_sub.close(0)
            self._state_pub.close(0)
            self._img_pub.close(0)
        except Exception:
            pass
        if self.cfg.verbose:
            print("[KuavoRosJointDevice] Closed sockets.")

    def add_callback(self, name: str, callback: Callable):
        """Kuavo ROS device does not use callbacks."""
        # Do nothing on purpose
        return

    def reset(self):
        """Reset device internal state on env reset."""
        # Typically we clear the last action
        with self._lock:
            self._latest_action = None
        # Not strictly required:
        # self._wait_for_first_command()
        return
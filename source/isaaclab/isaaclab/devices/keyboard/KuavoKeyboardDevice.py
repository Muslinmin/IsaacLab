import carb
import omni.appwindow
import weakref
from dataclasses import dataclass, field
from typing import Callable, Dict, List
from isaaclab.devices import DeviceBase
from isaaclab.utils import configclass


@configclass
@dataclass
class KuavoKeyboardDeviceCfg:
    """Event-based keyboard teleop config."""
    start_key: str = "W"
    stop_key: str = "S"
    reset_keys: List[str] = field(default_factory=lambda: ["R", "T"])
    verbose: bool = True


class KuavoKeyboardDevice:
    """Event-driven keyboard device like IsaacLab's se2_keyboard.

    Fires events on KEY_PRESS:
       W → START
       S → STOP
       R/T → RESET
    """

    def __init__(self, cfg: KuavoKeyboardDeviceCfg):
        # super().__init__(cfg)
        self.cfg = cfg

        # Event registry
        self._callbacks: Dict[str, List[Callable[[], None]]] = {}

        # Omniverse input interface
        self._input = carb.input.acquire_input_interface()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = self._appwindow.get_keyboard()

        # Subscribe to event stream
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        if self.cfg.verbose:
            print("[KuavoKeyboardDevice] Event-based keyboard active (W=START, S=STOP, R/T=RESET)")

    # --------------------
    # Event wiring
    # --------------------
    def add_callback(self, event_name: str, fn: Callable[[], None]):
        self._callbacks.setdefault(event_name, []).append(fn)

    def _fire_event(self, event_name: str):
        if event_name in self._callbacks:
            for fn in self._callbacks[event_name]:
                fn()

    # --------------------
    # Keyboard event handler
    # --------------------
    def _on_keyboard_event(self, event, *args):
        # Only react on key press
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        key_name = event.input.name  # e.g. "W", "A", "R"

        # START
        if key_name == self.cfg.start_key:
            if self.cfg.verbose:
                print("[KuavoKeyboardDevice] START pressed")
            self._fire_event("START")

        # STOP
        elif key_name == self.cfg.stop_key:
            if self.cfg.verbose:
                print("[KuavoKeyboardDevice] STOP pressed")
            self._fire_event("STOP")

        # RESET (R/T)
        elif key_name in self.cfg.reset_keys:
            if self.cfg.verbose:
                print("[KuavoKeyboardDevice] RESET pressed")
            self._fire_event("RESET")

        return True

    # --------------------
    # Advance (not used)
    # --------------------
    def advance(self, dt: float = 0.0):
        # No polling needed — event-based
        return None

    def close(self):
        if self.cfg.verbose:
            print("[KuavoKeyboardDevice] Closed.")
    def reset(self):
        self._prev_state.clear()
        if self.cfg.verbose:
            print("[KuavoKeyboardDevice] Reset.")

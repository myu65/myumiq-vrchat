"""Read raw device poses without becoming a Scene client or changing settings."""

import ctypes
import os
import subprocess
import time

from ..body import BodyState, Pose, PoseSignal, compose, inverse, q_from_matrix
from .osc import LiveConfig, from_openvr


def require_console() -> None:
    if os.name != "nt":
        raise RuntimeError("live validation currently supports Windows console sessions only")
    session = ctypes.c_uint32()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel.ProcessIdToSessionId(os.getpid(), ctypes.byref(session)):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.WTSGetActiveConsoleSessionId.restype = ctypes.c_uint32
    if session.value == 0 or session.value != kernel.WTSGetActiveConsoleSessionId():
        raise RuntimeError(
            "live output must run in the console session; RDP is not a rendering gate"
        )


class OpenVRReadback:
    def __init__(self, config: LiveConfig, hmd_serial: str):
        self.config, self.hmd_serial = config, hmd_serial
        self.vr = self.system = None

    def start(self):
        require_console()
        probe = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "[bool](Get-Process vrserver -ErrorAction SilentlyContinue)",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if probe.stdout.strip() != "True":
            raise RuntimeError("start the approved SteamVR/HMD configuration before live output")
        import openvr

        self.vr = openvr
        self.system = openvr.init(openvr.VRApplication_Background)
        serial = self.system.getStringTrackedDeviceProperty(0, openvr.Prop_SerialNumber_String)
        if serial != self.hmd_serial:
            self.close()
            raise RuntimeError("active HMD serial does not match the explicit local configuration")
        return self

    def _device_inventory(self):
        vr, system = self.vr, self.system
        inventory = {}
        for index in range(vr.k_unMaxTrackedDeviceCount):
            kind = system.getTrackedDeviceClass(index)
            if kind in (vr.TrackedDeviceClass_Controller, vr.TrackedDeviceClass_GenericTracker):
                serial = system.getStringTrackedDeviceProperty(index, vr.Prop_SerialNumber_String)
                inventory.setdefault((kind, serial), []).append(index)
        return inventory

    def _controller_index(self, role, serial, inventory=None):
        vr, system = self.vr, self.system
        assigned = system.getTrackedDeviceIndexForControllerRole(role)
        if assigned != vr.k_unTrackedDeviceIndexInvalid:
            if (
                system.getStringTrackedDeviceProperty(assigned, vr.Prop_SerialNumber_String)
                != serial
            ):
                raise RuntimeError(
                    "tracked role is occupied by a device outside this session's ownership"
                )
        # The deprecated role lookup can be unavailable while the owned device
        # still tracks. Read device identity directly; this does not establish
        # an application's action bindings or avatar IK.
        if inventory is None:
            inventory = self._device_inventory()
        matches = inventory.get((vr.TrackedDeviceClass_Controller, serial), [])
        if len(matches) > 1:
            raise RuntimeError("duplicate controller identity")
        if not matches:
            return vr.k_unTrackedDeviceIndexInvalid
        index = matches[0]
        hint = system.getInt32TrackedDeviceProperty(index, vr.Prop_ControllerRoleHint_Int32)
        if hint != role:
            raise RuntimeError("owned controller has an unexpected left/right role hint")
        return index

    def observe(self) -> BodyState:
        vr, system = self.vr, self.system
        poses = system.getDeviceToAbsoluteTrackingPose(vr.TrackingUniverseRawAndUncalibrated, 0, 64)
        now = time.perf_counter()
        inventory = self._device_inventory()

        def signal(index, expected_serial, frame):
            unavailable = PoseSignal(timestamp=now)
            if index == vr.k_unTrackedDeviceIndexInvalid:
                return unavailable
            serial = system.getStringTrackedDeviceProperty(index, vr.Prop_SerialNumber_String)
            if serial != expected_serial:
                raise RuntimeError(
                    "tracked role is occupied by a device outside this session's ownership"
                )
            p = poses[index]
            valid = bool(p.bDeviceIsConnected and p.bPoseIsValid and p.eTrackingResult == 200)
            pose = None
            if valid:
                matrix = p.mDeviceToAbsoluteTracking.m
                raw = Pose(
                    position=tuple(float(matrix[i][3]) for i in range(3)),
                    orientation=q_from_matrix(matrix),
                )
                pose = from_openvr(compose(inverse(frame), raw))
            return PoseSignal(
                pose=pose,
                valid=valid,
                connected=bool(p.bDeviceIsConnected),
                tracking_result=int(p.eTrackingResult),
                confidence=1.0 if valid else 0.0,
                source="openvr_raw",
                timestamp=now,
            )

        extra = {}
        for binding in self.config.trackers:
            expected_serial = f"VMT_{binding.index}"
            indices = inventory.get((vr.TrackedDeviceClass_GenericTracker, expected_serial), [])
            if len(indices) > 1:
                raise RuntimeError("duplicate owned tracker identity")
            observed = signal(
                indices[0] if indices else vr.k_unTrackedDeviceIndexInvalid,
                expected_serial,
                self.config.vmt_from_stage,
            )
            if observed.pose is not None:
                observed = observed.model_copy(
                    update={
                        "pose": compose(observed.pose, inverse(binding.body_from_tracker)),
                    }
                )
            extra[binding.part] = observed
        return BodyState(
            head=signal(0, self.hmd_serial, self.config.hmd_from_stage),
            left=signal(
                self._controller_index(
                    vr.TrackedControllerRole_LeftHand, f"VMT_{self.config.left_index}", inventory
                ),
                f"VMT_{self.config.left_index}",
                self.config.vmt_from_stage,
            ),
            right=signal(
                self._controller_index(
                    vr.TrackedControllerRole_RightHand, f"VMT_{self.config.right_index}", inventory
                ),
                f"VMT_{self.config.right_index}",
                self.config.vmt_from_stage,
            ),
            **extra,
        )

    def close(self):
        if self.system is not None:
            self.vr.shutdown()
            self.system = None

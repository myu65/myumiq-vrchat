"""Low-rate desktop vision for VRChat; motor control never depends on frame-rate inference."""

import ctypes
import ctypes.wintypes
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .adapters import AdapterSpec
from .body import WorldState
from .perception import SpatialDetection, TemporalWorldTracker

COCO_LABELS = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


class VisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Path | None = None
    detector_adapter: AdapterSpec | None = None
    capture_backend: Literal["desktop", "windows_graphics"] = "desktop"
    backend: Literal["yolo", "template"] = "yolo"
    target_name: str = Field(default="calibration-target", min_length=1, max_length=80)
    window_title: str = Field(default="VRChat", min_length=1, max_length=100)
    hz: float = Field(default=2.0, ge=0.2, le=10)
    fast_hz: float | None = Field(default=20.0, ge=10, le=30)
    confidence: float = Field(default=0.35, gt=0, lt=1)
    labels: tuple[str, ...] = COCO_LABELS
    player_labels: tuple[str, ...] = ("person",)
    min_player_height_fraction: float = Field(default=0.15, ge=0, le=1)

    @model_validator(mode="after")
    def detector_configured(self):
        if self.detector_adapter is None and self.model is None:
            raise ValueError("choose a detector adapter or model")
        return self


class Detector(Protocol):
    def detect(self, image, timestamp: float) -> list[SpatialDetection]: ...


class TemplateTargetDetector:
    """Supervised stationary-target calibration; not semantic player recognition."""

    def __init__(self, path: Path, *, name: str, confidence: float = 0.85):
        import cv2

        if not 0.7 <= confidence < 1 or not name or len(name) > 80:
            raise ValueError("invalid visual calibration detector configuration")
        self.template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if self.template is None or min(self.template.shape) < 12:
            raise ValueError("calibration template must be a readable image at least 12x12")
        if float(self.template.std()) < 8:
            raise ValueError("calibration template lacks visual texture")
        self.name, self.confidence = name, confidence

    def detect(self, image, timestamp: float) -> list[SpatialDetection]:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if any(a <= b for a, b in zip(gray.shape, self.template.shape)):
            raise ValueError("calibration template must be smaller than the camera image")
        scores = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        if score < self.confidence:
            return []
        height, width = gray.shape
        th, tw = self.template.shape
        # Reject ambiguous repeated objects instead of silently switching target.
        x, y = location
        scores[max(0, y - th) : y + th + 1, max(0, x - tw) : x + tw + 1] = -1
        if cv2.minMaxLoc(scores)[1] > score - 0.08:
            return []
        cx, cy = location[0] + tw / 2, location[1] + th / 2
        offset = (2 * cx / width - 1, 2 * cy / height - 1)
        return [
            SpatialDetection(
                # A unit-depth bearing proxy, never calibrated reach geometry.
                track_id=self.name,
                position=(1.0, -offset[0], 1.6 - offset[1]),
                confidence=min(1.0, score),
                kind="object",
                image_position=offset,
                image_box=(x / width, y / height, (x + tw) / width, (y + th) / height),
            )
        ]


class CaptureUnavailable(RuntimeError):
    """Temporary window occlusion; keep the detector loaded but invalidate frames."""


def vrchat_window(title_contains: str) -> int:
    """Resolve one visible named window without stealing desktop focus."""
    user32 = ctypes.windll.user32
    matches: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(handle, _):
        length = user32.GetWindowTextLengthW(handle)
        if length and user32.IsWindowVisible(handle):
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(handle, buffer, length + 1)
            if title_contains.casefold() in buffer.value.casefold():
                matches.append(handle)
        return True

    user32.EnumWindows(visit, 0)
    if len(matches) != 1:
        raise CaptureUnavailable(
            f"expected one visible {title_contains!r} window, found {len(matches)}"
        )
    return matches[0]


def vrchat_client_rect(title_contains: str) -> dict[str, int]:
    """Return the visible VRChat client rectangle in desktop coordinates."""
    user32 = ctypes.windll.user32
    handle = vrchat_window(title_contains)
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    if user32.IsIconic(handle) or user32.GetForegroundWindow() != handle:
        raise CaptureUnavailable(
            "VRChat must be foreground for desktop capture; perception is pending"
        )
    rect = ctypes.wintypes.RECT()
    if not user32.GetClientRect(handle, ctypes.byref(rect)):
        raise RuntimeError("GetClientRect failed")
    origin = ctypes.wintypes.POINT(0, 0)
    if not user32.ClientToScreen(handle, ctypes.byref(origin)):
        raise RuntimeError("ClientToScreen failed")
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width < 320 or height < 240:
        raise RuntimeError(f"VRChat client is too small: {width}x{height}")
    return {"left": origin.x, "top": origin.y, "width": width, "height": height}


class YoloOnnxDetector:
    """COCO detector producing coarse egocentric geometry for attention decisions."""

    def __init__(
        self,
        model_path: Path,
        *,
        confidence: float = 0.35,
        labels: tuple[str, ...] = COCO_LABELS,
        player_labels: tuple[str, ...] = ("person",),
        min_player_height_fraction: float = 0.15,
    ):
        if (
            not model_path.is_file()
            or not 0 < confidence < 1
            or not labels
            or not all(label and len(label) <= 80 for label in labels)
            or not set(player_labels).issubset(labels)
            or not 0 <= min_player_height_fraction <= 1
        ):
            raise ValueError("invalid vision model configuration")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self.confidence = confidence
        self.labels = labels
        self.player_labels = frozenset(player_labels)
        self.min_player_height_fraction = min_player_height_fraction
        self._next_id = 1
        self._centres: dict[str, tuple[float, float, int]] = {}
        self._seen: dict[str, float] = {}

    def detect(self, image, timestamp: float) -> list[SpatialDetection]:
        import cv2
        import numpy as np

        height, width = image.shape[:2]
        # A single missed frame must not invent a new identity. This is short
        # visual continuity only; never a cross-session/person identity claim.
        self._centres = {
            k: v for k, v in self._centres.items() if timestamp - self._seen.get(k, -100) < 1.5
        }
        scale = min(640 / width, 640 / height)
        resized = cv2.resize(image, (round(width * scale), round(height * scale)))
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        ypad, xpad = (640 - resized.shape[0]) // 2, (640 - resized.shape[1]) // 2
        canvas[ypad : ypad + resized.shape[0], xpad : xpad + resized.shape[1]] = resized
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        output = self._session.run(None, {self._input: tensor})[0][0].T
        boxes, scores, classes = [], [], []
        for row in output:
            class_id = int(np.argmax(row[4:]))
            score = float(row[4 + class_id])
            if score < self.confidence:
                continue
            cx, cy, bw, bh = (float(x) for x in row[:4])
            boxes.append([cx - bw / 2, cy - bh / 2, bw, bh])
            scores.append(score)
            classes.append(class_id)
        keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, 0.45)
        detections = []
        new_centres: dict[str, tuple[float, float, int]] = {}
        used_tracks: set[str] = set()
        for index in keep:
            x, y, bw, bh = boxes[int(index)]
            class_id, score = classes[int(index)], scores[int(index)]
            cx = (x + bw / 2 - xpad) / scale
            cy = (y + bh / 2 - ypad) / scale
            pixel_height = max(1.0, bh / scale)
            label = self.labels[class_id] if class_id < len(self.labels) else "object"
            if (
                label in self.player_labels
                and pixel_height / height < self.min_player_height_fraction
            ):
                continue
            track_id = self._match(class_id, cx, cy, used_tracks)
            used_tracks.add(track_id)
            new_centres[track_id] = (cx, cy, class_id)
            self._seen[track_id] = timestamp
            focal = width * 0.8
            assumed_height = 1.7 if label in self.player_labels else 0.5
            forward = min(8.0, max(0.5, assumed_height * focal / pixel_height))
            left = -(cx - width / 2) * forward / focal
            up = 1.6 - (cy - height / 2) * forward / focal
            detections.append(
                SpatialDetection(
                    track_id=f"{label}-{track_id}",
                    position=(forward, left, up),
                    confidence=score,
                    kind="player" if label in self.player_labels else "object",
                    image_position=(
                        max(-1.0, min(1.0, 2 * cx / width - 1)),
                        max(-1.0, min(1.0, 2 * cy / height - 1)),
                    ),
                    image_box=(
                        max(0, (x - xpad) / scale / width),
                        max(0, (y - ypad) / scale / height),
                        min(1, (x + bw - xpad) / scale / width),
                        min(1, (y + bh - ypad) / scale / height),
                    ),
                )
            )
        self._centres.update(new_centres)
        self._seen = {k: v for k, v in self._seen.items() if k in self._centres}
        return detections

    def _match(self, class_id: int, cx: float, cy: float, used: set[str]) -> str:
        candidates = [
            (key, (x - cx) ** 2 + (y - cy) ** 2)
            for key, (x, y, kind) in self._centres.items()
            if kind == class_id and key not in used
        ]
        candidates.sort(key=lambda item: item[1])
        if candidates and candidates[0][1] < 150**2:
            # At crossings, nearest-centre matching can silently switch people.
            # Start a new anonymous track when two identities are similarly likely.
            if len(candidates) == 1 or candidates[1][1] > candidates[0][1] * 2 + 20**2:
                return candidates[0][0]
        value = str(self._next_id)
        self._next_id += 1
        return value


class LiveVisionLoop:
    """Run capture/detection at a bounded low rate and expose the latest WorldState."""

    def __init__(
        self,
        detector: Detector,
        *,
        title: str = "VRChat",
        hz: float = 2.0,
        capture_backend: str = "desktop",
        fast_hz: float | None = None,
    ):
        if not 0.2 <= hz <= 10:
            raise ValueError("vision rate must be 0.2..10 Hz")
        self.detector, self.title, self.hz = detector, title, hz
        if fast_hz is not None and not 10 <= fast_hz <= 30:
            raise ValueError("fast tracking rate must be 10..30 Hz")
        self.fast_hz = fast_hz
        self.tracking_health = {}
        self._image_world = None
        if capture_backend not in ("desktop", "windows_graphics"):
            raise ValueError("unknown capture backend")
        self.capture_backend = capture_backend
        self.tracker = TemporalWorldTracker(ttl_s=max(1.5, 3 / hz))
        self._world = WorldState()
        self._decision_image = None
        self._lock = Lock()
        self._running = False
        self._thread: Thread | None = None
        self.error: Exception | None = None
        self.capture_pending: str | None = None
        self.frames = 0

    def start(self) -> None:
        self._running = True
        self._thread = Thread(target=self._work, name="myumiq-vision", daemon=True)
        self._thread.start()

    def _work(self) -> None:
        if self.fast_hz is not None:
            from .vision_multirate import run

            run(self)
            return
        if self.capture_backend == "windows_graphics":
            self._window_work()
            return
        try:
            import mss
            import numpy as np

            with mss.mss() as capture:
                while self._running:
                    started = time.perf_counter()
                    try:
                        rect = vrchat_client_rect(self.title)
                    except CaptureUnavailable as exc:
                        with self._lock:
                            self._world = WorldState()
                            self._decision_image = None
                            self.capture_pending = str(exc)
                        time.sleep(1 / self.hz)
                        continue
                    image = np.asarray(capture.grab(rect))[:, :, :3]
                    detections = self.detector.detect(image, started)
                    world = self.tracker.update(detections, started)
                    encoded = decision_image(image)
                    with self._lock:
                        self._world = world
                        self._decision_image = encoded
                        self.capture_pending = None
                    self.frames += 1
                    time.sleep(max(0.0, 1 / self.hz - (time.perf_counter() - started)))
        except Exception as exc:
            self.error = exc
            self._running = False

    def _window_work(self):
        from .window_capture import WindowCapture

        capture = None
        try:
            capture = WindowCapture(self.title, self.hz)
            while self._running:
                began = time.perf_counter()
                try:
                    image, captured_at = capture.read()
                except CaptureUnavailable as exc:
                    with self._lock:
                        self._world = WorldState()
                        self._decision_image = None
                        self.capture_pending = str(exc)
                else:
                    detections = self.detector.detect(image, captured_at)
                    world = self.tracker.update(detections, captured_at)
                    encoded = decision_image(image)
                    with self._lock:
                        self._world, self._decision_image = world, encoded
                        self.capture_pending = None
                    self.frames += 1
                time.sleep(max(0.0, 1 / self.hz - (time.perf_counter() - began)))
        except Exception as exc:
            self.error = exc
            self._running = False
        finally:
            if capture is not None:
                capture.close()

    def world(self) -> WorldState:
        if self.error is not None:
            raise RuntimeError(f"vision loop failed: {self.error}")
        with self._lock:
            return self._world

    def decision_snapshot(self):
        """Read the coherent pair already encoded by the vision worker."""
        if self.error is not None:
            raise RuntimeError("vision loop unavailable")
        with self._lock:
            world, encoded = self._image_world or self._world, self._decision_image
        if encoded is None or world.timestamp is None:
            raise RuntimeError("no decision image yet")
        return world, encoded, world.timestamp

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)


def decision_image(frame):
    """Bound the image portion, leaving room for structured state in model requests."""
    import base64

    import cv2

    for side in (640, 512, 384, 256):
        ratio = min(1.0, side / max(frame.shape[:2]))
        image = cv2.resize(
            frame, (max(1, round(frame.shape[1] * ratio)), max(1, round(frame.shape[0] * ratio)))
        )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise RuntimeError("decision image encoding failed")
        data = base64.b64encode(encoded).decode()
        if len(data) <= 80000:
            return data
    raise RuntimeError("decision image exceeded the size limit")

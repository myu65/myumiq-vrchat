"""Single-window WGC frames; independent of desktop focus, bounded and freshness checked."""

import ctypes
import ctypes.wintypes
import multiprocessing
import time
from threading import Lock

from .vision import CaptureUnavailable, vrchat_window


def crop_client(image, bounds, origin, size):
    """Exclude window chrome without guessing its thickness or DPI scale."""
    left, top, right, bottom = bounds
    x, y = origin[0] - left, origin[1] - top
    width, height = size
    if (
        image.shape[:2] != (bottom - top, right - left)
        or width < 320
        or height < 240
        or x < 0
        or y < 0
        or x + width > image.shape[1]
        or y + height > image.shape[0]
    ):
        raise CaptureUnavailable("window geometry changed; awaiting matching capture")
    return image[y : y + height, x : x + width]


class LatestWindowFrame:
    """Copy borrowed native pixels once; never rejuvenate a repeated capture timestamp."""

    def __init__(self, hz):
        self.interval = min(0.1, 1 / hz)
        self.max_age = max(0.5, 2 / hz)
        self.frame = None
        self.timestamp = -float("inf")
        self.source_timestamp = -float("inf")
        self.closed = False
        self.lock = Lock()

    def accept(self, image, captured_at, now):
        # Compositor presentation timestamps can lead callback arrival by one
        # display interval. Bound that skew and never publish a future timestamp.
        if not -0.05 <= now - captured_at <= self.max_age:
            return
        with self.lock:
            if self.closed or captured_at - self.source_timestamp < self.interval:
                return
            if image.ndim != 3 or image.shape[2] != 4 or min(image.shape[:2]) < 240:
                return
            self.frame = image[:, :, :3].copy()
            self.source_timestamp = captured_at
            self.timestamp = min(captured_at, now)

    def read(self, now):
        with self.lock:
            if self.closed or self.frame is None or not 0 <= now - self.timestamp <= self.max_age:
                raise CaptureUnavailable("window capture has no fresh frame")
            return self.frame, self.timestamp

    def close(self):
        with self.lock:
            self.closed, self.frame = True, None


class NativeWindowCapture:
    def __init__(self, title, hz):
        from windows_capture import WindowsCapture

        self.handle = vrchat_window(title)
        self.latest = LatestWindowFrame(hz)
        self.capture = WindowsCapture(
            cursor_capture=False, draw_border=None, window_hwnd=self.handle, monitor_index=None
        )

        @self.capture.event
        def on_frame_arrived(frame, control):
            if self.latest.closed:
                control.stop()
                return
            # WGC SystemRelativeTime is a QPC timestamp in 100 ns ticks.
            self.latest.accept(frame.frame_buffer, frame.timespan / 10_000_000, time.perf_counter())

        @self.capture.event
        def on_closed():
            self.latest.close()

        self.control = self.capture.start_free_threaded()

    def read(self):
        user32 = ctypes.windll.user32
        handle = ctypes.c_void_p(self.handle)
        if not user32.IsWindow(handle) or self.latest.closed:
            # VRChat replaces its startup HWND on some display transitions.
            # Let the existing service lifecycle resolve a fresh unique window.
            raise RuntimeError("captured window closed; capture must be reopened")
        if user32.IsIconic(handle):
            raise CaptureUnavailable("captured window is minimized")
        image, captured_at = self.latest.read(time.perf_counter())
        bounds, client = ctypes.wintypes.RECT(), ctypes.wintypes.RECT()
        origin = ctypes.wintypes.POINT()
        if (
            ctypes.windll.dwmapi.DwmGetWindowAttribute(
                handle, 9, ctypes.byref(bounds), ctypes.sizeof(bounds)
            )
            != 0
            or not user32.GetClientRect(handle, ctypes.byref(client))
            or not user32.ClientToScreen(handle, ctypes.byref(origin))
        ):
            raise CaptureUnavailable("window client geometry unavailable")
        image = crop_client(
            image,
            (bounds.left, bounds.top, bounds.right, bounds.bottom),
            (origin.x, origin.y),
            (client.right, client.bottom),
        )
        return image, captured_at

    def close(self):
        self.latest.close()
        self.control.stop()


def _capture_worker(connection, title, hz):
    """All native capture lifetime stays inside this disposable child process."""
    capture = None
    try:
        capture = NativeWindowCapture(title, hz)
        while connection.recv() == "frame":
            try:
                image, captured_at = capture.read()
                if max(image.shape[:2]) > 1920:
                    import cv2

                    ratio = 1920 / max(image.shape[:2])
                    image = cv2.resize(
                        image, (round(image.shape[1] * ratio), round(image.shape[0] * ratio))
                    )
                connection.send(("frame", image, captured_at))
            except CaptureUnavailable as exc:
                connection.send(("pending", str(exc)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(("error", str(exc)[:300]))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()
        if capture is not None:
            capture.close()


class WindowCapture:
    """One outstanding frame request; native GIL/driver stalls cannot block the agent."""

    def __init__(self, title, hz):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_capture_worker,
            args=(child, title, hz),
            name="myumiq-window-capture",
            daemon=True,
        )
        self.process.start()
        child.close()
        self.pending_since = None
        self.latest = None
        self.max_age = max(0.5, 2 / hz)
        self.closed = False

    def read(self):
        if self.closed:
            raise RuntimeError("capture process is closed")
        now = time.perf_counter()
        if self.pending_since is None:
            self.connection.send("frame")
            self.pending_since = now
        # A fresh on-demand frame avoids an extra perception interval of lag.
        # This bounded wait is only on the vision worker, never the motor loop.
        if self.connection.poll(0.03):
            message = self.connection.recv()
            self.pending_since = None
            if message[0] == "frame":
                self.latest = message[1:]
            elif message[0] == "error":
                raise RuntimeError(message[1])
            else:
                self.latest = None
        if not self.process.is_alive():
            raise RuntimeError("capture process exited")
        if self.pending_since is not None and now - self.pending_since > 10:
            raise RuntimeError("capture process response deadline expired")
        if self.latest is None or not 0 <= time.perf_counter() - self.latest[1] <= self.max_age:
            raise CaptureUnavailable("waiting for a fresh window frame")
        return self.latest

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.is_alive():
                self.connection.send("close")
                self.process.join(0.2)
        except (EOFError, BrokenPipeError, OSError):
            pass
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(0.5)
            self.connection.close()
            if not self.process.is_alive():
                self.process.close()

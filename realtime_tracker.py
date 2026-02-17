"""
High-Performance Real-Time Visual Tracking System
=================================================
Single-file implementation with:
- Screen capture: dxcam (preferred) or mss fallback
- Frame processing: OpenCV (cv2)
- Object detection: Ultralytics YOLO with GPU acceleration
- Target selection: highest_confidence | closest_to_center
- Mouse smoothing: exponential smoothing | PID controller
- Threaded capture + inference pipeline
- FPS counter and performance logging

Usage:
    python realtime_tracker.py
"""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------


class SelectionStrategy(Enum):
    HIGHEST_CONFIDENCE = "highest_confidence"
    CLOSEST_TO_CENTER = "closest_to_center"


class SmoothingMode(Enum):
    EXPONENTIAL = "exponential"
    PID = "pid"


@dataclass
class TrackerConfig:
    # Model (use .onnx for ONNX Runtime, .pt for PyTorch)
    model_path: str = "models/yolov8n.pt"
    conf_threshold: float = 0.5
    device: str = "cuda"  # "cuda" | "cpu" | "dml" (AMD)
    half: bool = False  # FP16 for CUDA

    # Capture & preprocess
    capture_region: Optional[Tuple[int, int, int, int]] = None
    inference_size: Optional[Tuple[int, int]] = None  # (w, h) to resize for faster inference
    target_fps: int = 60
    use_dxcam: bool = True  # False = mss fallback

    # Target selection
    selection_strategy: SelectionStrategy = SelectionStrategy.CLOSEST_TO_CENTER
    preferred_class: Optional[int] = 0  # 0 = person, None = all
    exclusion_zones: List[Tuple[int, int, int, int]] = None  # (left, top, right, bottom) in frame coords
    exclude_own_windows: bool = True  # Eigene Fenster nicht verfolgen
    exclude_center_zone: bool = True  # Zentrum ausschließen (eigene Arme/Waffe in FPS)
    center_exclusion_ratio: float = 0.35  # Breite: 0.35 = 70% der Bildmitte (erhöht von 0.25)
    center_exclusion_extend_bottom: float = 0.4  # Zusätzlich unten (Waffe): 0.4 = Zone reicht 40% weiter nach unten (erhöht)
    min_distance_from_center: int = 150  # Ziel näher als X Pixel zur Mitte ignorieren (erhöht von 80)

    # Mouse smoothing
    smoothing_mode: SmoothingMode = SmoothingMode.EXPONENTIAL
    smoothing_alpha: float = 0.35  # 0.0 = max smooth, 1.0 = direct (exponential)
    # PID gains (when mode=PID)
    pid_kp: float = 0.15
    pid_ki: float = 0.02
    pid_kd: float = 0.08
    pid_max_output: float = 25.0

    # Mouse mode
    use_relative_mouse: bool = True  # delta movement for FPS crosshair

    # Performance
    log_fps_interval: float = 1.0
    log_latency: bool = True

    def __post_init__(self):
        if self.exclusion_zones is None:
            self.exclusion_zones = []


# -----------------------------------------------------------------------------
# SCREEN CAPTURE
# -----------------------------------------------------------------------------


class ScreenCapture:
    """Threaded screen capture using dxcam (preferred) or mss."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._camera = None
        self._mss = None
        self._region = config.capture_region
        self._init_backend()

    def _init_backend(self):
        if self.config.use_dxcam:
            try:
                import dxcam
                self._camera = dxcam.create(output_color="BGR")
                print("[Capture] Using dxcam (low-latency)")
            except (ImportError, Exception) as e:
                print(f"[Capture] dxcam unavailable ({e}), falling back to mss")
                self._camera = None

        if self._camera is None:
            try:
                import mss
                self._mss = mss.mss()
                print("[Capture] Using mss")
            except ImportError:
                raise RuntimeError("Need dxcam or mss. Install: pip install dxcam mss")

    def start(self):
        self._running = True
        if self._camera:
            self._camera.start(
                target_fps=self.config.target_fps,
                video_mode=True,
                region=self._region,
            )
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="Capture")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._camera:
            self._camera.stop()
        if self._mss:
            self._mss.close()

    def _capture_loop(self):
        if self._camera:
            self._capture_dxcam()
        else:
            self._capture_mss()

    def _capture_dxcam(self):
        interval = 1.0 / self.config.target_fps if self.config.target_fps > 0 else 0
        last = time.perf_counter()
        while self._running:
            frame = self._camera.get_latest_frame()
            if frame is not None:
                with self._lock:
                    self._latest_frame = np.ascontiguousarray(frame)  # ensure layout
            dt = interval - (time.perf_counter() - last)
            if dt > 0:
                time.sleep(min(dt, 0.001))
            last = time.perf_counter()

    def _capture_mss(self):
        import mss
        monitor = self._region
        if monitor:
            monitor = {
                "left": monitor[0],
                "top": monitor[1],
                "width": monitor[2] - monitor[0],
                "height": monitor[3] - monitor[1],
            }
        else:
            monitor = self._mss.monitors[0]
        interval = 1.0 / self.config.target_fps if self.config.target_fps > 0 else 0
        last = time.perf_counter()
        while self._running:
            shot = self._mss.grab(monitor)
            # BGRA -> BGR, no copy if possible
            frame = np.array(shot)[:, :, :3]
            with self._lock:
                self._latest_frame = np.ascontiguousarray(frame)
            dt = interval - (time.perf_counter() - last)
            if dt > 0:
                time.sleep(min(dt, 0.001))
            last = time.perf_counter()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame


# -----------------------------------------------------------------------------
# DETECTOR
# -----------------------------------------------------------------------------


class YOLODetector:
    """YOLO object detector with GPU acceleration."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._device = self._resolve_device()
        from ultralytics import YOLO
        self._model = YOLO(config.model_path)
        self._model.to(self._device)
        print(f"[Detector] Model on {self._device}")

    def _resolve_device(self):
        if self.config.device == "dml":
            try:
                import torch
                import torch_directml
                dev = torch_directml.device()
                print("[Detector] AMD GPU (DirectML)")
                return dev
            except ImportError:
                return "cpu"
        if self.config.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"[Detector] NVIDIA GPU: {torch.cuda.get_device_name(0)}")
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
        """
        Returns (detections, scale_to_original).
        detections: Nx6 [x1,y1,x2,y2,conf,cls] in inference frame coords.
        scale_to_original: (sx, sy) to map back to capture coords, or None if no resize.
        """
        orig_h, orig_w = frame.shape[:2]
        scale = None
        if self.config.inference_size is not None:
            import cv2
            iw, ih = self.config.inference_size
            frame = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_LINEAR)
            scale = (orig_w / iw, orig_h / ih)
        
        try:
            results = self._model(
                frame,
                conf=self.config.conf_threshold,
                verbose=False,
                device=self._device,
                half=self.config.half and str(self._device) == "cuda",
            )[0]
        except RuntimeError as e:
            # DirectML/torch_directml compatibility issue - fallback to CPU
            if "version_counter" in str(e) and str(self._device) != "cpu":
                print(f"[Detector] DirectML error detected, falling back to CPU")
                self._device = "cpu"
                self._model.to("cpu")
                results = self._model(
                    frame,
                    conf=self.config.conf_threshold,
                    verbose=False,
                    device="cpu",
                )[0]
            else:
                raise
        
        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32), scale
        # Vectorized extraction
        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = results.boxes.cls.cpu().numpy().reshape(-1, 1).astype(np.float32)
        return np.hstack([xyxy, conf, cls]), scale


# -----------------------------------------------------------------------------
# EXCLUSION ZONES (eigene Fenster nicht verfolgen)
# -----------------------------------------------------------------------------


def _get_own_window_rects() -> List[Tuple[int, int, int, int]]:
    """Returns screen rects (left, top, right, bottom) of this process's visible windows."""
    try:
        import os
        import win32gui
        import win32process

        own_pid = os.getpid()
        rects = []

        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    if pid == own_pid:
                        rects.append(win32gui.GetWindowRect(hwnd))
                except Exception:
                    pass
            return True

        win32gui.EnumWindows(callback, None)
        return rects
    except ImportError:
        return []


def _screen_to_frame_rect(
    rect: Tuple[int, int, int, int],
    region: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """Convert screen rect to frame coords; returns None if outside capture area."""
    sl, st, sr, sb = rect
    if region is None:
        return (sl, st, sr, sb)
    rl, rt, rr, rb = region
    fl = max(0, sl - rl)
    ft = max(0, st - rt)
    fr = min(rr - rl, sr - rl)
    fb = min(rb - rt, sb - rt)
    if fl >= fr or ft >= fb:
        return None
    return (fl, ft, fr, fb)


# -----------------------------------------------------------------------------
# TARGET SELECTOR
# -----------------------------------------------------------------------------


def _point_in_rect(x: float, y: float, rect: Tuple[int, int, int, int]) -> bool:
    """Check if point (x,y) is inside rect (left, top, right, bottom)."""
    l, t, r, b = rect
    return l <= x <= r and t <= y <= b


def select_target(
    detections: np.ndarray,
    center_x: float,
    center_y: float,
    strategy: SelectionStrategy,
    preferred_class: Optional[int],
    exclusion_zones: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Optional[Tuple[float, float]]:
    """
    Select best target from detections [Nx6: x1,y1,x2,y2,conf,cls].
    Skips targets whose centroid is inside any exclusion zone.
    AIM FOR HEAD: Returns head position (cx, cy_head) instead of body center for better headshots.
    """
    if detections.size == 0:
        return None
    exclusion_zones = exclusion_zones or []
    if preferred_class is not None:
        mask = detections[:, 5] == preferred_class
        if not np.any(mask):
            detections = detections
        else:
            detections = detections[mask]
    if detections.size == 0:
        return None

    # Horizontal center (same as before)
    cx = (detections[:, 0] + detections[:, 2]) * 0.5
    # AIM FOR HEAD: Target ~30% from top of bounding box (head region) instead of center
    # Formula: y1 + 0.3 * (y2 - y1) puts crosshair in head zone for better accuracy
    cy = detections[:, 1] + (detections[:, 3] - detections[:, 1]) * 0.3

    if strategy == SelectionStrategy.HIGHEST_CONFIDENCE:
        order = np.argsort(-detections[:, 4])  # descending conf
    else:
        dist_sq = (cx - center_x) ** 2 + (cy - center_y) ** 2
        order = np.argsort(dist_sq)

    for idx in order:
        px, py = float(cx[idx]), float(cy[idx])
        in_exclusion = any(_point_in_rect(px, py, z) for z in exclusion_zones)
        if not in_exclusion:
            return (px, py)
    return None


# -----------------------------------------------------------------------------
# MOUSE SMOOTHING
# -----------------------------------------------------------------------------


class ExponentialSmoothing:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    def update(self, target_x: float, target_y: float) -> Tuple[float, float]:
        if self._x is None:
            self._x, self._y = target_x, target_y
        else:
            self._x = self.alpha * target_x + (1 - self.alpha) * self._x
            self._y = self.alpha * target_y + (1 - self.alpha) * self._y
        return (self._x, self._y)

    def reset(self):
        self._x = self._y = None


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, max_output: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.max_output = max_output
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._last_time: Optional[float] = None

    def update(
        self,
        current_x: float,
        current_y: float,
        target_x: float,
        target_y: float,
    ) -> Tuple[float, float]:
        now = time.perf_counter()
        dt = now - self._last_time if self._last_time else 0.016
        self._last_time = now
        dt = max(dt, 1e-6)

        ex = target_x - current_x
        ey = target_y - current_y

        self._integral_x += ex * dt
        self._integral_y += ey * dt
        self._integral_x = np.clip(self._integral_x, -100, 100)
        self._integral_y = np.clip(self._integral_y, -100, 100)

        dx = self.kp * ex + self.ki * self._integral_x + self.kd * (ex - self._prev_error_x) / dt
        dy = self.kp * ey + self.ki * self._integral_y + self.kd * (ey - self._prev_error_y) / dt

        self._prev_error_x, self._prev_error_y = ex, ey

        mag = np.sqrt(dx * dx + dy * dy)
        if mag > self.max_output:
            scale = self.max_output / mag
            dx *= scale
            dy *= scale
        return (dx, dy)

    def reset(self):
        self._integral_x = self._integral_y = 0.0
        self._prev_error_x = self._prev_error_y = 0.0
        self._last_time = None


class MouseController:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self._user32 = ctypes.windll.user32
        self._screen_w = self._user32.GetSystemMetrics(0)
        self._screen_h = self._user32.GetSystemMetrics(1)
        if config.smoothing_mode == SmoothingMode.EXPONENTIAL:
            self._smoother = ExponentialSmoothing(config.smoothing_alpha)
        else:
            self._smoother = PIDController(
                config.pid_kp,
                config.pid_ki,
                config.pid_kd,
                config.pid_max_output,
            )
        self._relative = config.use_relative_mouse
        self._last_x = self._screen_w / 2
        self._last_y = self._screen_h / 2

    @staticmethod
    def region_to_screen(
        x: float, y: float,
        region: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[float, float]:
        if region is None:
            return (x, y)
        return (region[0] + x, region[1] + y)

    def _get_screen_center(self, region: Optional[Tuple[int, int, int, int]]) -> Tuple[float, float]:
        """Screen coords of capture area center (crosshair position for FPS)."""
        if region is None:
            return (self._screen_w / 2, self._screen_h / 2)
        cx = (region[0] + region[2]) / 2.0
        cy = (region[1] + region[3]) / 2.0
        return (cx, cy)

    def update_and_move(
        self,
        target_x: float,
        target_y: float,
        region: Optional[Tuple[int, int, int, int]],
    ):
        screen_x, screen_y = self.region_to_screen(target_x, target_y, region)
        if self.config.smoothing_mode == SmoothingMode.EXPONENTIAL:
            out_x, out_y = self._smoother.update(screen_x, screen_y)
            self._last_x, self._last_y = out_x, out_y
            if self._relative:
                dx, dy = out_x - self._get_screen_center(region)[0], out_y - self._get_screen_center(region)[1]
                self._move_relative(dx, dy)
            else:
                self._move_absolute(out_x, out_y)
        else:
            if self._relative:
                curr_x, curr_y = self._get_screen_center(region)
            else:
                curr_x, curr_y = self._last_x, self._last_y
            dx, dy = self._smoother.update(curr_x, curr_y, screen_x, screen_y)
            if self._relative:
                self._move_relative(dx, dy)
            else:
                self._last_x = np.clip(self._last_x + dx, 0, self._screen_w - 1)
                self._last_y = np.clip(self._last_y + dy, 0, self._screen_h - 1)
                self._move_absolute(self._last_x, self._last_y)

    def _move_absolute(self, x: float, y: float):
        x = np.clip(x, 0, self._screen_w - 1)
        y = np.clip(y, 0, self._screen_h - 1)
        self._user32.SetCursorPos(int(x), int(y))

    def _move_relative(self, dx: float, dy: float):
        MOUSEEVENTF_MOVE = 0x0001
        self._user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

    def reset_smoothing(self):
        self._smoother.reset()


# -----------------------------------------------------------------------------
# FPS / LATENCY LOGGING
# -----------------------------------------------------------------------------


class PerformanceLogger:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self._count = 0
        self._last_time = time.perf_counter()
        self._latencies: list = []
        self._max_latency_samples = 60

    def tick(self, latency_ms: Optional[float] = None):
        self._count += 1
        if latency_ms is not None and self.config.log_latency:
            self._latencies.append(latency_ms)
            if len(self._latencies) > self._max_latency_samples:
                self._latencies.pop(0)
        now = time.perf_counter()
        if now - self._last_time >= self.config.log_fps_interval:
            fps = self._count / (now - self._last_time)
            msg = f"[Perf] {fps:.1f} FPS"
            if self._latencies:
                msg += f" | Latency: p50={np.median(self._latencies):.1f}ms p99={np.percentile(self._latencies, 99):.1f}ms"
            print(msg)
            self._count = 0
            self._last_time = now
            self._latencies.clear()


# -----------------------------------------------------------------------------
# MAIN TRACKING PIPELINE
# -----------------------------------------------------------------------------


class RealtimeTracker:
    """High-performance real-time visual tracking system."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._running = False
        self._infer_thread: Optional[threading.Thread] = None
        self._exclusion_cache: List[Tuple[int, int, int, int]] = []
        self._exclusion_cache_time = 0.0

        self._capture = ScreenCapture(config)
        self._detector = YOLODetector(config)
        self._mouse = MouseController(config)
        self._perf = PerformanceLogger(config)

        # Frame center (for target selection) - in inference frame coords
        inf_size = config.inference_size
        if inf_size:
            self._center_x = inf_size[0] / 2.0
            self._center_y = inf_size[1] / 2.0
        elif config.capture_region:
            r = config.capture_region
            self._center_x = (r[2] - r[0]) / 2.0
            self._center_y = (r[3] - r[1]) / 2.0
        else:
            self._center_x = self._mouse._screen_w / 2.0
            self._center_y = self._mouse._screen_h / 2.0

    def start(self):
        self._running = True
        self._capture.start()
        self._infer_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="Inference",
        )
        self._infer_thread.start()
        print("\n[Tracker] Running. Press Ctrl+C to stop.\n")

    def stop(self):
        self._running = False
        self._capture.stop()
        if self._infer_thread:
            self._infer_thread.join(timeout=2.0)
        print("[Tracker] Stopped.")

    def _inference_loop(self):
        while self._running:
            t0 = time.perf_counter()
            frame = self._capture.get_frame()
            if frame is None:
                time.sleep(0.0005)
                continue

            # Run detection (frame passed by ref where possible)
            detections, scale = self._detector.detect(frame)
            t1 = time.perf_counter()

            # Exclusion zones (eigene Fenster + Bildmitte nicht verfolgen)
            exclusion_zones = list(self.config.exclusion_zones)
            # Zentrum: eigene Arme/Waffe in FPS (asymmetrisch – Waffe unten)
            if self.config.exclude_center_zone:
                fw = frame.shape[1]
                fh = frame.shape[0]
                rx = self.config.center_exclusion_ratio / 2.0
                ry_top = rx
                ry_bottom = rx + self.config.center_exclusion_extend_bottom
                cx, cy = fw / 2.0, fh / 2.0
                exclusion_zones.append((
                    int(cx - fw * rx), int(cy - fh * ry_top),
                    int(cx + fw * rx), int(cy + fh * ry_bottom),
                ))
            if self.config.exclude_own_windows and time.perf_counter() - self._exclusion_cache_time > 0.5:
                self._exclusion_cache = []
                for rect in _get_own_window_rects():
                    fr = _screen_to_frame_rect(rect, self.config.capture_region)
                    if fr is not None:
                        self._exclusion_cache.append(fr)
                self._exclusion_cache_time = time.perf_counter()
            exclusion_zones.extend(self._exclusion_cache)

            if scale is not None:
                sx, sy = scale[0], scale[1]
                exclusion_zones = [
                    (z[0] / sx, z[1] / sy, z[2] / sx, z[3] / sy)
                    for z in exclusion_zones
                ]

            target = select_target(
                detections,
                self._center_x,
                self._center_y,
                self.config.selection_strategy,
                self.config.preferred_class,
                exclusion_zones,
            )

            if target is not None:
                tx, ty = target[0], target[1]
                if scale is not None:
                    tx *= scale[0]
                    ty *= scale[1]
                # Fallback: Ziel zu nah an Mitte = vermutlich eigenes Modell
                fw = frame.shape[1]
                fh = frame.shape[0]
                dx = tx - fw / 2.0
                dy = ty - fh / 2.0
                dist = (dx * dx + dy * dy) ** 0.5
                if dist >= self.config.min_distance_from_center:
                    self._mouse.update_and_move(tx, ty, self.config.capture_region)

            latency_ms = (t1 - t0) * 1000
            self._perf.tick(latency_ms)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------


def main():
    from pathlib import Path
    Path("models").mkdir(exist_ok=True)

    config = TrackerConfig(
        model_path="models/yolov8n.pt",  # ⭐ RX 6750 XT: yolov8n stabiler mit DirectML (60 FPS, gute Accuracy mit Tweaks)
        conf_threshold=0.55,  # Leicht erhöht - reduziert False Positives
        device="dml",  # AMD DirectML - optimiert für RX 6750 XT
        capture_region=None,
        target_fps=60,
        use_dxcam=True,
        selection_strategy=SelectionStrategy.CLOSEST_TO_CENTER,
        preferred_class=0,  # person
        exclusion_zones=[],
        exclude_own_windows=True,  # Eigene Fenster (GUI, Preview) nicht verfolgen
        exclude_center_zone=True,
        center_exclusion_ratio=0.35,  # ⭐ ERHÖHT - größere Dead-Zone um Fadenkreuz
        center_exclusion_extend_bottom=0.4,  # ⭐ ERHÖHT - verhindert Self-Focus
        min_distance_from_center=150,  # ⭐ ERHÖHT - 150px Fallback-Distanz statt 80px
        smoothing_mode=SmoothingMode.EXPONENTIAL,
        smoothing_alpha=0.35,
        use_relative_mouse=True,
        log_fps_interval=1.0,
        log_latency=True,
    )

    tracker = RealtimeTracker(config)
    try:
        tracker.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()

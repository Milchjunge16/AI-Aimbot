"""
Screen Capture Modul - Verantwortlich für das Aufnehmen des Bildschirms.
Verwendet dxcam für hohe Performance und geringe Latenz.
"""
import threading
import time
from typing import Optional, Tuple

import dxcam
import numpy as np

from config import Config


class ScreenCapture:
    """Erfasst den Bildschirm in einem separaten Thread für maximale Performance."""
    
    def __init__(self, config: Config):
        self.config = config
        self.camera = dxcam.create(output_color="BGR")
        self.region = config.capture_region
        self.target_fps = config.target_fps
        
        # Thread-sichere Frame-Speicherung
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
    def start(self):
        """Startet die Bildschirmaufnahme."""
        self.running = True
        self.camera.start(
            target_fps=self.target_fps, 
            video_mode=True, 
            region=self.region
        )
        self.thread = threading.Thread(
            target=self._capture_loop, 
            daemon=True,
            name="CaptureThread"
        )
        self.thread.start()
        print(f"📸 Screen Capture gestartet - {self.target_fps} FPS Ziel")
        if self.region:
            print(f"   Region: {self.region}")
        
    def stop(self):
        """Stoppt die Bildschirmaufnahme."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.camera.stop()
        print("📸 Screen Capture gestoppt")
        
    def _capture_loop(self):
        """Interne Schleife für kontinuierliche Aufnahme."""
        while self.running:
            frame = self.camera.get_latest_frame()  # Non-blocking
            if frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.001)  # CPU-Entlastung
                
    def get_frame(self) -> Optional[np.ndarray]:
        """Gibt den neuesten Frame zurück (thread-safe)."""
        with self.frame_lock:
            return self.latest_frame
            
    def update_region(self, new_region: Optional[Tuple[int, int, int, int]]):
        """Aktualisiert die Capture-Region (z.B. wenn Fenster bewegt wird)."""
        if new_region != self.region:
            self.region = new_region
            # Kamera neustarten mit neuer Region
            self.camera.stop()
            self.camera.start(
                target_fps=self.target_fps, 
                video_mode=True, 
                region=self.region
            )
            print(f"📸 Capture-Region aktualisiert: {new_region}")
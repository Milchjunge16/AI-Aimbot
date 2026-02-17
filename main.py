"""
Hauptprogramm des AI Aimbot Tracking-Systems.
Startet und koordiniert alle Komponenten.
Unterstützt Hotkey-Toggle (F6) und Auto-Shooting.
"""
import threading
import time
import sys
from pathlib import Path

# Stelle sicher, dass wir im richtigen Verzeichnis sind
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from core.capture import ScreenCapture
from core.detector import ObjectDetector
from core.selector import TargetSelector
from core.mouse_controller import MouseController
from utils.fps_counter import FPSCounter
from utils.hotkey_manager import get_hotkey_manager

# Optional: Window-Tracking
try:
    from utils.window_selector import WindowSelector
    HAS_WINDOW_SELECTOR = True
except ImportError:
    HAS_WINDOW_SELECTOR = False


class TrackingSystem:
    """Haupt-Koordinator des Tracking-Systems."""
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.tracking_enabled = True  # Für Hotkey-Toggle
        self.threads = []
        
        # Bildschirmgröße ermitteln
        import ctypes
        self.screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        
        # Komponenten initialisieren
        print("\n" + "="*60)
        print("🎯 AI AIMBOT TRACKING SYSTEM - ERWEITERTE EDITION".center(60))
        print("="*60)
        
        self.capture = ScreenCapture(config)
        self.detector = ObjectDetector(config)
        self.selector = TargetSelector(config, self.screen_width, self.screen_height)
        self.mouse = MouseController(config)
        
        # Optional: Fenster-Tracking
        self.window_selector = None
        if config.track_window and HAS_WINDOW_SELECTOR:
            self.window_selector = WindowSelector()
            if self.window_selector.select_window(config.window_title):
                config.capture_region = self.window_selector.get_capture_region()
                self.capture.update_region(config.capture_region)
        
        # FPS Counter
        self.capture_fps = FPSCounter("Capture", config.log_fps_interval)
        self.inference_fps = FPSCounter("Inference", config.log_fps_interval)
        self.total_fps = FPSCounter("Total", config.log_fps_interval)
        
        # Auto-Shoot Tracking
        self.last_shoot_time = 0
        self.shoot_cooldown = config.shoot_cooldown_ms / 1000.0  # In Sekunden
        
        # Hotkey Manager
        self.hotkey_manager = get_hotkey_manager()
        self.hotkey_manager.register_hotkey('f6', self._toggle_tracking)
        
        self.last_window_check = 0
        
    def start(self):
        """Startet das Tracking-System."""
        self.running = True
        self.capture.start()
        
        # Hotkey-Manager starten
        self.hotkey_manager.start()
        
        # Inference Thread starten
        infer_thread = threading.Thread(
            target=self._inference_loop, 
            daemon=True,
            name="InferenceThread"
        )
        infer_thread.start()
        self.threads.append(infer_thread)
        
        print("\n✨ System bereit!")
        print(f"   🎮 F6 = Toggle Tracking (aktuell: {'EIN' if self.tracking_enabled else 'AUS'})")
        print(f"   🔫 Auto-Shoot: {'✅ EIN' if self.config.auto_shoot else '❌ AUS'}")
        if self.config.auto_shoot:
            print(f"   ⏲️  Schuss-Cooldown: {self.config.shoot_cooldown_ms:.0f}ms")
        print("   📊 Drücke Ctrl+C zum Beenden\n")
        
        try:
            while self.running:
                time.sleep(0.1)
                self.total_fps.tick()
        except KeyboardInterrupt:
            self.stop()
    
    def _toggle_tracking(self):
        """Toggle für Tracking Ein/Aus (Hotkey F6)."""
        self.tracking_enabled = not self.tracking_enabled
        status = "✅ AKTIVIERT" if self.tracking_enabled else "❌ DEAKTIVIERT"
        print(f"\n🎮 Tracking {status}\n")
        
        if not self.tracking_enabled:
            # Reset smoothing wenn deaktiviert
            self.mouse.reset_smoothing()
    
    def stop(self):
        """Stoppt das Tracking-System sauber."""
        print("\n\n🛑 Stoppe Tracking-System...")
        self.running = False
        self.capture.stop()
        self.hotkey_manager.stop()
        
        for t in self.threads:
            try:
                t.join(timeout=1.0)
            except:
                pass
            
        print("✅ System gestoppt")
        
    def _inference_loop(self):
        """Hauptschleife für Detektion und Tracking."""
        while self.running:
            # Fenster-Position aktualisieren (wenn aktiv)
            if self.window_selector and time.time() - self.last_window_check > 1.0:
                region = self.window_selector.get_capture_region()
                if region:
                    self.capture.update_region(region)
                self.last_window_check = time.time()
            
            # Frame holen
            frame = self.capture.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
                
            self.capture_fps.tick()
            
            # Nur wenn Tracking aktiviert ist
            if not self.tracking_enabled:
                time.sleep(0.01)
                continue
            
            # Detektion durchführen
            detections = self.detector.detect(frame)
            self.inference_fps.tick()
            
            # Ziel auswählen
            target = self.selector.select(detections)
            
            # Maus bewegen (wenn Ziel gefunden); Region-Offset anwenden wenn Teilbereich erfasst wird
            if target is not None:
                tx, ty = target
                screen_x, screen_y = MouseController.region_to_screen(tx, ty, self.capture.region)
                sx, sy = self.mouse.update(screen_x, screen_y)
                self.mouse.move_to(sx, sy)
                
                # Auto-Shoot wenn aktiv und Ziel nah genug
                if self.config.auto_shoot:
                    self._try_shoot(target)
                
                if self.config.debug_mode:
                    print(f"🎯 Ziel (Region): ({tx:.0f}, {ty:.0f}) → Bildschirm: ({screen_x:.0f}, {screen_y:.0f}) → Maus: ({sx:.0f}, {sy:.0f})")
    
    def _try_shoot(self, target):
        """
        Schießt wenn das Ziel nah genug am Fadenkreuz ist und Cooldown abgelaufen.
        
        Args:
            target: (x, y) Koordinaten des Ziels
        """
        current_time = time.time()
        
        # Cooldown prüfen
        if current_time - self.last_shoot_time < self.shoot_cooldown:
            return
        
        # Abstand vom Bildschirm-Mittelpunkt berechnen
        tx, ty = target
        center_x = self.capture.region[0] + (self.capture.region[2] - self.capture.region[0]) / 2 if self.capture.region else self.screen_width / 2
        center_y = self.capture.region[1] + (self.capture.region[3] - self.capture.region[1]) / 2 if self.capture.region else self.screen_height / 2
        
        # Erneut berechnen basierend auf ScreenCoords statt Region
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        
        distance = ((tx - center_x) ** 2 + (ty - center_y) ** 2) ** 0.5
        
        if distance <= self.config.shoot_threshold_px:
            # Schießen!
            try:
                MouseController.click_left()
                self.last_shoot_time = current_time
                if self.config.debug_mode:
                    print(f"💥 SCHUSS! (Distanz: {distance:.0f}px)")
            except Exception as e:
                print(f"❌ Schiessschuss-Fehler: {e}")


def main():
    """Einstiegspunkt."""
    # Konfiguration - OPTIMIERT FÜR AMD RX 6750XT + 1920x1080
    config = Config(
        # Modell
        model_path="models/yolov8s.pt",
        conf_threshold=0.5,
        device="dml",  # ✅ AMD GPU (DirectML) - perfekt für RX 6750XT!
        half=False,
        
        # 🎯 CAPTURE REGION: NUR OBERER BEREICH (1920x1080 Auflösung)
        capture_region=(0, 0, 1920, 950),  # Keine Hand/Waffe unten
        target_fps=120,  # AMD GPU kann das locker
        
        # Target Selection
        selection_strategy="closest_to_center",
        preferred_class=0,  # 0 = Person
        class_filter=None,
        
        # Window Tracking
        track_window=False,
        window_title="",
        
        # Mouse Smoothing
        smoothing_alpha=0.2,
        use_relative_mouse=False,
        
        # Auto-Shoot Konfiguration
        auto_shoot=True,
        shoot_threshold_px=30,
        shoot_cooldown_ms=80.0,
        
        # Logging
        log_fps_interval=1.0,
        debug_mode=False,
        
        # Visualisierung
        show_preview=False,
        show_detections=False
    )
    
    # Stelle sicher, dass models Verzeichnis existiert
    Path("models").mkdir(exist_ok=True)
    
    # System starten
    system = TrackingSystem(config)
    system.start()


if __name__ == "__main__":
    main()
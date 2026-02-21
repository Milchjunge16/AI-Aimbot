"""
Zentrale Konfiguration für das AI Aimbot Tracking-System.
Alle einstellbaren Parameter an einem Ort für einfache Anpassung.

Beispiele:
    config = Config(
        model_path="models/yolov8s.pt",      # Bessere Genauigkeit aber langsamer
        conf_threshold=0.6,                   # Höhere Konfidenz = weniger Falsch-Positive
        auto_shoot=True,                      # Auto-Shooting aktiviert
        shoot_threshold_px=30,                # Nur innerhalb von 30px Schießen
    )
"""
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Config:
    # ==================== MODEL KONFIGURATION ====================
    model_path: str = "models/yolov8n.pt"
    """Pfad zum YOLO Modell. Optionen: yolov8n.pt (schnell), yolov8s.pt (ausgewogen), yolov8m.pt (genau)"""
    
    conf_threshold: float = 0.5
    """Confidence-Threshold (0.0-1.0). Höher = weniger Falsch-Positive, aber weniger Detektionen"""
    
    device: str = "dml"
    """GPU-Device: 'dml' (AMD), 'cuda' (NVIDIA), 'cpu' (Fallback)"""
    
    half: bool = False
    """Halbe Precision (16-bit) - nur für CUDA, schneller aber weniger genau"""
    
    # ==================== SCREEN CAPTURE ====================
    capture_region: Optional[Tuple[int, int, int, int]] = None
    """Aufnahme-Region als (left, top, right, bottom). None = gesamter Bildschirm"""
    
    target_fps: int = 30
    """Ziel-FPS für Screen Capture. Höher = bessere Reaktion aber mehr CPU"""
    
    # ==================== TARGET SELECTION ====================
    selection_strategy: str = "closest_to_center"
    """Zielselektion-Strategie: 'highest_confidence' oder 'closest_to_center'"""
    
    preferred_class: Optional[int] = 0
    """Bevorzugte Objekt-Klasse: 0=Person, 2=Auto, 5=Bus, None=alle Klassen"""
    
    class_filter: Optional[list] = None
    """Klassen-Filter (Whitelist). z.B. [0, 2] für nur Personen und Autos"""
    
    # ==================== WINDOW TRACKING ====================
    track_window: bool = False
    """Nur ein spezifisches Fenster tracken? (z.B. Spiel-Fenster)"""
    
    window_title: str = ""
    """Fenstertitel (oder Teil davon) zum Tracken. z.B. 'Counter-Strike' oder 'VALORANT'"""
    
    # ==================== MOUSE CONTROL ====================
    smoothing_alpha: float = 0.3
    """Smoothing-Faktor: 0.0=max glatt (träge), 1.0=keine Glättung (zitternd). Optimal: 0.2-0.4"""
    
    use_relative_mouse: bool = True
    """True=Relative Bewegung (Fadenkreuz bleibt gleich), False=Absolute Position"""
    
    # ==================== AUTO-SHOOT (Linke Maustaste) ====================
    auto_shoot: bool = True
    """Automatisch schießen wenn auf Ziel? (Linker Mausklick)"""
    
    shoot_threshold_px: int = 40
    """Max. Abstand vom Fadenkreuz-Mittelpunkt (Pixel) um zu schießen. Höher=größerer Toleranz-Bereich"""
    
    shoot_cooldown_ms: float = 100.0
    """Minimale Zeit zwischen Schüssen (Millisekunden). Verhindert zu schnelle Schüsse"""
    
    # ==================== PERFORMANCE & LOGGING ====================
    log_fps_interval: float = 1.0
    """FPS-Logging Intervall in Sekunden"""
    
    debug_mode: bool = False
    """Debug-Mode: Zusätzliche Console-Ausgaben und Metriken"""
    
    # ==================== VISUALISIERUNG ====================
    show_preview: bool = False
    """Vorschau-Fenster mit Detektionen anzeigen (langsamer)"""
    
    show_detections: bool = False
    """Erkannte Objekte mit Bounding Boxes einzeichnen"""
    
    # ==================== HOTKEYS ====================
    toggle_hotkey: str = "f6"
    """Hotkey zum Ein/Ausschalten des Trackings. z.B. 'f6', 'f8', 'space'"""
    # ==================== DISPLAY / SCREEN ====================
    display_resolution: Optional[Tuple[int, int]] = None
    """Optional: Manuell eingestellte Display-Auflösung als (width, height).
    None = System-Auflösung verwenden (Standard)."""
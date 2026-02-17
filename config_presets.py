"""
AI Aimbot - Configuration Examples
Vorgefertigte Konfigurationen für verschiedene Szenarien und Spiele
"""

from config import Config


# ==================== PRESET CONFIGS ====================

def get_config_speed_optimized() -> Config:
    """
    Maximale Geschwindigkeit
    Ideal für: Casual Gaming, alte Hardware, CPU-Only
    FPS: 60+ FPS
    """
    return Config(
        # Kleine, schnelle Modell
        model_path="models/yolov8n.pt",
        conf_threshold=0.4,
        
        # CPU reicht aus
        device="cpu",
        
        # Minimale Glättung
        smoothing_alpha=0.5,
        
        # Schnelle Schüsse
        auto_shoot=True,
        shoot_threshold_px=60,
        shoot_cooldown_ms=50,
        
        # Hohes FPS Target
        target_fps=120,
        
        # Kein Debug
        debug_mode=False,
        show_preview=False,
    )


def get_config_accuracy_optimized() -> Config:
    """
    Maximale Genauigkeit
    Ideal für: Competitive Gaming, High-End Hardware
    FPS: 30-50 FPS
    """
    return Config(
        # Großes, genaues Modell
        model_path="models/yolov8m.pt",
        conf_threshold=0.7,
        
        # GPU dringend empfohlen
        device="cuda",  # oder "dml" für AMD
        half=True,
        
        # Maximale Glättung
        smoothing_alpha=0.2,
        
        # Strenge Schuss-Kriterien
        auto_shoot=True,
        shoot_threshold_px=30,
        shoot_cooldown_ms=150,
        
        # Normal FPS
        target_fps=60,
        
        # Debug bei Bedarf
        debug_mode=False,
        show_detections=False,
    )


def get_config_balanced() -> Config:
    """
    Ausgewogene Konfiguration
    Ideal für: Allgemeiner Einsatz, die meisten Spiele
    FPS: 45-60 FPS (mit GPU)
    """
    return Config(
        model_path="models/yolov8s.pt",
        conf_threshold=0.5,
        device="dml",  # AMD
        
        smoothing_alpha=0.3,
        
        auto_shoot=True,
        shoot_threshold_px=40,
        shoot_cooldown_ms=100,
        
        target_fps=60,
        selection_strategy="closest_to_center",
        debug_mode=False,
    )


# ==================== SPIELE-SPEZIFISCHE CONFIGS ====================

def get_config_csgo() -> Config:
    """Counter-Strike 2 / CSGO"""
    return Config(
        model_path="models/yolov8s.pt",
        conf_threshold=0.6,
        device="dml",
        
        preferred_class=0,  # Nur Menschen
        smoothing_alpha=0.2,  # Glattes Tracking für Headshots
        
        auto_shoot=True,
        shoot_threshold_px=30,  # Enger Threshold für Headshots
        shoot_cooldown_ms=80,
        
        target_fps=120,
        debug_mode=False,
    )


def get_config_valorant() -> Config:
    """Valorant"""
    return Config(
        model_path="models/yolov8s.pt",
        conf_threshold=0.55,
        device="dml",
        
        preferred_class=0,
        smoothing_alpha=0.3,
        
        auto_shoot=True,
        shoot_threshold_px=35,
        shoot_cooldown_ms=60,  # Schneller (Pistol-Runden)
        
        target_fps=120,
        selection_strategy="closest_to_center",
    )


def get_config_apex() -> Config:
    """Apex Legends"""
    return Config(
        model_path="models/yolov8m.pt",
        conf_threshold=0.5,
        device="cuda",  # Bessere Performance für großes Modell
        
        preferred_class=0,
        smoothing_alpha=0.4,  # Schneller für schnelle Spiele
        
        auto_shoot=True,
        shoot_threshold_px=50,  # Mehr Toleranz
        shoot_cooldown_ms=120,  # Langsamere, präzisere Schüsse
        
        target_fps=60,
        debug_mode=False,
    )


def get_config_fortnite() -> Config:
    """Fortnite"""
    return Config(
        model_path="models/yolov8n.pt",
        conf_threshold=0.45,
        device="dml",
        
        preferred_class=0,
        smoothing_alpha=0.35,
        
        auto_shoot=True,
        shoot_threshold_px=55,  # Größeres Ziel-Fenster
        shoot_cooldown_ms=100,
        
        target_fps=120,
    )


def get_config_overwatch() -> Config:
    """Overwatch 2"""
    return Config(
        model_path="models/yolov8s.pt",
        conf_threshold=0.52,
        device="cuda",
        
        preferred_class=0,
        smoothing_alpha=0.25,
        
        auto_shoot=True,
        shoot_threshold_px=45,
        shoot_cooldown_ms=110,
        
        target_fps=100,
    )


def get_config_debug() -> Config:
    """Debug/Testing Konfiguration"""
    return Config(
        model_path="models/yolov8n.pt",
        conf_threshold=0.5,
        device="cpu",
        
        auto_shoot=False,  # Nicht schießen im Debug-Mode!
        
        target_fps=30,
        debug_mode=True,  # Alle Debug-Ausgaben
        show_preview=True,  # Zeige Vorschau
        show_detections=True,  # Zeige Bounding Boxes
        log_fps_interval=0.5,  # Häufigere FPS-Logs
    )


# ==================== GPU-SPEZIFISCHE CONFIGS ====================

def get_config_nvidia_gpu() -> Config:
    """Optimiert für NVIDIA GPUs (CUDA)"""
    return Config(
        model_path="models/yolov8s.pt",
        conf_threshold=0.5,
        device="cuda",
        half=True,  # FP16 für NVIDIA
        
        smoothing_alpha=0.3,
        auto_shoot=True,
        shoot_threshold_px=40,
        shoot_cooldown_ms=100,
        target_fps=120,
    )


def get_config_amd_gpu() -> Config:
    """Optimiert für AMD GPUs (DirectML)"""
    return Config(
        model_path="models/yolov8s.pt",
        conf_threshold=0.5,
        device="dml",
        half=False,  # DirectML mit half precision kann Probleme haben
        
        smoothing_alpha=0.3,
        auto_shoot=True,
        shoot_threshold_px=40,
        shoot_cooldown_ms=100,
        target_fps=60,
    )


def get_config_cpu_only() -> Config:
    """CPU-Only Konfiguration (keine GPU)"""
    return Config(
        model_path="models/yolov8n.pt",  # Kleinste Modell!
        conf_threshold=0.4,
        device="cpu",
        
        smoothing_alpha=0.4,
        auto_shoot=True,
        shoot_threshold_px=50,
        shoot_cooldown_ms=150,  # Mehr Cooldown wegen Latenz
        
        target_fps=30,  # 30 FPS ist realistisch auf CPU
    )


# ==================== FENSTER-TRACKING CONFIGS ====================

def get_config_window_csgo() -> Config:
    """CS:GO mit Fenster-Tracking"""
    return Config(
        model_path="models/yolov8s.pt",
        conf_threshold=0.6,
        device="dml",
        
        track_window=True,
        window_title="Counter-Strike",
        
        preferred_class=0,
        smoothing_alpha=0.2,
        
        auto_shoot=True,
        shoot_threshold_px=30,
        shoot_cooldown_ms=80,
    )


def get_config_window_valorant() -> Config:
    """Valorant mit Fenster-Tracking"""
    return Config(
        model_path="models/yolov8s.pt",
        conf_threshold=0.55,
        device="dml",
        
        track_window=True,
        window_title="VALORANT",
        
        preferred_class=0,
        smoothing_alpha=0.3,
        
        auto_shoot=True,
        shoot_threshold_px=35,
        shoot_cooldown_ms=60,
    )


# ==================== HELPER FUNCTION ====================

def get_config_by_name(name: str) -> Config:
    """
    Gibt eine vorgefertigte Konfiguration anhand des Namens zurück.
    
    Verfügbare Optionen:
    - 'speed': Maximale Geschwindigkeit
    - 'accuracy': Maximale Genauigkeit  
    - 'balanced': Ausgewogene Einstellung
    - 'csgo': Counter-Strike optimiert
    - 'valorant': Valorant optimiert
    - 'apex': Apex Legends optimiert
    - 'fortnite': Fortnite optimiert
    - 'overwatch': Overwatch 2 optimiert
    - 'debug': Debug-Modus
    - 'nvidia': NVIDIA GPU optimiert
    - 'amd': AMD GPU optimiert
    - 'cpu': CPU-Only (keine GPU)
    """
    configs = {
        'speed': get_config_speed_optimized,
        'accuracy': get_config_accuracy_optimized,
        'balanced': get_config_balanced,
        'csgo': get_config_csgo,
        'valorant': get_config_valorant,
        'apex': get_config_apex,
        'fortnite': get_config_fortnite,
        'overwatch': get_config_overwatch,
        'debug': get_config_debug,
        'nvidia': get_config_nvidia_gpu,
        'amd': get_config_amd_gpu,
        'cpu': get_config_cpu_only,
        'window_csgo': get_config_window_csgo,
        'window_valorant': get_config_window_valorant,
    }
    
    if name.lower() not in configs:
        print(f"Unbekannte Konfiguration: {name}")
        print(f"Verfügbar: {', '.join(configs.keys())}")
        return get_config_balanced()
    
    return configs[name.lower()]()


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    """
    Beispiel: Lade eine vorgefertigte Konfiguration
    """
    # Option 1: Direkte Funktion
    config = get_config_csgo()
    print(f"✅ CS:GO Konfiguration geladen")
    
    # Option 2: Über Namen
    config = get_config_by_name("valorant")
    print(f"✅ Valorant Konfiguration geladen")
    
    # Option 3: Mit deinen Änderungen
    config = get_config_balanced()
    config.smoothing_alpha = 0.4  # Anpassung
    print(f"✅ Balanced Konfiguration mit Änderung geladen")

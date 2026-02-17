"""
Maus-Steuerung Modul - Bewegt die Maus sanft zum Ziel.
Verwendet exponentielles Smoothing für flüssige Bewegung.
"""
import ctypes
import time
from typing import Optional, Tuple

from config import Config

# Region = (left, top, right, bottom) in screen coordinates
Region = Optional[Tuple[int, int, int, int]]


class MouseController:
    """Steuert die Maus mit Smoothing für natürliche Bewegung."""
    
    def __init__(self, config: Config):
        self.alpha = config.smoothing_alpha
        self.smoothed_x: Optional[float] = None
        self.smoothed_y: Optional[float] = None
        
        # Für Bewegungseinschränkungen
        self.screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        
        print(f"🖱️ Maus-Smoothing: {self.alpha} (0 = glatt, 1 = direkt)")

    @staticmethod
    def region_to_screen(x: float, y: float, region: Region) -> Tuple[float, float]:
        """
        Konvertiert Koordinaten aus der Capture-Region in Bildschirmkoordinaten.
        Wenn region None ist (ganzer Bildschirm), werden x,y unverändert zurückgegeben.
        """
        if region is None:
            return (x, y)
        left, top, _, _ = region
        return (left + x, top + y)

    def update(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """
        Wendet exponentielles Smoothing auf Zielposition an.
        
        Args:
            target_x, target_y: Absolute Zielkoordinaten
            
        Returns:
            Geglättete Koordinaten
        """
        if self.smoothed_x is None:
            self.smoothed_x = target_x
            self.smoothed_y = target_y
        else:
            self.smoothed_x = self.alpha * target_x + (1 - self.alpha) * self.smoothed_x
            self.smoothed_y = self.alpha * target_y + (1 - self.alpha) * self.smoothed_y
            
        return (self.smoothed_x, self.smoothed_y)
        
    def move_to(self, x: float, y: float):
        """Bewegt Maus zu absoluten Bildschirmkoordinaten (SetCursorPos)."""
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
    
    @staticmethod
    def move_relative(dx: float, dy: float):
        """
        Bewegt Maus relativ (Delta) – für FPS-Spiele mit Fadenkreuz in der Mitte.
        Das Spiel dreht die Sicht, das Fadenkreuz bleibt zentriert.
        """
        MOUSEEVENTF_MOVE = 0x0001
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
    
    @staticmethod
    def click_left():
        """Simuliert linken Mausklick (Down + Up)."""
        MOUSEEVENTF_LEFTDOWN = 0x0002
        MOUSEEVENTF_LEFTUP = 0x0004
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        
    def reset_smoothing(self):
        """Setzt den Smoothing-Puffer zurück."""
        self.smoothed_x = None
        self.smoothed_y = None
"""
Target Selektion Modul - Wählt das beste Ziel aus allen Detektionen aus.
"""
from typing import List, Optional, Tuple

from config import Config


class TargetSelector:
    """Wählt das optimale Ziel basierend auf Konfiguration."""
    
    def __init__(self, config: Config, screen_width: int, screen_height: int):
        self.config = config
        self.screen_center = (screen_width / 2, screen_height / 2)
        print(f"🎯 Bildschirm-Mitte: ({self.screen_center[0]:.0f}, {self.screen_center[1]:.0f})")
        
        # Cache für letztes Ziel (für Stabilität)
        self.last_target: Optional[Tuple[float, float]] = None
        self.last_target_time = 0
        
    def select(self, detections: List) -> Optional[Tuple[float, float]]:
        """
        Wählt das beste Ziel aus und gibt dessen Mittelpunkt zurück.
        
        Strategien:
        - 'highest_confidence': Objekt mit höchster Konfidenz
        - 'closest_to_center': Objekt am nächsten zur Bildschirm-Mitte
        """
        if not detections:
            self.last_target = None
            return None
            
        if self.config.selection_strategy == "highest_confidence":
            target = self._select_highest_confidence(detections)
        elif self.config.selection_strategy == "closest_to_center":
            target = self._select_closest_to_center(detections)
        else:
            raise ValueError(f"Unbekannte Strategie: {self.config.selection_strategy}")
            
        if target:
            self.last_target = target
            
        return target
        
    def _select_highest_confidence(self, detections: List) -> Optional[Tuple[float, float]]:
        """Wählt Detektion mit höchster Konfidenz."""
        best = max(detections, key=lambda d: d[1])
        bbox = best[0]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        
        if self.config.debug_mode:
            print(f"📊 Höchste Konfidenz: Klasse {best[2]} ({best[1]:.2f})")
            
        return (cx, cy)
        
    def _select_closest_to_center(self, detections: List) -> Optional[Tuple[float, float]]:
        """Wählt Detektion am nächsten zur Bildschirm-Mitte."""
        best = None
        best_dist = float("inf")
        
        for bbox, conf, cls_id in detections:
            # Klassen-Filter
            if self.config.preferred_class is not None and cls_id != self.config.preferred_class:
                continue
                
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            
            # Quadratische Distanz (keine Wurzel für Performance)
            dist = (cx - self.screen_center[0])**2 + (cy - self.screen_center[1])**2
            
            if dist < best_dist:
                best_dist = dist
                best = (cx, cy)
                
        if best:
            if self.config.debug_mode:
                print(f"📊 Nächstes Ziel: Distanz {best_dist**0.5:.1f}px")
            return best
            
        # Fallback: Höchste Konfidenz wenn keine passende Klasse
        return self._select_highest_confidence(detections)
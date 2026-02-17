"""
FPS Counter für Performance-Monitoring.
"""
import time


class FPSCounter:
    """Zählt Frames pro Sekunde für verschiedene Pipeline-Stufen."""
    
    def __init__(self, name: str, log_interval: float = 1.0):
        self.name = name
        self.log_interval = log_interval
        self.count = 0
        self.last_time = time.perf_counter()
        
    def tick(self) -> None:
        """Zählt einen Frame und loggt bei Intervall."""
        self.count += 1
        now = time.perf_counter()
        
        if now - self.last_time >= self.log_interval:
            fps = self.count / (now - self.last_time)
            print(f"📊 {self.name}: {fps:.1f} FPS")
            self.count = 0
            self.last_time = now
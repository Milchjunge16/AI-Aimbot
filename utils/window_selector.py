"""
Fenster-Selektion Modul - Ermöglicht Tracking eines bestimmten Fensters.
"""
import time
from typing import Optional, Tuple

import win32gui
import win32con


class WindowSelector:
    """Hilfsklasse zum Finden und Tracken von Windows-Fenstern."""
    
    def __init__(self):
        self.target_window = None
        self.window_rect = None
        self.window_title = ""
        
    def list_windows(self) -> list:
        """Listet alle sichtbaren Fenster mit Titel auf und zeigt sie im Terminal."""
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if window_text:
                    windows.append((hwnd, window_text))
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        # Zeige Fenster im Terminal
        if windows:
            print("\n" + "="*80)
            print("VERFUEGBARE FENSTER".center(80))
            print("="*80)
            for idx, (hwnd, title) in enumerate(windows, 1):
                print(f"{idx:3d}. {title}")
            print("="*80 + "\n")
        
        return windows
    
    def select_window(self, title_substring: str) -> bool:
        """
        Wählt erstes Fenster dessen Titel den Substring enthält.
        
        Returns:
            True wenn Fenster gefunden, sonst False
        """
        windows = self.list_windows()
        for hwnd, title in windows:
            if title_substring.lower() in title.lower():
                self.target_window = hwnd
                self.window_title = title
                self.update_rect()
                print(f"[WINDOW] Fenster ausgewaehlt: {title}")
                return True
                
        print(f"[ERROR] Kein Fenster mit '{title_substring}' gefunden")
        return False
    
    def update_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Aktualisiert die Fenster-Position."""
        if self.target_window:
            try:
                rect = win32gui.GetWindowRect(self.target_window)
                self.window_rect = rect  # (left, top, right, bottom)
                return rect
            except:
                # Fenster könnte geschlossen worden sein
                self.target_window = None
                self.window_rect = None
        return None
    
    def get_capture_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Gibt Region für Screen Capture zurück und aktualisiert Position."""
        # Aktualisiere Position für den Fall, dass das Fenster bewegt wurde
        if self.update_rect():
            return self.window_rect
        return None
    
    def get_windows_dict(self) -> dict:
        """Gibt Dictionary mit Fenstern für UI zurück."""
        windows = self.list_windows()
        windows_dict = {}
        for hwnd, title in windows:
            windows_dict[title] = hwnd
        return windows_dict
"""
GUI Starter für AI Aimbot.
Startet die grafische Benutzeroberfläche.
"""
import sys
from pathlib import Path

# Stelle sicher, dass wir im richtigen Verzeichnis sind
sys.path.insert(0, str(Path(__file__).parent))

from gui.main_window import main

if __name__ == "__main__":
    main()
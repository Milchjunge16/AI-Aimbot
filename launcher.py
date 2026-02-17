#!/usr/bin/env python3
"""
AI Aimbot Launcher - Vereinfachter Einstiegspunkt
Ermöglicht Auswahl zwischen CLI und GUI
"""
import sys
import os
from pathlib import Path

# Stelle sicher, dass wir im richtigen Verzeichnis sind
sys.path.insert(0, str(Path(__file__).parent))


def print_header():
    """Druckt das Programm-Banner."""
    print("\n" + "="*70)
    print("    🎯 AI AIMBOT TRACKING SYSTEM - LAUNCHER 🎯".center(70))
    print("="*70)
    print()


def print_menu():
    """Zeigt das Hauptmenü."""
    print("Wähle einen Modus:")
    print()
    print("  1️⃣  CLI Modus (Kommandozeile)")
    print("      - Einfach, schnell")
    print("      - Hotkey F6 zum Toggle")
    print()
    print("  2️⃣  GUI Modus (Grafische Oberfläche)")
    print("      - Vollständige Kontrolle")
    print("      - Echtzeit-Vorschau")
    print()
    print("  3️⃣  Fenster-Auswahl (List Selector)")
    print("      - Wähle ein bestimmtes Fenster")
    print()
    print("  0️⃣  Beenden")
    print()


def check_requirements():
    """Prüft ob alle notwendigen Pakete installiert sind."""
    print("🔍 Überprüfe Anforderungen...")
    
    missing = []
    required = {
        'torch': 'PyTorch',
        'ultralytics': 'YOLOv8',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'dxcam': 'DXCAM (Screen Capture)',
        'pynput': 'Pynput (Hotkeys)',
    }
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Fehlende Pakete: {', '.join(missing)}")
        print("\nInstalliere mit:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✅ Alle Anforderungen erfüllt!\n")
    return True


def run_cli_mode():
    """Startet den CLI-Modus."""
    print("\n" + "="*70)
    print("CLI MODUS GESTARTET".center(70))
    print("="*70)
    print()
    print("Tipps:")
    print("  🎮 Drücke F6 um Tracking ein/auszuschalten")
    print("  🛑 Drücke Ctrl+C um zu beenden")
    print()
    print("-"*70)
    print()
    
    try:
        from main import main
        main()
    except Exception as e:
        print(f"\n❌ Fehler im CLI-Modus: {e}")
        import traceback
        traceback.print_exc()


def run_gui_mode():
    """Startet den GUI-Modus."""
    print("\n" + "="*70)
    print("GUI MODUS GESTARTET".center(70))
    print("="*70)
    print()
    
    try:
        from gui_main import main
        main()
    except Exception as e:
        print(f"\n❌ Fehler im GUI-Modus: {e}")
        import traceback
        traceback.print_exc()


def run_window_selector():
    """Listet Fenster auf und lässt Benutzer eines auswählen."""
    print("\n" + "="*70)
    print("FENSTER-SELECTOR".center(70))
    print("="*70)
    print()
    
    try:
        from utils.window_selector import WindowSelector
        selector = WindowSelector()
        selector.list_windows()
        
        print("\nWende Window Tracking an? (y/n) [n]: ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            print("\nGib Fenster-Titel (oder Teil davon) ein: ", end="")
            window_title = input().strip()
            
            if window_title:
                # Starte mit Window Tracking aktiviert
                from main import TrackingSystem
                from config import Config
                
                config = Config(
                    track_window=True,
                    window_title=window_title,
                    auto_shoot=True,
                    shoot_threshold_px=50,
                    shoot_cooldown_ms=100,
                )
                
                from pathlib import Path
                Path("models").mkdir(exist_ok=True)
                
                print("\n" + "="*70)
                print("STARTE MIT WINDOW TRACKING".center(70))
                print("="*70)
                print()
                print(f"Ziel-Fenster: {window_title}")
                print(f"Hotkey: F6 (Toggle)")
                print(f"Ctrl+C zum Beenden")
                print()
                
                system = TrackingSystem(config)
                system.start()
        else:
            print("Abgebrochen.")
            
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()


def main_menu():
    """Hauptmenü-Loop."""
    while True:
        print_menu()
        
        try:
            choice = input("Deine Wahl (0-3): ").strip()
            
            if choice == "1":
                run_cli_mode()
                break
            elif choice == "2":
                run_gui_mode()
                break
            elif choice == "3":
                run_window_selector()
                break
            elif choice == "0":
                print("\n👋 Auf Wiedersehen!")
                break
            else:
                print(f"\n❌ Ungültige Wahl: {choice}\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Auf Wiedersehen!")
            break
        except Exception as e:
            print(f"\n❌ Fehler: {e}\n")


def main():
    """Einstiegspunkt."""
    print_header()
    
    # Überprüfe Anforderungen
    if not check_requirements():
        print("❌ Bitte installiere zuerst alle Anforderungen:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Zeige Hauptmenü
    main_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Auf Wiedersehen!")
    except Exception as e:
        print(f"\n❌ Fataler Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

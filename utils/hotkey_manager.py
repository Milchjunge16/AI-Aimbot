"""
Hotkey Manager - Verwaltet Hotkeys für Ein/Aus-Schalten des Aimbots.
Verwendet pynput für plattformübergreifende Hotkey-Unterstützung.
"""
import threading
from typing import Callable, Optional

try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    print("⚠️ pynput nicht installiert! Hotkeys funktionieren nicht.")


class HotkeyManager:
    """Verwaltet Hotkeys für das Tracking-System."""
    
    # Mapping von Tastennamen zu pynput Key-Objekten
    KEY_MAP = {
        'f1': keyboard.Key.f1,
        'f2': keyboard.Key.f2,
        'f3': keyboard.Key.f3,
        'f4': keyboard.Key.f4,
        'f5': keyboard.Key.f5,
        'f6': keyboard.Key.f6,
        'f7': keyboard.Key.f7,
        'f8': keyboard.Key.f8,
        'f9': keyboard.Key.f9,
        'f10': keyboard.Key.f10,
        'f11': keyboard.Key.f11,
        'f12': keyboard.Key.f12,
        'esc': keyboard.Key.esc,
        'tab': keyboard.Key.tab,
        'space': keyboard.Key.space,
    }
    
    def __init__(self):
        self.listener = None
        self.callbacks = {}
        self.running = False
        
    def register_hotkey(self, key_name: str, callback: Callable) -> bool:
        """
        Registriert einen Hotkey.
        
        Args:
            key_name: Tastenname (z.B. 'f6', 'f5', etc.)
            callback: Funktion, die aufgerufen wird wenn Taste gedrückt wird
            
        Returns:
            True wenn erfolgreich registriert
        """
        if not HAS_PYNPUT:
            print("⚠️ pynput nicht installiert!")
            return False
            
        key_name = key_name.lower().strip('<>')
        if key_name not in self.KEY_MAP:
            print(f"❌ Unbekannte Taste: {key_name}")
            return False
            
        self.callbacks[key_name] = callback
        print(f"✅ Hotkey registriert: {key_name.upper()}")
        return True
    
    def start(self):
        """Startet den Hotkey-Listener."""
        if not HAS_PYNPUT:
            print("⚠️ pynput nicht installiert!")
            return
            
        if self.running:
            print("⚠️ Hotkey-Listener läuft bereits")
            return
            
        self.running = True
        self.listener = keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()
        print("🎮 Hotkey-Listener gestartet")
    
    def stop(self):
        """Stoppt den Hotkey-Listener."""
        self.running = False
        if self.listener:
            self.listener.stop()
            print("🎮 Hotkey-Listener gestoppt")
    
    def _on_key_press(self, key):
        """Interne Callback für Tastendruck."""
        try:
            # Versuche den Tastennamen zu bestimmen
            if hasattr(key, 'name'):
                key_name = key.name.lower()
            else:
                key_name = str(key).lower()
                
            # Rufe Callback auf wenn registriert
            if key_name in self.callbacks:
                callback = self.callbacks[key_name]
                # Callback in separatem Thread ausführen um Blockierung zu vermeiden
                thread = threading.Thread(target=callback, daemon=True)
                thread.start()
                
        except Exception as e:
            print(f"❌ Fehler im Hotkey-Handler: {e}")


# Singleton-Instanz
_hotkey_manager = None

def get_hotkey_manager() -> HotkeyManager:
    """Gibt eine Singleton-Instanz des HotkeyManagers zurück."""
    global _hotkey_manager
    if _hotkey_manager is None:
        _hotkey_manager = HotkeyManager()
    return _hotkey_manager

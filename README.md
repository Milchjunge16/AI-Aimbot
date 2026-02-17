# 🎯 AI Aimbot - Counter Strike Source

KI-gestütztes Auto-Aiming System mit YOLO v8 Objekterkennung für FPS Games.

## 🚀 Schneller Start

```bash
# 1. Dependencies installieren (einmalig)
pip install -r requirements.txt

# 2. Programm starten
python launcher.py

# 3. Hotkey zum Aktivieren/Deaktivieren
F6 = Toggle Tracking
```

## 📁 Ordnerstruktur

```
.
├── launcher.py                 # 🎮 Start-Menü (HIER STARTEN!)
├── main.py                     # 💻 CLI Hauptprogramm
├── gui_main.py                 # 🎨 GUI Alternative
├── config.py                   # ⚙️ Konfigurationsoptionen
├── config_presets.py           # 📋 Vorgefertigte Konfigurationen
├── requirements.txt            # 📦 Python Dependencies
│
├── core/                       # 🧠 Kern-Module
│   ├── capture.py              # 📸 Screen Capture (dxcam)
│   ├── detector.py             # 🤖 YOLO Objekterkennung
│   ├── selector.py             # 🎯 Ziel-Auswahl-Logik
│   └── mouse_controller.py     # 🖱️ Maus-Steuerung & Smoothing
│
├── gui/                        # 🎨 Grafische Oberfläche
│   └── main_window.py          # PyQt5 GUI
│
├── utils/                      # 🛠️ Hilfsfunktionen
│   ├── hotkey_manager.py       # ⌨️ Hotkey-System (F6)
│   ├── fps_counter.py          # 📊 Performance Monitoring
│   └── window_selector.py      # 🪟 Fenster-Selektion
│
├── models/                     # 🤖 KI-Modelle (YOLO)
│   ├── yolov8n.pt              # Schnell (60+ FPS)
│   ├── yolov8s.pt              # Ausgewogen ← AKTUELL GENUTZT
│   └── yolov8m.pt              # Genau (bis 30 FPS)
│
└── README.md                   # Diese Datei
```

---

## 📄 Was macht jede Datei?

### **Hauptprogramme**
- **launcher.py** - Menü zum Starten von CLI, GUI oder Window-Selector
- **main.py** - Kernprogramm (TrackingSystem Klasse + Konfiguration)
- **gui_main.py** - Grafische Oberfläche mit Live-Vorschau

### **Konfiguration**
- **config.py** - Alle einstellbaren Parameter (Model, GPU, Auto-Shoot, etc.)
- **config_presets.py** - Vorgefertigte Configs (CSGO, Valorant, Balanced, etc.)
- **requirements.txt** - Python Pakete (torch, ultralytics, opencv, etc.)

### **Core Module (die Magie)**
- **capture.py** - Macht Screenshots (60-120 FPS möglich)
- **detector.py** - Erkennt Personen mit YOLO (KI-Modell)
- **selector.py** - Wählt das beste Ziel (nächstes zum Fadenkreuz)
- **mouse_controller.py** - Bewegt Maus glatt + Auto-Click

### **Utilities**
- **hotkey_manager.py** - F6 Toggle Listener
- **fps_counter.py** - Zeigt Performance (FPS) in Console
- **window_selector.py** - Fenster-Liste zum Tracking

---

## ⚙️ Konfiguration

Alle Einstellungen in **config.py** (main.py → main() Funktion):

```python
# GPU (WICHTIG!)
device="dml"                    # AMD RX 6750XT → "dml"
                                # NVIDIA → "cuda"
                                # CPU → "cpu"

# Modell
model_path="models/yolov8s.pt"  # n=schnell, s=ausgewogen, m=genau

# Region (1920x1080 optimiert)
capture_region=(0, 0, 1920, 950)  # Nur oberer Bereich (keine Waffe)

# Tracking
conf_threshold=0.5              # 0.0-1.0 (höher=genauer)
smoothing_alpha=0.2             # 0.0=glatt, 1.0=zitternd

# Auto-Shoot
auto_shoot=True
shoot_threshold_px=30           # Nur schießen wenn nah genug
shoot_cooldown_ms=80.0          # Minimale Zeit zwischen Schüssen
```

---

## 🎮 Verwendung

### **Option 1: CLI (Standard)**
```bash
python launcher.py
# Wähle: 1 (CLI Mode)
```
- Schnell und leicht
- F6 zum Aktivieren
- Ctrl+C zum Beenden

### **Option 2: GUI**
```bash
python launcher.py
# Wähle: 2 (GUI Mode)
```
- Vorschau + Live-Stats
- Alle Parameter änderbar
- Echtzeit-Detektionen sehen

### **Option 3: Window-Selector**
```bash
python launcher.py
# Wähle: 3 (Window Selector)
```
- Fenster-Tracking
- Nur Spiel-Fenster tracken

---

## 🔫 Auto-Shoot Einstellung

**Zu aggressiv?** (schießt überall)
```python
shoot_threshold_px=20   # War 30 (kleiner = seltener)
conf_threshold=0.6      # War 0.5 (mehr Filter)
```

**Zu schwach?** (schießt nicht genug)
```python
shoot_threshold_px=40   # War 30 (größer = öfter)
conf_threshold=0.4      # War 0.5 (weniger Filter)
```

---

## 📊 FPS Erwartung

Mit **RX 6750XT + yolov8s + 1920x1080**:
- **Capture FPS:** 120 FPS
- **Inference FPS:** 80-120 FPS
- **Total System:** 80+ FPS realistisch

---

## 🔧 Häufige Probleme

**Problem: "Module nicht gefunden"**
```bash
pip install -r requirements.txt
```

**Problem: "GPU wird nicht erkannt"**
```python
# In config.py: device="cpu"
```

**Problem: "Hotkey funktioniert nicht"**
- Starte als Administrator
- Oder nutze Ctrl+C zum Beenden

**Problem: "Schießt überall hin"**
→ Siehe "Auto-Shoot Einstellung" oben

---

## 🎯 Performance-Tipps

**Wenn zu langsam:**
- Modell kleiner: `yolov8n.pt`
- FPS senken: `target_fps=60`
- Region kleiner (Teil des Screens)

**Wenn Latenz hoch:**
- GPU-Modus prüfen: `device="dml"`
- Smoothing erhöhen: `smoothing_alpha=0.3`

---

## 💡 Hotkeys

| Taste | Funktion |
|-------|----------|
| F6 | Tracking an/aus |
| Ctrl+C | Programm beenden |

---

## 📦 Installation

```bash
# Abhängigkeiten einmalig installieren
pip install -r requirements.txt

# YOLO Model wird beim ersten Start automatisch heruntergeladen
# Falls nicht, lade manuell mit:
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

---

## ⚠️ Wichtig

- **Nur für Bildungszwecke & Tests nutzen**
- Nicht in kompetitiven Online-Spielen (VAC-Ban Risiko)
- Lokale Gesetze / Nutzungsbedingungen beachten
- F6 zum schnellen Deaktivieren

---

**Viel Erfolg! 🎯**

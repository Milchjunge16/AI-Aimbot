"""
GUI Hauptfenster für das AI Aimbot Tracking-System.
Ermöglicht einfache Konfiguration und Steuerung des Systems.
"""
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from pynput import keyboard

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QTextEdit, QSlider, QTabWidget,
    QFileDialog, QMessageBox, QProgressBar, QGridLayout, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPalette, QColor

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap

# Füge Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from core.capture import ScreenCapture
from core.detector import ObjectDetector
from core.selector import TargetSelector
from core.mouse_controller import MouseController
from utils.fps_counter import FPSCounter
from utils.window_selector import WindowSelector


class InferenceThread(QThread):
    """Separater Thread für die Bildverarbeitung."""
    frame_ready = pyqtSignal(object)  # Für Vorschaubild
    detection_update = pyqtSignal(str)  # Für Detektions-Info
    detection_count_changed = pyqtSignal(int)  # Anzahl erkannter Objekte
    fps_update = pyqtSignal(float, float, float)  # capture, inference, total
    tracker_state_changed = pyqtSignal(bool)  # Für Tracker Active/Inactive
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = False
        self.tracking_system = None
        self.tracking_active = True  # Hotkey schaltet dies um
        self.last_shot_time = 0.0  # Für Schuss-Cooldown
        self.window_selector = None  # Optional für Fenster-Tracking
        
    def setup(self, tracking_system, window_selector=None):
        """Tracking-System setzen."""
        self.tracking_system = tracking_system
        self.window_selector = window_selector
        
    def toggle_tracking(self):
        """Hotkey: Tracker aktivieren/deaktivieren."""
        self.tracking_active = not self.tracking_active
        self.tracker_state_changed.emit(self.tracking_active)
        
    def run(self):
        """Hauptschleife für Tracking."""
        self.running = True
        frame_count = 0
        
        while self.running:
            # Aktualisiere Fenster-Position wenn Fenster tracked wird
            if self.window_selector and self.window_selector.target_window:
                capture_region = self.window_selector.get_capture_region()
                if capture_region and self.tracking_system:
                    self.tracking_system.capture.region = capture_region
            
            if self.tracking_system:
                frame = self.tracking_system.capture.get_frame()
                if frame is not None:
                    detections = []
                    # Nur detektieren und Maus bewegen wenn Tracker aktiv
                    if self.tracking_active:
                        # Detektion durchführen
                        detections = self.tracking_system.detector.detect(frame)
                        
                        # Ziel auswählen
                        target = self.tracking_system.selector.select(detections)
                        
                        # Fadenkreuz/Ziel bewegen (relativ für FPS oder absolut für Mauszeiger)
                        if target is not None:
                            tx, ty = target
                            screen_x, screen_y = MouseController.region_to_screen(
                                tx, ty, self.tracking_system.capture.region
                            )
                            region = self.tracking_system.capture.region
                            if region:
                                cx = (region[0] + region[2]) / 2
                                cy = (region[1] + region[3]) / 2
                            else:
                                m = self.tracking_system.mouse
                                cx, cy = m.screen_width / 2, m.screen_height / 2
                            use_rel = getattr(self.config, 'use_relative_mouse', True)
                            if use_rel:
                                dx = screen_x - cx
                                dy = screen_y - cy
                                alpha = self.config.smoothing_alpha
                                self.tracking_system.mouse.move_relative(dx * alpha, dy * alpha)
                            else:
                                sx, sy = self.tracking_system.mouse.update(screen_x, screen_y)
                                self.tracking_system.mouse.move_to(sx, sy)
                            
                            # Auto-Schießen: wenn Ziel nah am Fadenkreuz (Mitte)
                            if getattr(self.config, 'auto_shoot', False):
                                dist = ((screen_x - cx) ** 2 + (screen_y - cy) ** 2) ** 0.5
                                cooldown_sec = getattr(self.config, 'shoot_cooldown_ms', 100) / 1000.0
                                threshold = getattr(self.config, 'shoot_threshold_px', 40)
                                now = time.time()
                                if dist < threshold and (now - self.last_shot_time) >= cooldown_sec:
                                    self.tracking_system.mouse.click_left()
                                    self.last_shot_time = now
                        
                        self.detection_count_changed.emit(len(detections))
                    else:
                        self.detection_count_changed.emit(0)
                    
                    # Vorschaubild mit Detektionen (nur Boxen wenn aktiv)
                    if hasattr(self, 'show_preview') and self.show_preview:
                        preview = frame.copy()
                        for bbox, conf, cls_id in detections:
                            x1, y1, x2, y2 = map(int, bbox)
                            # Bounding Box zeichnen
                            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Label zeichnen
                            class_name = ObjectDetector.COCO_CLASSES.get(cls_id, f"ID:{cls_id}")
                            label = f"{class_name} {conf:.2f}"
                            cv2.putText(preview, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Konvertiere für Qt
                        self.frame_ready.emit(preview)
                    
                    # FPS Updates
                    # TODO: FPS von TrackingSystem holen
                    
            time.sleep(0.001)
    
    def stop(self):
        """Thread stoppen."""
        self.running = False


class MainWindow(QMainWindow):
    """Hauptfenster der GUI."""
    
    def __init__(self):
        super().__init__()
        self.tracking_system = None
        self.inference_thread = None
        self.config = self.load_config()
        self.hotkey_listener = None
        self._current_hotkey_display = "F6"  # Anzeige für Status-Label
        self.window_selector = WindowSelector()  # Fenster-Selektor
        self.target_window_hwnd = None  # Speichert das ausgewählte Fenster
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Benutzeroberfläche initialisieren."""
        self.setWindowTitle("AI Aimbot Tracking System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Zentrale Widgets
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Hauptlayout
        main_layout = QHBoxLayout(central_widget)
        
        # Linke Seite: Vorschau
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Vorschau-Label
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("border: 2px solid #444; background-color: #222;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("Vorschau gestoppt")
        left_layout.addWidget(self.preview_label)
        
        # FPS Anzeige
        fps_layout = QHBoxLayout()
        self.capture_fps_label = QLabel("Capture: 0 FPS")
        self.inference_fps_label = QLabel("Inference: 0 FPS")
        self.total_fps_label = QLabel("Total: 0 FPS")
        for label in [self.capture_fps_label, self.inference_fps_label, self.total_fps_label]:
            label.setStyleSheet("color: #0f0; font-weight: bold;")
            fps_layout.addWidget(label)
        left_layout.addLayout(fps_layout)
        
        # Rechte Seite: Steuerung
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(20)
        
        # Status-Anzeige
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("System bereit")
        self.status_label.setStyleSheet("color: #0a0; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        # Tracker-Status (Hotkey umschaltbar, nur sichtbar wenn Tracking läuft)
        self.tracker_status_label = QLabel("Tracker: ● Active (Hotkey zum Umschalten)")
        self.tracker_status_label.setStyleSheet("color: #0f0; font-weight: bold;")
        self.tracker_status_label.setVisible(False)  # Erst sichtbar nach Start
        status_layout.addWidget(self.tracker_status_label)
        
        # Erkannte Objekte (live während Tracking)
        self.detection_count_label = QLabel("Erkannte Objekte: 0")
        self.detection_count_label.setStyleSheet("color: #aaa;")
        self.detection_count_label.setVisible(False)
        status_layout.addWidget(self.detection_count_label)
        
        self.detection_info = QTextEdit()
        self.detection_info.setMaximumHeight(100)
        self.detection_info.setReadOnly(True)
        status_layout.addWidget(self.detection_info)
        
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        
        # Tabs für Konfiguration
        tabs = QTabWidget()
        
        # Tab 1: Allgemein
        general_tab = self.create_general_tab()
        tabs.addTab(general_tab, "Allgemein")
        
        # Tab 2: Detection
        detection_tab = self.create_detection_tab()
        tabs.addTab(detection_tab, "Detektion")
        
        # Tab 3: Zielauswahl
        target_tab = self.create_target_tab()
        tabs.addTab(target_tab, "Zielauswahl")
        
        # Tab 4: Maus
        mouse_tab = self.create_mouse_tab()
        tabs.addTab(mouse_tab, "Maus")
        
        right_layout.addWidget(tabs)
        
        # Steuerungs-Buttons
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("▶ Start")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_button.clicked.connect(self.start_tracking)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.stop_button.clicked.connect(self.stop_tracking)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        self.save_button = QPushButton("💾 Config speichern")
        self.save_button.clicked.connect(self.save_config)
        control_layout.addWidget(self.save_button)
        
        right_layout.addLayout(control_layout)
        
        # Log-Ausgabe
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # Splitter für linke/rechte Seite
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 500])
        
        main_layout.addWidget(splitter)
        
        # Timer für GUI-Updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_gui)
        self.update_timer.start(100)  # 10 FPS
        
        # Initialisiere Fenster-Liste beim Start
        self.refresh_windows()
        
    def create_general_tab(self) -> QWidget:
        """Erstellt den Allgemein-Tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Modell-Pfad
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("YOLO Modell:"))
        self.model_path_edit = QLabel("models/yolov8n.pt")
        model_layout.addWidget(self.model_path_edit)
        self.model_browse_button = QPushButton("📁 Durchsuchen")
        self.model_browse_button.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_browse_button)
        layout.addLayout(model_layout)
        
        # Device Auswahl
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["dml (AMD GPU)", "cuda (NVIDIA GPU)", "cpu (CPU)"])
        self.device_combo.setCurrentIndex(0)
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)
        
        # Display-Auflösung Auswahl
        display_layout = QHBoxLayout()
        display_layout.addWidget(QLabel("Display-Auflösung:"))
        self.display_combo = QComboBox()
        self.display_combo.addItems([
            "Auto (System)",
            "1920x1080",
            "2560x1440",
            "1280x720",
            "Custom"
        ])
        self.display_combo.setToolTip("Wähle die Display-Auflösung, damit die Bildschirmmitte korrekt berechnet wird")
        display_layout.addWidget(self.display_combo)

        self.display_width_spin = QSpinBox()
        self.display_width_spin.setRange(100, 10000)
        self.display_width_spin.setValue(1920)
        self.display_width_spin.setEnabled(False)
        display_layout.addWidget(QLabel("W:"))
        display_layout.addWidget(self.display_width_spin)

        self.display_height_spin = QSpinBox()
        self.display_height_spin.setRange(100, 10000)
        self.display_height_spin.setValue(1080)
        self.display_height_spin.setEnabled(False)
        display_layout.addWidget(QLabel("H:"))
        display_layout.addWidget(self.display_height_spin)

        self.display_combo.currentTextChanged.connect(self.toggle_display_selection)
        layout.addLayout(display_layout)
        
        # Fenster-Auswahl
        window_group = QGroupBox("Fenster-Auswahl")
        window_layout = QVBoxLayout()
        
        window_select_layout = QHBoxLayout()
        window_select_layout.addWidget(QLabel("Ziel-Fenster:"))
        self.window_combo = QComboBox()
        self.window_combo.setMinimumWidth(300)
        window_select_layout.addWidget(self.window_combo)
        
        self.refresh_windows_button = QPushButton("🔄 Fenster aktualisieren")
        self.refresh_windows_button.setMaximumWidth(150)
        self.refresh_windows_button.clicked.connect(self.refresh_windows)
        window_select_layout.addWidget(self.refresh_windows_button)
        
        window_layout.addLayout(window_select_layout)
        
        # Fullscreen Checkbox (als Alternative)
        self.window_fullscreen_check = QCheckBox("Ganzer Bildschirm statt Fenster")
        self.window_fullscreen_check.setChecked(True)
        self.window_fullscreen_check.toggled.connect(self.toggle_window_selection)
        window_layout.addWidget(self.window_fullscreen_check)
        
        # Info-Text
        window_info = QLabel("💡 Wähle ein Fenster oder nutze 'Ganzer Bildschirm'")
        window_info.setStyleSheet("color: #888; font-size: 10px;")
        window_layout.addWidget(window_info)
        
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)
        
        # FPS Ziel
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Ziel FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(15, 240)
        self.fps_spin.setValue(60)
        fps_layout.addWidget(self.fps_spin)
        layout.addLayout(fps_layout)
        
        # Capture Region
        region_group = QGroupBox("Capture Region (nur bei Custom Auswahl)")
        region_layout = QVBoxLayout()
        
        self.fullscreen_check = QCheckBox("Ganzer Bildschirm")
        self.fullscreen_check.setChecked(True)
        self.fullscreen_check.toggled.connect(self.toggle_region_inputs)
        region_layout.addWidget(self.fullscreen_check)
        
        region_grid = QGridLayout()
        region_grid.addWidget(QLabel("Links:"), 0, 0)
        self.left_spin = QSpinBox()
        self.left_spin.setRange(0, 5000)
        region_grid.addWidget(self.left_spin, 0, 1)
        
        region_grid.addWidget(QLabel("Oben:"), 0, 2)
        self.top_spin = QSpinBox()
        self.top_spin.setRange(0, 5000)
        region_grid.addWidget(self.top_spin, 0, 3)
        
        region_grid.addWidget(QLabel("Rechts:"), 1, 0)
        self.right_spin = QSpinBox()
        self.right_spin.setRange(0, 5000)
        region_grid.addWidget(self.right_spin, 1, 1)
        
        region_grid.addWidget(QLabel("Unten:"), 1, 2)
        self.bottom_spin = QSpinBox()
        self.bottom_spin.setRange(0, 5000)
        region_grid.addWidget(self.bottom_spin, 1, 3)
        
        region_layout.addLayout(region_grid)
        region_group.setLayout(region_layout)
        layout.addWidget(region_group)
        
        # Preview Option
        self.preview_check = QCheckBox("Vorschau anzeigen (erhöht CPU-Last)")
        self.preview_check.setChecked(True)
        layout.addWidget(self.preview_check)
        
        # Debug Mode
        self.debug_check = QCheckBox("Debug Mode (mehr Ausgaben)")
        layout.addWidget(self.debug_check)
        
        # Hotkey: Tracker Ein/Aus
        hotkey_layout = QHBoxLayout()
        hotkey_layout.addWidget(QLabel("Tracker Ein/Aus (Hotkey):"))
        self.hotkey_combo = QComboBox()
        self.hotkey_combo.addItems([
            "F6", "F7", "F8", "F9",
            "Ctrl+Alt+T", "Ctrl+Shift+T", "Ctrl+Alt+Q", "Alt+Shift+T"
        ])
        self.hotkey_combo.setToolTip("Schaltet das visuelle Erkennungssystem an/aus")
        hotkey_layout.addWidget(self.hotkey_combo)
        layout.addLayout(hotkey_layout)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def create_detection_tab(self) -> QWidget:
        """Erstellt den Detection-Tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Confidence Threshold (niedriger = mehr Detektionen, besser für Spiele)
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(30)
        conf_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.30")
        conf_layout.addWidget(self.conf_label)
        layout.addLayout(conf_layout)
        
        conf_hint = QLabel("Niedriger (0.2–0.35) = mehr Detektionen, besser für Spiele")
        conf_hint.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(conf_hint)
        
        # Klassen-Filter (nur Gegner = Personen)
        filter_group = QGroupBox("Klassen-Filter")
        filter_layout = QVBoxLayout()
        
        self.filter_all_check = QCheckBox("Alle Klassen")
        self.filter_all_check.setChecked(False)
        self.filter_all_check.toggled.connect(self.toggle_class_filter)
        filter_layout.addWidget(self.filter_all_check)
        
        # Klasse 0 (Person) = Gegner
        self.class_person_check = QCheckBox("Personen (ID: 0) – nur Gegner")
        self.class_person_check.setChecked(True)  # Standard: nur Gegner tracken
        filter_layout.addWidget(self.class_person_check)
        
        # Klasse 2 (Auto)
        self.class_car_check = QCheckBox("Autos (ID: 2)")
        filter_layout.addWidget(self.class_car_check)
        
        # Klasse 5 (Bus)
        self.class_bus_check = QCheckBox("Busse (ID: 5)")
        filter_layout.addWidget(self.class_bus_check)
        
        # Klasse 7 (LKW)
        self.class_truck_check = QCheckBox("LKWs (ID: 7)")
        filter_layout.addWidget(self.class_truck_check)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Half Precision (nur für CUDA)
        self.half_check = QCheckBox("Half Precision (nur NVIDIA GPU)")
        layout.addWidget(self.half_check)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def create_target_tab(self) -> QWidget:
        """Erstellt den Zielauswahl-Tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Strategie
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Auswahl-Strategie:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Nächstes zur Mitte (closest_to_center)",
            "Höchste Konfidenz (highest_confidence)"
        ])
        strategy_layout.addWidget(self.strategy_combo)
        layout.addLayout(strategy_layout)
        
        # Bevorzugte Klasse
        pref_layout = QHBoxLayout()
        pref_layout.addWidget(QLabel("Bevorzugte Klasse:"))
        self.pref_class_combo = QComboBox()
        self.pref_class_combo.addItems([
            "Alle Klassen",
            "Personen (0)",
            "Autos (2)",
            "Busse (5)",
            "LKWs (7)"
        ])
        self.pref_class_combo.setCurrentIndex(1)  # Personen (0) - Standard für Gegner
        pref_layout.addWidget(self.pref_class_combo)
        layout.addLayout(pref_layout)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def create_mouse_tab(self) -> QWidget:
        """Erstellt den Maus-Tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Fadenkreuz-Modus (relative Bewegung)
        self.relative_mouse_check = QCheckBox("Fadenkreuz-Modus (relativ bewegen statt Mauszeiger)")
        self.relative_mouse_check.setChecked(True)
        self.relative_mouse_check.setToolTip("Für FPS: Bewegt die Sicht, Fadenkreuz bleibt in der Mitte")
        layout.addWidget(self.relative_mouse_check)
        
        # Smoothing Alpha
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Smoothing Faktor:"))
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(30)
        smooth_layout.addWidget(self.smooth_slider)
        self.smooth_label = QLabel("0.30")
        smooth_layout.addWidget(self.smooth_label)
        layout.addLayout(smooth_layout)
        
        # Info-Text
        info_label = QLabel(
            "0.0 = sehr glatt (langsam)\n"
            "0.5 = mittel\n"
            "1.0 = direkt (kein Smoothing)"
        )
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(info_label)
        
        # Auto-Schießen (linke Maustaste)
        shoot_group = QGroupBox("Auto-Schießen")
        shoot_layout = QVBoxLayout()
        
        self.auto_shoot_check = QCheckBox("Linke Maustaste wenn auf Ziel (automatisch schießen)")
        self.auto_shoot_check.setChecked(True)
        shoot_layout.addWidget(self.auto_shoot_check)
        
        shoot_params = QHBoxLayout()
        shoot_params.addWidget(QLabel("Max. Abstand (px):"))
        self.shoot_threshold_spin = QSpinBox()
        self.shoot_threshold_spin.setRange(5, 150)
        self.shoot_threshold_spin.setValue(40)
        self.shoot_threshold_spin.setToolTip("Schießt wenn Maus näher als X Pixel am Ziel")
        shoot_params.addWidget(self.shoot_threshold_spin)
        
        shoot_params.addWidget(QLabel("Cooldown (ms):"))
        self.shoot_cooldown_spin = QSpinBox()
        self.shoot_cooldown_spin.setRange(50, 500)
        self.shoot_cooldown_spin.setValue(100)
        self.shoot_cooldown_spin.setToolTip("Mindestabstand zwischen Schüssen")
        shoot_params.addWidget(self.shoot_cooldown_spin)
        shoot_layout.addLayout(shoot_params)
        
        shoot_group.setLayout(shoot_layout)
        layout.addWidget(shoot_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def setup_connections(self):
        """Verbindet Signale mit Slots."""
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v/100:.2f}")
        )
        self.smooth_slider.valueChanged.connect(
            lambda v: self.smooth_label.setText(f"{v/100:.2f}")
        )
        
    def refresh_windows(self):
        """Aktualisiert die Liste der verfügbaren Fenster."""
        self.window_combo.clear()
        windows = self.window_selector.list_windows()
        
        for hwnd, title in windows:
            self.window_combo.addItem(title, hwnd)
        
        if windows:
            self.log(f"✅ {len(windows)} Fenster gefunden")
        else:
            self.log("⚠️ Keine Fenster gefunden")
    
    def toggle_window_selection(self, checked):
        """Aktiviert/deaktiviert Fenster-Auswahloptionen."""
        self.window_combo.setEnabled(not checked)
        self.refresh_windows_button.setEnabled(not checked)
        
        if checked:
            # Ganzer Bildschirm gewählt
            self.target_window_hwnd = None
        else:
            # Fenster auswählen
            if self.window_combo.count() == 0:
                self.refresh_windows()

        
    def toggle_region_inputs(self, checked):
        """Aktiviert/deaktiviert Region-Eingaben."""
        self.left_spin.setEnabled(not checked)
        self.top_spin.setEnabled(not checked)
        self.right_spin.setEnabled(not checked)
        self.bottom_spin.setEnabled(not checked)

    def toggle_display_selection(self, text: str):
        """Aktiviert die Custom-Width/Height Eingaben wenn 'Custom' gewählt ist."""
        is_custom = (text == "Custom")
        self.display_width_spin.setEnabled(is_custom)
        self.display_height_spin.setEnabled(is_custom)
        
    def toggle_class_filter(self, checked):
        """Aktiviert/deaktiviert Klassen-Filter."""
        enabled = not checked
        self.class_person_check.setEnabled(enabled)
        self.class_car_check.setEnabled(enabled)
        self.class_bus_check.setEnabled(enabled)
        self.class_truck_check.setEnabled(enabled)
        
    def browse_model(self):
        """Öffnet Dateiauswahl für YOLO Modell."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "YOLO Modell auswählen", "models", 
            "Model files (*.pt *.onnx)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            
    def load_config(self) -> Config:
        """Lädt gespeicherte Konfiguration."""
        # TODO: Von Festplatte laden
        return Config()
        
    def save_config(self):
        """Speichert aktuelle Konfiguration."""
        # TODO: Auf Festplatte speichern
        QMessageBox.information(self, "Gespeichert", "Konfiguration gespeichert!")
        
    def get_config_from_gui(self) -> Config:
        """Erstellt Config-Objekt aus GUI-Einstellungen."""
        device_map = {
            "dml (AMD GPU)": "dml",
            "cuda (NVIDIA GPU)": "cuda",
            "cpu (CPU)": "cpu"
        }
        
        # Klassen-Filter
        class_filter = None
        if not self.filter_all_check.isChecked():
            class_filter = []
            if self.class_person_check.isChecked():
                class_filter.append(0)
            if self.class_car_check.isChecked():
                class_filter.append(2)
            if self.class_bus_check.isChecked():
                class_filter.append(5)
            if self.class_truck_check.isChecked():
                class_filter.append(7)
        
        # Bevorzugte Klasse
        pref_map = {
            "Alle Klassen": None,
            "Personen (0)": 0,
            "Autos (2)": 2,
            "Busse (5)": 5,
            "LKWs (7)": 7
        }
        
        # Capture Region
        capture_region = None
        if not self.fullscreen_check.isChecked():
            capture_region = (
                self.left_spin.value(),
                self.top_spin.value(),
                self.right_spin.value(),
                self.bottom_spin.value()
            )

        # Display-Auflösung aus GUI
        display_text = self.display_combo.currentText()
        if display_text == "Auto (System)":
            display_resolution = None
        elif display_text == "Custom":
            display_resolution = (self.display_width_spin.value(), self.display_height_spin.value())
        else:
            try:
                w, h = display_text.split('x')
                display_resolution = (int(w), int(h))
            except Exception:
                display_resolution = None
        
        # Strategie
        strategy_map = {
            "Nächstes zur Mitte (closest_to_center)": "closest_to_center",
            "Höchste Konfidenz (highest_confidence)": "highest_confidence"
        }
        
        hotkey_map = {
            "F6": "<f6>", "F7": "<f7>", "F8": "<f8>", "F9": "<f9>",
            "Ctrl+Alt+T": "<ctrl>+<alt>+t", "Ctrl+Shift+T": "<ctrl>+<shift>+t",
            "Ctrl+Alt+Q": "<ctrl>+<alt>+q", "Alt+Shift+T": "<alt>+<shift>+t"
        }
        
        return Config(
            model_path=self.model_path_edit.text(),
            conf_threshold=self.conf_slider.value() / 100,
            device=device_map[self.device_combo.currentText()],
            half=self.half_check.isChecked(),
            capture_region=capture_region,
            target_fps=self.fps_spin.value(),
            selection_strategy=strategy_map[self.strategy_combo.currentText()],
            preferred_class=pref_map[self.pref_class_combo.currentText()],
            class_filter=class_filter,
            smoothing_alpha=self.smooth_slider.value() / 100,
            use_relative_mouse=self.relative_mouse_check.isChecked(),
            auto_shoot=self.auto_shoot_check.isChecked(),
            shoot_threshold_px=self.shoot_threshold_spin.value(),
            shoot_cooldown_ms=float(self.shoot_cooldown_spin.value()),
            log_fps_interval=1.0,
            debug_mode=self.debug_check.isChecked(),
            show_preview=self.preview_check.isChecked(),
            display_resolution=display_resolution,
            toggle_hotkey=hotkey_map.get(self.hotkey_combo.currentText(), "<f6>")
        )
        
    def start_tracking(self):
        """Startet das Tracking-System."""
        try:
            # Fenster auswählen wenn nicht Fullscreen
            if not self.window_fullscreen_check.isChecked():
                window_hwnd = self.window_combo.currentData()
                if window_hwnd:
                    self.target_window_hwnd = window_hwnd
                    window_title = self.window_combo.currentText()
                    result = self.window_selector.select_window(window_title)
                    if not result:
                        QMessageBox.warning(self, "Fenster nicht gefunden", 
                                          f"Fenster '{window_title}' konnte nicht gefunden werden!")
                        return
                    self.log(f"🪟 Fenster '{window_title}' ausgewählt")
                else:
                    QMessageBox.warning(self, "Fenster nicht gewählt", 
                                      "Bitte wählen Sie ein Fenster oder 'Ganzer Bildschirm'")
                    return
            
            # Config aus GUI holen
            config = self.get_config_from_gui()
            
            # Tracking-System erstellen
            self.tracking_system = TrackingSystem(config)
            
            # Wenn Fenster gewählt: Capture Region vom Fenster setzen
            if self.target_window_hwnd and self.window_selector.window_rect:
                self.tracking_system.capture.region = self.window_selector.window_rect
            
            # Capture starten
            self.tracking_system.capture.start()
            
            # Inference Thread
            self.inference_thread = InferenceThread(config)
            self.inference_thread.frame_ready.connect(self.update_preview)
            self.inference_thread.detection_update.connect(self.update_detection_info)
            self.inference_thread.detection_count_changed.connect(self.update_detection_count)
            self.inference_thread.tracker_state_changed.connect(self.update_tracker_status)
            # Übergebe den window_selector wenn ein Fenster selected ist
            if self.target_window_hwnd:
                self.inference_thread.setup(self.tracking_system, self.window_selector)
            else:
                self.inference_thread.setup(self.tracking_system)
            self.inference_thread.show_preview = config.show_preview
            self.inference_thread.start()
            
            # Global Hotkey: Tracker Ein/Aus
            self._start_hotkey_listener(config.toggle_hotkey)
            
            # UI aktualisieren
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("🟢 Tracking läuft")
            self.status_label.setStyleSheet("color: #0f0; font-weight: bold;")
            self.update_tracker_status(True)  # Start mit Active
            self.tracker_status_label.setVisible(True)
            self.detection_count_label.setVisible(True)
            
            self.log("✅ Tracking gestartet")
            
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Start: {str(e)}")
            self.log(f"❌ Fehler: {str(e)}")
            
    def _start_hotkey_listener(self, hotkey: str = "<f6>"):
        """Startet globalen Hotkey-Listener (Tracker Ein/Aus)."""
        def on_activate():
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, self._hotkey_toggle_tracking)
        
        try:
            self.hotkey_listener = keyboard.GlobalHotKeys({hotkey: on_activate})
            self.hotkey_listener.start()
            self._current_hotkey_display = self.hotkey_combo.currentText()
            self.log(f"⌨️ Hotkey {self._current_hotkey_display}: Tracker an/aus")
        except Exception as e:
            self.log(f"⚠️ Hotkey konnte nicht registriert werden: {e}")
    
    def _stop_hotkey_listener(self):
        """Stoppt den Hotkey-Listener."""
        if self.hotkey_listener:
            try:
                self.hotkey_listener.stop()
            except Exception:
                pass
            self.hotkey_listener = None
    
    def _hotkey_toggle_tracking(self):
        """Hotkey-Handler: Tracker aktivieren/deaktivieren (läuft im Qt-Hauptthread)."""
        if self.inference_thread:
            self.inference_thread.toggle_tracking()
    
    def update_detection_count(self, count: int):
        """Aktualisiert die Anzahl erkannter Objekte in der GUI."""
        self.detection_count_label.setText(f"Erkannte Objekte: {count}")
        self.detection_count_label.setStyleSheet(
            "color: #0f0; font-weight: bold;" if count > 0 else "color: #aaa;"
        )
    
    def update_tracker_status(self, active: bool):
        """Aktualisiert die Tracker-Status-Anzeige in der GUI."""
        hotkey = getattr(self, "_current_hotkey_display", "F6")
        if active:
            self.tracker_status_label.setText(f"Tracker: ● Active ({hotkey} zum Umschalten)")
            self.tracker_status_label.setStyleSheet("color: #0f0; font-weight: bold;")
        else:
            self.tracker_status_label.setText(f"Tracker: ○ Inactive ({hotkey} zum Umschalten)")
            self.tracker_status_label.setStyleSheet("color: #f80; font-weight: bold;")
    
    def stop_tracking(self):
        """Stoppt das Tracking-System."""
        self._stop_hotkey_listener()
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread.wait()
            
        if self.tracking_system:
            self.tracking_system.capture.stop()
            
        # UI aktualisieren
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("🔴 Gestoppt")
        self.status_label.setStyleSheet("color: #f00; font-weight: bold;")
        self.tracker_status_label.setVisible(False)  # Nur relevant wenn Tracking läuft
        self.detection_count_label.setVisible(False)
        
        self.preview_label.setText("Vorschau gestoppt")
        self.log("⏹ Tracking gestoppt")
        
    def update_preview(self, frame):
        """Aktualisiert das Vorschaubild."""
        if frame is not None:
            # Konvertiere OpenCV BGR zu RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # Erstelle QImage
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Skaliere auf Label-Größe
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.preview_label.setPixmap(scaled_pixmap)
            
    def update_detection_info(self, info: str):
        """Aktualisiert Detektions-Info."""
        self.detection_info.append(info)
        # Automatisch scrollen
        cursor = self.detection_info.textCursor()
        cursor.movePosition(cursor.End)
        self.detection_info.setTextCursor(cursor)
        
    def update_gui(self):
        """Periodische GUI-Updates."""
        if self.tracking_system:
            # TODO: FPS-Werte holen und anzeigen
            pass
            
    def log(self, message: str):
        """Fügt Nachricht zum Log hinzu."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Automatisch scrollen
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)
        
    def closeEvent(self, event):
        """Beim Schließen des Fensters."""
        self.stop_tracking()
        event.accept()


# Füge TrackingSystem Klasse hier ein (oder importiere sie)
# Da wir TrackingSystem bereits in main.py haben, importieren wir es
from main import TrackingSystem


def main():
    """GUI starten."""
    app = QApplication(sys.argv)
    
    # Dunkles Theme
    app.setStyle('Fusion')
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
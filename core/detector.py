"""
Objekt Detektion Modul - Verwendet YOLO für Echtzeit-Objekterkennung.
Unterstützt AMD (DirectML), NVIDIA (CUDA) und CPU.
"""
from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from config import Config


class ObjectDetector:
    """YOLO-basierter Detektor mit automatischer Hardware-Erkennung."""
    
    # COCO Klassen für Referenz
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
        59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
        79: 'toothbrush'
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.device = self._setup_device()
        self._dml_fallback_used = False
        
        print(f"🤖 Lade YOLO Modell: {config.model_path}")
        self.model = YOLO(config.model_path)
        self.model.to(self.device)
        print(f"✅ Modell geladen auf: {self.device}")
        
    def _setup_device(self):
        """Richtet das optimale Device ein (GPU/CPU)."""
        if self.config.device == "dml":
            try:
                import torch_directml
                device = torch_directml.device()
                print(f"🎮 AMD GPU erkannt: {device}")
                return device
            except ImportError:
                print("⚠️ torch-directml nicht installiert, verwende CPU")
                return "cpu"
                
        elif self.config.device == "cuda":
            if torch.cuda.is_available():
                print(f"🎮 NVIDIA GPU erkannt: {torch.cuda.get_device_name()}")
                return "cuda"
            else:
                print("⚠️ CUDA nicht verfügbar, verwende CPU")
                return "cpu"
        else:
            return "cpu"
            
    def detect(self, frame: np.ndarray) -> List:
        """
        Führt Objekterkennung auf einem Frame durch.
        
        Returns:
            Liste von Detektionen: (bbox, confidence, class_id)
            bbox = [x1, y1, x2, y2] in Pixelkoordinaten
        """
        frame = np.ascontiguousarray(frame.copy())
        try:
            results = self.model(
                frame,
                conf=self.config.conf_threshold,
                verbose=False,
                device=self.device
            )[0]
        except RuntimeError as e:
            if "version_counter" in str(e) and not self._dml_fallback_used:
                self._dml_fallback_used = True
                self.device = "cpu"
                self.model.to("cpu")
                print("⚠️ DirectML-Fehler, wechsle zu CPU")
                results = self.model(
                    frame,
                    conf=self.config.conf_threshold,
                    verbose=False,
                    device="cpu"
                )[0]
            else:
                raise
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Optional: Klassen-Filter anwenden
            if self.config.class_filter and cls_id not in self.config.class_filter:
                continue
                
            detections.append(([x1, y1, x2, y2], conf, cls_id))
            
            # Debug: Zeige erkannte Objekte
            if self.config.debug_mode:
                class_name = self.COCO_CLASSES.get(cls_id, f"Unbekannt({cls_id})")
                print(f"🔍 Erkannt: {class_name} ({conf:.2f})")
        
        return detections
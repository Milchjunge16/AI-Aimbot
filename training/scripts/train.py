"""
YOLO Model Training Script
Trainiert ein benutzerdefiniertes YOLO-Modell
"""

from ultralytics import YOLO
import os
from pathlib import Path

# Verzeichnisse
BASE_DIR = Path(__file__).parent.parent
DATA_YAML = BASE_DIR / "data.yaml"
MODELS_DIR = BASE_DIR / "models_trained"

# Sicherstellen, dass Ausgabeverzeichnis existiert
MODELS_DIR.mkdir(exist_ok=True)

def train_model():
    """Trainiert das YOLO-Modell"""
    
    print("=" * 60)
    print("YOLO Modell Training")
    print("=" * 60)
    
    # Modell laden (pre-trained)
    # Optionen: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    # n=nano (schnell), s=small, m=medium, l=large, x=extra large (genau)
    model = YOLO('yolov8s.pt')
    
    print(f"\n📦 Verwende: yolov8s.pt")
    print(f"📁 Dataset: {DATA_YAML}")
    print(f"💾 Trainingsergebnisse speichern in: {MODELS_DIR}\n")
    
    # Training starten
    results = model.train(
        data=str(DATA_YAML),
        epochs=50,              # Anzahl der Trainings-Epochen
        imgsz=640,              # Input Image Size
        device=0,               # GPU ID (0 = erste GPU, -1 = CPU)
        batch=8,                # Batch Size (erhöhen wenn genug VRAM)
        patience=10,            # Early Stopping (stoppe nach 10 Epochen ohne Verbesserung)
        save=True,              # Modell speichern
        project=str(MODELS_DIR),
        name='yolov8_custom',
        pretrained=True,        # Pre-trained weights verwenden
        optimizer='SGD',        # Optimizer: SGD oder Adam
        lr0=0.01,               # Initial Learning Rate
        lrf=0.01,               # Final Learning Rate
        momentum=0.937,         # Momentum
        weight_decay=0.0005,    # Weight Decay
        warmup_epochs=3,        # Warmup Epochen
        augment=True,           # Data Augmentation
        hsv_h=0.015,            # HSV-Hue Augmentation
        hsv_s=0.7,              # HSV-Saturation Augmentation
        hsv_v=0.4,              # HSV-Value Augmentation
        degrees=10.0,           # Rotation
        translate=0.1,          # Translation
        scale=0.5,              # Scale
        flipud=0.0,             # Flip Upside-Down
        fliplr=0.5,             # Flip Left-Right
    )
    
    print("\n" + "=" * 60)
    print("✅ Training abgeschlossen!")
    print("=" * 60)
    
    return results


def validate_model():
    """Validiert das trainierte Modell"""
    
    print("\n" + "=" * 60)
    print("Modell Validierung")
    print("=" * 60 + "\n")
    
    # Letztes trainiertes Modell laden
    model_path = MODELS_DIR / "yolov8_custom" / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"❌ Modell nicht gefunden: {model_path}")
        return
    
    model = YOLO(str(model_path))
    
    # Validierung
    metrics = model.val()
    
    print("\n📊 Validierungsergebnisse:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    return metrics


def test_inference(image_path):
    """Testet das Modell mit einem Bild"""
    
    print("\n" + "=" * 60)
    print("Inferenz Test")
    print("=" * 60 + "\n")
    
    # Bestes Modell laden
    model_path = MODELS_DIR / "yolov8_custom" / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"❌ Modell nicht gefunden: {model_path}")
        return
    
    model = YOLO(str(model_path))
    
    # Inferenz
    results = model.predict(
        source=image_path,
        conf=0.5,  # Confidence Threshold
        iou=0.45,  # IOU Threshold
        save=True,
        project=str(MODELS_DIR),
        name='predictions'
    )
    
    print(f"✅ Inferenz abgeschlossen")
    print(f"   Ergebnis gespeichert in: {MODELS_DIR}/predictions")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "train":
            train_model()
        elif command == "validate":
            validate_model()
        elif command == "inference" and len(sys.argv) > 2:
            test_inference(sys.argv[2])
        else:
            print("Verwendung:")
            print("  python train.py train          # Trainiert das Modell")
            print("  python train.py validate       # Validiert das Modell")
            print("  python train.py inference <image_path>  # Testet mit Bild")
    else:
        # Standard: Training starten
        train_model()

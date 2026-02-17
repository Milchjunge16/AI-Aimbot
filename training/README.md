# 🎯 YOLO Model Training Guide

Dieser Ordner enthält alles für das Training eines benutzerdefinierten YOLO-Modells.

## 📁 Ordnerstruktur

```
training/
├── dataset/
│   ├── images/
│   │   ├── train/      ← 70% der Bilder hier
│   │   ├── val/        ← 20% der Bilder hier
│   │   └── test/       ← 10% der Bilder hier
│   └── labels/
│       ├── train/      ← Entsprechende Label (.txt)
│       ├── val/
│       └── test/
├── scripts/
│   ├── train.py        ← Haupttraining-Skript
│   └── prepare_dataset.py  ← Dataset vorbereiten
├── models_trained/     ← Trainierte Modelle speichern
├── data.yaml          ← Dataset-Konfiguration
└── README.md          ← Diese Datei
```

## 🚀 Schnellstart

### 1. **Abhängigkeiten installieren**
```bash
pip install ultralytics opencv-python torch torchvision
```

### 2. **Bilder vorbereiten**

**Option A: Automatisch aufteilen**
```bash
# Legen Sie alle Bilder in dataset/images/ ab
python scripts/prepare_dataset.py
```

**Option B: Manuell**
- Legen Sie ~70% der Bilder in `dataset/images/train/`
- ~20% in `dataset/images/val/`
- ~10% in `dataset/images/test/`

### 3. **Labels annotieren**

Labels müssen im YOLO-Format sein (**[0, 1]**)
- Dateiname: `image_name.txt`
- Format pro Box: `class_id x_center y_center width height`

**Beispiel:**
```
0 0.5 0.5 0.3 0.4
```
(Klasse 0, Mittelpunkt bei 50%, Breite 30%, Höhe 40%)

**Annotation Tools:**
- [Roboflow Annotator](https://roboflow.com/)
- [LabelImg](https://github.com/heartexlabs/labelImg)
- [CVAT](https://www.cvat.ai/)

### 4. **Dataset-Konfiguration**

Bearbeiten Sie `data.yaml`:
```yaml
nc: 1                    # Anzahl Klassen
names:
  0: 'target'           # Klassenname
  # 1: 'second_class'
```

### 5. **Modell trainieren**

```bash
# Einfaches Training
python scripts/train.py train

# Oder direkt
cd scripts
python train.py

# Validierung
python train.py validate

# Inferenz testen
python train.py inference path/to/image.jpg
```

## 📊 Training Parameter

In `scripts/train.py` können Sie anpassen:

| Parameter | Bedeutung | Standard |
|-----------|-----------|----------|
| `epochs` | Anzahl Trainings-Durchläufe | 50 |
| `imgsz` | Input Bild Größe | 640 |
| `batch` | Bilder pro Batch | 8 |
| `device` | GPU ID (0) oder CPU (-1) | 0 |
| `patience` | Early Stopping nach N Epochen | 10 |
| `lr0` | Anfängliche Learning Rate | 0.01 |

**Modell-Varianten:**
- `yolov8n` - Nano (schnell, weniger Speicher)
- `yolov8s` - Small (Standard) ✅
- `yolov8m` - Medium (besser)
- `yolov8l` - Large (langsam)
- `yolov8x` - Extra Large (sehr langsar)

## 💡 Tipps für bessere Ergebnisse

✅ **Daten-Qualität**
- Mindestens 100-200 Bilder pro Klasse
- Verschiedene Beleuchtung und Winkel
- Unterschiedliche Abstände und Größen
- Hintergrund-Variabilität

✅ **Training**
- Starten Sie mit `yolov8s` oder `yolov8m`
- Nutzen Sie GPU (device=0)
- Augmentation hilft bei wenigen Bildern
- Erhöhen Sie `batch` für schneller Training (wenn VRAM erlaubt)

✅ **Hyperparameter**
- Learning Rate: 0.001-0.01
- Batch Size: 8-32 (je nach GPU)
- Epochs: 50-200 (abhängig von Datenmenge)

⚠️ **Häufige Probleme**
- "CUDA out of memory": Reduzieren Sie `batch` oder nutzen Sie CPU
- Schlechte Ergebnisse: Mehr Trainings-Daten, bessere Annotationen
- Langsames Training: Nutzen Sie GPU, erhöhen Sie `batch`

## 📈 Ergebnisse

Nach dem Training finden Sie:

```
models_trained/
└── yolov8_custom/
    ├── weights/
    │   ├── best.pt      ← Bestes Modell (verwenden!)
    │   └── last.pt      ← Letztes Modell
    └── results.csv      ← Trainings-Statistiken
```

## 🔄 Modell in Hauptprogramm verwenden

```python
from ultralytics import YOLO

model = YOLO('training/models_trained/yolov8_custom/weights/best.pt')
results = model.predict('image.jpg')
```

## 📚 Weiterführende Ressourcen

- [YOLOv8 Dokumentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [Roboflow - Dataset Management](https://roboflow.com/)

---

**Viel Erfolg beim Training! 🚀**

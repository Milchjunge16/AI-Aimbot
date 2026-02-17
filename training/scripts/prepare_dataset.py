"""
Dataset Vorbereitung für YOLO Training
- Bilder in train/val/test aufteilen
- YOLO Label Format konvertieren
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# Split Verhältnisse
TRAIN_RATIO = 0.7      # 70% Training
VAL_RATIO = 0.2        # 20% Validierung
TEST_RATIO = 0.1       # 10% Testing


def split_dataset():
    """Teilt Bilder in train/val/test auf"""
    
    print("=" * 60)
    print("Dataset Aufteilen")
    print("=" * 60 + "\n")
    
    # Alle Bilder sammeln
    all_images = list(Path(IMAGES_DIR).glob("*.jpg")) + \
                 list(Path(IMAGES_DIR).glob("*.png")) + \
                 list(Path(IMAGES_DIR).glob("*.JPG"))
    
    if not all_images:
        print("❌ Keine Bilder im 'images' Ordner gefunden!")
        return
    
    print(f"📷 Gefunden: {len(all_images)} Bilder\n")
    
    # Shuffle
    random.shuffle(all_images)
    
    # Indices berechnen
    train_count = int(len(all_images) * TRAIN_RATIO)
    val_count = int(len(all_images) * VAL_RATIO)
    
    train_images = all_images[:train_count]
    val_images = all_images[train_count:train_count + val_count]
    test_images = all_images[train_count + val_count:]
    
    print(f"📊 Split:")
    print(f"   🚂 Training:   {len(train_images)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"   ✔️  Validierung: {len(val_images)} ({VAL_RATIO*100:.0f}%)")
    print(f"   🧪 Testing:    {len(test_images)} ({TEST_RATIO*100:.0f}%)\n")
    
    # Verschiebe Bilder
    def move_images(image_list, split_name):
        split_dir = IMAGES_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in image_list:
            try:
                # Wenn bereits im Unterordner, überspringen
                if img_path.parent.name in ['train', 'val', 'test']:
                    continue
                    
                dest = split_dir / img_path.name
                shutil.copy2(img_path, dest)
                print(f"   ✓ {img_path.name} → {split_name}/")
            except Exception as e:
                print(f"   ✗ Fehler bei {img_path.name}: {e}")
    
    print("Verschiebe Bilder...\n")
    move_images(train_images, 'train')
    move_images(val_images, 'val')
    move_images(test_images, 'test')
    
    print("\n✅ Dataset aufgeteilt!")


def convert_to_yolo_format(image_path, boxes):
    """
    Konvertiert Bounding Boxes zu YOLO Format
    
    Args:
        image_path: Pfad zum Bild
        boxes: Liste von [x_min, y_min, x_max, y_max, class_id]
    
    Returns:
        YOLO Format String: "class_id x_center y_center width height"
        (alle Werte normalisiert 0-1)
    """
    
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    yolo_labels = []
    for box in boxes:
        x_min, y_min, x_max, y_max, class_id = box
        
        # Normalisieren
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return "\n".join(yolo_labels)


def create_label_files():
    """Erstellt Dummy Label-Dateien für Beispiele"""
    
    print("\n" + "=" * 60)
    print("Label-Dateien erstellen")
    print("=" * 60 + "\n")
    
    for split in ['train', 'val', 'test']:
        images_path = IMAGES_DIR / split
        labels_path = LABELS_DIR / split
        
        labels_path.mkdir(parents=True, exist_ok=True)
        
        # Für jedes Bild eine Label-Datei erstellen
        for img_file in images_path.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.png']:
                label_file = labels_path / (img_file.stem + ".txt")
                
                # Falls noch nicht vorhanden, Dummy Label erstellen
                if not label_file.exists():
                    # Beispiel: Klasse 0, Box in der Mitte (0.5, 0.5, 0.3, 0.3)
                    label_file.write_text("0 0.5 0.5 0.3 0.3\n")
                    print(f"✓ {label_file.name}")
    
    print("\n⚠️  Hinweis: Bitte ersetzen Sie die Label-Dateien mit echten Annotationen!")
    print("   Format: class_id x_center y_center width height (alle 0-1)")


if __name__ == "__main__":
    # Stelle sicher, dass Basis-Verzeichnis existiert
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🎯 YOLO Dataset Vorbereitung")
    print("=" * 60)
    print("\nPlatzieren Sie Ihre Bilder im: dataset/images/ Ordner")
    print("Die Bilder werden dann in train/val/test aufgeteilt.\n")
    
    input("Drücken Sie Enter wenn Sie bereit sind...")
    
    split_dataset()
    create_label_files()
    
    print("\n" + "=" * 60)
    print("✅ Vorbereitung abgeschlossen!")
    print("=" * 60)
    print("\nNächste Schritte:")
    print("1. Überprüfen Sie die Label-Dateien in 'dataset/labels/'")
    print("2. Führen Sie 'python train.py train' aus zum Trainieren")

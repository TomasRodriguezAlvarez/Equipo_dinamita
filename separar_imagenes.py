import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = Path("dataset/dataset-resized")
OUTPUT_DIR = Path("dataset")

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

classes = [d.name for d in SOURCE_DIR.iterdir() if d.is_dir()]

for split in ["train", "validation", "test"]:
    for cls in classes:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

for cls in classes:
    images = list((SOURCE_DIR / cls).glob("*"))
    images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    train_imgs = images[:train_end]
    valid_imgs = images[train_end:valid_end]
    test_imgs = images[valid_end:]

    for img in train_imgs:
        shutil.copy2(img, OUTPUT_DIR / "train" / cls / img.name)

    for img in valid_imgs:
        shutil.copy2(img, OUTPUT_DIR / "validation" / cls / img.name)

    for img in test_imgs:
        shutil.copy2(img, OUTPUT_DIR / "test" / cls / img.name)

    print(f"{cls}: train={len(train_imgs)}, validation={len(valid_imgs)}, test={len(test_imgs)}")

print("Dataset separado correctamente.")
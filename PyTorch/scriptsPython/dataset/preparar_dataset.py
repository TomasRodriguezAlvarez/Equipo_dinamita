"""
Separación del dataset TrashNet en train / validation / test.

Independiente del entrenamiento (punto 2 de la pauta): este script solo copia
imágenes de `dataset/dataset-resized/<clase>/` a `dataset/{train,validation,test}/<clase>/`.
Ningún script de `modelos/` descarga ni reparte datos; todos asumen que esto ya corrió.

Es la versión argumentable de `separar_imagenes.py` (raíz del repo), con las mismas
proporciones por defecto (70/15/15) y la misma semilla (42) para que el split sea
reproducible.

Uso:
    python preparar_dataset.py
    python preparar_dataset.py --train 0.8 --valid 0.1 --test 0.1
    python preparar_dataset.py --origen otra_carpeta/dataset-resized --destino otro_dataset
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

SCRIPTSPY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTORCH_DIR   = os.path.dirname(SCRIPTSPY_DIR)
PROJECT_DIR   = os.path.dirname(PYTORCH_DIR)

DEFAULT_ORIGEN   = os.path.join(PROJECT_DIR, "dataset", "dataset-resized")
DEFAULT_DESTINO  = os.path.join(PROJECT_DIR, "dataset")

EXTENSIONES = (".jpg", ".jpeg", ".png")


def separar(origen: Path, destino: Path, train_ratio: float, valid_ratio: float,
           test_ratio: float, seed: int):
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            f"Las proporciones deben sumar 1.0 (train {train_ratio} + valid "
            f"{valid_ratio} + test {test_ratio} = {train_ratio + valid_ratio + test_ratio})"
        )
    if not origen.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de origen: {origen}")

    random.seed(seed)

    clases = sorted(d.name for d in origen.iterdir() if d.is_dir())
    if not clases:
        raise ValueError(f"No se encontraron subcarpetas de clase en {origen}")

    for split in ("train", "validation", "test"):
        for clase in clases:
            (destino / split / clase).mkdir(parents=True, exist_ok=True)

    print(f"Origen:  {origen}")
    print(f"Destino: {destino}")
    print(f"Clases:  {clases}")
    print(f"Proporciones: train={train_ratio:.0%} validation={valid_ratio:.0%} test={test_ratio:.0%}")
    print(f"Semilla: {seed}\n")

    for clase in clases:
        imagenes = [
            img for img in (origen / clase).glob("*")
            if img.suffix.lower() in EXTENSIONES
        ]
        random.shuffle(imagenes)

        total = len(imagenes)
        fin_train = int(total * train_ratio)
        fin_valid = fin_train + int(total * valid_ratio)

        splits = {
            "train": imagenes[:fin_train],
            "validation": imagenes[fin_train:fin_valid],
            "test": imagenes[fin_valid:],
        }

        for split, archivos in splits.items():
            for img in archivos:
                shutil.copy2(img, destino / split / clase / img.name)

        print(
            f"{clase:<12} train={len(splits['train']):>4}  "
            f"validation={len(splits['validation']):>4}  test={len(splits['test']):>4}"
        )

    print("\nDataset separado correctamente.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--origen", default=DEFAULT_ORIGEN,
                   help=f"carpeta con las imágenes originales por clase (default: {DEFAULT_ORIGEN})")
    p.add_argument("--destino", default=DEFAULT_DESTINO,
                   help=f"carpeta donde crear train/validation/test (default: {DEFAULT_DESTINO})")
    p.add_argument("--train", type=float, default=0.70, help="proporción de train (default 0.70)")
    p.add_argument("--valid", type=float, default=0.15, help="proporción de validation (default 0.15)")
    p.add_argument("--test",  type=float, default=0.15, help="proporción de test (default 0.15)")
    p.add_argument("--seed",  type=int,   default=42,   help="semilla del shuffle (default 42)")
    return p.parse_args()


def main():
    args = parse_args()
    separar(
        Path(args.origen), Path(args.destino),
        args.train, args.valid, args.test, args.seed,
    )


if __name__ == "__main__":
    main()

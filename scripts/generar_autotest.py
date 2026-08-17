"""
Prepara el autotest que trae la app Ionic: copia las imágenes de
`imagenes_a_test/` a los assets y genera el `esperado.json` con la predicción de
referencia de cada modelo PyTorch sobre cada imagen.

La app corre esas mismas imágenes con onnxruntime-web y compara contra este
archivo. Es la Fase 5 del plan hecha dentro de la app, así que también se puede
correr en el celular y no solo en el navegador de escritorio.

Requiere el env con los pesos (conda activate frameworks-ia).

Uso:
    python scripts/generar_autotest.py
"""

import json
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from export_onnx import MODELOS, IMG_SIZE, MEAN, STD, cargar_modelo

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGEN = REPO_ROOT / "imagenes_a_test"
DESTINO = REPO_ROOT / "trashnet-tester/src/assets/pruebas"

eval_tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def nombre_seguro(ruta):
    """Sin espacios: el nombre termina siendo parte de una URL de asset."""
    return ruta.name.replace(" ", "_")


def decidir(info, probs):
    """Misma regla que aplica la app: umbral calibrado en el binario, argmax en el resto."""
    if "umbral_basura" in info:
        idx = info["idx_basura"]
        elegido = idx if probs[idx] >= info["umbral_basura"] else 1 - idx
        return info["clases"][elegido], float(probs[elegido])
    elegido = int(probs.argmax())
    return info["clases"][elegido], float(probs[elegido])


def main():
    imagenes = sorted(ORIGEN.glob("*.jp*g"))
    if not imagenes:
        raise SystemExit(f"No hay imágenes en {ORIGEN}")

    DESTINO.mkdir(parents=True, exist_ok=True)
    for img in imagenes:
        shutil.copy(img, DESTINO / nombre_seguro(img))

    casos = []
    for info in MODELOS:
        modelo = cargar_modelo(info["checkpoint"], len(info["clases"]))
        for ruta in imagenes:
            tensor = eval_tf(Image.open(ruta).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(modelo(tensor), dim=1)[0].numpy()
            clase, confianza = decidir(info, probs)
            casos.append(
                {
                    "modelo": info["nombre"],
                    "imagen": f"assets/pruebas/{nombre_seguro(ruta)}",
                    "clase": clase,
                    "confianza": round(confianza, 6),
                }
            )
            print(f"{info['nombre']:22s} {ruta.name:14s} -> {clase} ({confianza:.4f})")

    salida = DESTINO / "esperado.json"
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generado_por": "scripts/generar_autotest.py",
                "referencia": "PyTorch (los .pt de _optuna), pipeline eval_tf",
                "casos": casos,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nOK: {salida} con {len(casos)} casos")


if __name__ == "__main__":
    main()

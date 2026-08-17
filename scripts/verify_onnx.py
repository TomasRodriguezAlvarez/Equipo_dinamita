"""
Verifica cada .onnx exportado contra el modelo PyTorch original, usando las
imágenes de imagenes_a_test/. Compara logits (deben coincidir con tolerancia
numérica chica) y confirma que la clase predicha es la misma.

Requiere el mismo env de export_onnx.py más `onnxruntime` (ver ONNX_EXPORT_UBUNTU.md).

Uso:
    python scripts/verify_onnx.py
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

from export_onnx import MODELOS, IMG_SIZE, MEAN, STD, cargar_modelo

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGENES_TEST = list((REPO_ROOT / "imagenes_a_test").glob("*.jp*g"))

eval_tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def verificar(info):
    n_classes = len(info["clases"])
    modelo_pt = cargar_modelo(info["checkpoint"], n_classes)
    sesion = ort.InferenceSession(str(info["salida"]))

    print(f"\n=== {info['nombre']} ===")
    for ruta_img in IMAGENES_TEST:
        img = Image.open(ruta_img).convert("RGB")
        tensor = eval_tf(img).unsqueeze(0)

        with torch.no_grad():
            logits_pt = modelo_pt(tensor).numpy()

        logits_onnx = sesion.run(None, {"input": tensor.numpy()})[0]

        diff = np.abs(logits_pt - logits_onnx).max()
        clase_pt = info["clases"][int(logits_pt.argmax())]
        clase_onnx = info["clases"][int(logits_onnx.argmax())]

        estado = "OK" if diff < 1e-3 and clase_pt == clase_onnx else "MISMATCH"
        print(
            f"  [{estado}] {ruta_img.name}: pt={clase_pt} onnx={clase_onnx} "
            f"diff_max={diff:.2e}"
        )


if __name__ == "__main__":
    if not IMAGENES_TEST:
        print(f"No se encontraron imágenes en {REPO_ROOT / 'imagenes_a_test'}")
    for info in MODELOS:
        verificar(info)

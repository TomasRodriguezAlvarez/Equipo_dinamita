"""
Exporta a ONNX los 3 modelos PyTorch "_optuna" de TrashNet:
  - trashnet_mobilenetv3 (6 clases de material)
  - trashnet_jerarquico  (4 clases de contenedor)
  - trashnet_binario     (2 clases: basura/reciclable)

Correr con el env que tiene los pesos entrenados (conda activate frameworks-ia),
con onnx y onnxruntime instalados encima. Ver ONNX_EXPORT_UBUNTU.md para el detalle.

Uso:
    python scripts/export_onnx.py
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

REPO_ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MODELOS = [
    {
        "nombre": "trashnet_mobilenetv3",
        "checkpoint": REPO_ROOT / "PyTorch/modelos/trashnet_mobilenetv3_optuna.pt",
        "clases": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
        "salida": REPO_ROOT / "onnx/trashnet_mobilenetv3.onnx",
    },
    {
        "nombre": "trashnet_jerarquico",
        "checkpoint": REPO_ROOT / "PyTorch/jerarquico/modelos/trashnet_jerarquico_optuna.pt",
        "clases": ["amarillo", "azul", "gris", "verde"],
        "salida": REPO_ROOT / "onnx/trashnet_jerarquico.onnx",
    },
    {
        "nombre": "trashnet_binario",
        "checkpoint": REPO_ROOT / "PyTorch/binario/modelos/trashnet_binario_optuna.pt",
        "clases": ["basura", "reciclable"],
        "salida": REPO_ROOT / "onnx/trashnet_binario.onnx",
        # Umbral calibrado sobre validación (no usar argmax/0.5) — ver PyTorch/CONTEXT.md §8.
        "umbral_basura": 0.837,
        "idx_basura": 0,
    },
]


class TrashNetMobileNetV3(nn.Module):
    """Misma arquitectura que en los notebooks: backbone MobileNetV3Small +
    GlobalAvgPool + Linear-ReLU-Dropout-Linear. Sin pesos preentrenados de ImageNet
    acá porque los va a pisar el state_dict cargado a continuación."""

    def __init__(self, dense_units, dropout_rate, n_classes):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        self.features = base.features
        n_features = base.classifier[0].in_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


def cargar_modelo(checkpoint_path, n_classes):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Los notebooks guardan un dict con el state_dict + metadata (dense_units,
    # classes, img_size...). Se acepta también un state_dict pelado por si acaso.
    state_dict = checkpoint.get("state_dict", checkpoint)
    # dense_units varía por modelo (lo eligió Optuna); se infiere de la forma
    # guardada en el propio checkpoint en vez de hardcodearlo.
    dense_units = state_dict["head.1.weight"].shape[0]
    modelo = TrashNetMobileNetV3(dense_units, dropout_rate=0.0, n_classes=n_classes)
    modelo.load_state_dict(state_dict)
    modelo.eval()
    return modelo


def clases_del_checkpoint(checkpoint_path):
    """Clases guardadas en el checkpoint, para contrastarlas con las de MODELOS."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("classes") if isinstance(checkpoint, dict) else None


def exportar(info):
    n_classes = len(info["clases"])

    # El orden de las clases decide el significado de cada logit: si no coincide
    # con el del entrenamiento, la app etiquetaría mal sin dar ningún error.
    clases_ck = clases_del_checkpoint(info["checkpoint"])
    if clases_ck is not None and list(clases_ck) != info["clases"]:
        raise ValueError(
            f"{info['nombre']}: las clases del checkpoint {list(clases_ck)} no "
            f"coinciden con {info['clases']}"
        )

    modelo = cargar_modelo(info["checkpoint"], n_classes)

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    salida = Path(info["salida"])
    salida.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        modelo,
        dummy,
        str(salida),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        # 18 es el opset nativo del exportador de torch 2.12 para este modelo.
        # Pedir 17 dispara el version converter de onnx, que falla en el Reshape
        # del pooling; onnxruntime-web soporta 18 de sobra, así que no hace falta.
        opset_version=18,
        # Sin esto el exportador de torch 2.12 saca los pesos a un .onnx.data
        # aparte; la app los quiere en un único archivo por modelo.
        external_data=False,
    )

    labels = {
        "clases": info["clases"],
        "input_size": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
        "nota": "Resize directo a input_size x input_size, sin crop (ver eval_tf en los notebooks).",
    }
    if "umbral_basura" in info:
        labels["umbral_basura"] = info["umbral_basura"]
        labels["idx_basura"] = info["idx_basura"]
        labels["nota_umbral"] = (
            "Este modelo NO usa argmax: es basura si prob[idx_basura] >= umbral_basura."
        )

    with open(salida.with_suffix(".labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"OK: {info['nombre']} -> {salida} ({salida.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    for info in MODELOS:
        exportar(info)

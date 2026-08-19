"""
Carga cualquiera de los pesos guardados en scriptsPython/pesos/ y clasifica una
imagen suelta. No entrena nada, no importa Optuna ni depende del dataset de TrashNet:
es el artefacto que se usaría en producción para consumir un modelo ya entrenado.

Sirve para los tres modelos de scriptsPython/modelos/ (mobilenetv3, jerarquico,
binario) porque comparten la misma arquitectura (TrashNetMobileNetV3); el checkpoint
trae las clases y los hiperparámetros necesarios para reconstruirla.

Uso:
    python predecir.py --modelo ../pesos/mobilenetv3_optuna.pt --imagen foto.jpg
    python predecir.py --modelo ../pesos/jerarquico_optuna.pt --imagen foto.jpg
    python predecir.py --modelo ../pesos/binario_optuna.pt --imagen foto.jpg --umbral 0.837
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# parents[1] = scriptsPython/ (este archivo vive en scriptsPython/consumo/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import utils as u


def cargar_modelo(ruta_pesos):
    checkpoint = torch.load(ruta_pesos, map_location=u.DEVICE)

    classes = checkpoint["classes"]
    model = u.TrashNetMobileNetV3(
        dense_units=checkpoint["dense_units"],
        dropout_rate=checkpoint["dropout_rate"],
        n_classes=len(classes),
    ).to(u.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    img_size = tuple(checkpoint.get("img_size", u.IMG_SIZE))
    return model, classes, img_size


def preparar_imagen(ruta_imagen, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(u.MEAN, u.STD),
    ])
    imagen = Image.open(ruta_imagen).convert("RGB")
    return transform(imagen).unsqueeze(0)


@torch.no_grad()
def predecir(model, tensor_imagen):
    tensor_imagen = tensor_imagen.to(u.DEVICE)
    probs = torch.softmax(model(tensor_imagen), dim=1).cpu().numpy()[0]
    return probs


def main():
    args = parse_args()

    if not os.path.isfile(args.modelo):
        raise SystemExit(f"No existe el archivo de pesos: {args.modelo}")
    if not os.path.isfile(args.imagen):
        raise SystemExit(f"No existe la imagen: {args.imagen}")

    model, classes, img_size = cargar_modelo(args.modelo)
    tensor_imagen = preparar_imagen(args.imagen, img_size)
    probs = predecir(model, tensor_imagen)

    print(f"Modelo: {args.modelo}")
    print(f"Clases: {classes}")
    print(f"Imagen: {args.imagen}\n")

    orden = np.argsort(probs)[::-1]
    print(f"{'clase':<14}{'probabilidad':>14}")
    print("-" * 28)
    for i in orden:
        marca = "  <-- predicción" if i == orden[0] else ""
        print(f"{classes[i]:<14}{probs[i]:>14.4f}{marca}")

    # Caso especial: modelo binario con umbral calibrado, en vez de argmax.
    if args.umbral is not None:
        if "basura" not in classes:
            print("\nAviso: --umbral solo tiene sentido con el modelo binario "
                  "(clases 'basura'/'reciclable'); se ignora.")
        else:
            idx_basura = classes.index("basura")
            p_basura = probs[idx_basura]
            decision = "basura" if p_basura >= args.umbral else "reciclable"
            print(f"\nCon umbral calibrado {args.umbral:.3f}: "
                  f"P(basura)={p_basura:.4f} -> {decision}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True, help="ruta al .pt (scriptsPython/pesos/...)")
    p.add_argument("--imagen", required=True, help="ruta a la imagen a clasificar")
    p.add_argument("--umbral", type=float, default=None,
                   help="umbral de P(basura) para el modelo binario (si no se da, usa argmax)")
    return p.parse_args()


if __name__ == "__main__":
    main()

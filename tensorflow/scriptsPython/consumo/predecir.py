"""
Script de consumo independiente (punto 3 de la pauta) — versión TensorFlow.

Carga cualquiera de los .keras guardados en scriptsPython/pesos/ y clasifica una
imagen suelta. No entrena nada, no depende de Optuna ni del dataset de TrashNet.

Uso:
    python predecir.py --modelo ../pesos/mobilenetv3_optuna.keras --imagen foto.jpg
    python predecir.py --modelo ../pesos/jerarquico_optuna.keras --imagen foto.jpg --clases amarillo azul gris verde
    python predecir.py --modelo ../pesos/binario_optuna.keras --imagen foto.jpg --binario --umbral 0.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# parents[1] = scriptsPython/ (este archivo vive en scriptsPython/consumo/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import utils as u

CLASSES_6 = u.ORIGINAL_CLASSES
CLASSES_JERARQUICO = ["amarillo", "azul", "gris", "verde"]
CLASSES_BINARIO = ["basura", "reciclable"]


def preparar_imagen(ruta_imagen, img_size):
    datos = tf.io.read_file(ruta_imagen)
    imagen = tf.io.decode_image(datos, channels=3, expand_animations=False)
    imagen.set_shape([None, None, 3])
    imagen = tf.image.resize(imagen, img_size)
    imagen = tf.cast(imagen, tf.float32)
    return tf.expand_dims(imagen, axis=0)


def main():
    args = parse_args()

    if not os.path.isdir(args.modelo) and not os.path.isfile(args.modelo):
        raise SystemExit(f"No existe el modelo: {args.modelo}")
    if not os.path.isfile(args.imagen):
        raise SystemExit(f"No existe la imagen: {args.imagen}")

    model = keras.models.load_model(args.modelo)
    tensor_imagen = preparar_imagen(args.imagen, u.IMG_SIZE)

    print(f"Modelo: {args.modelo}")
    print(f"Imagen: {args.imagen}\n")

    if args.binario:
        p_reciclable = float(model.predict(tensor_imagen, verbose=0).reshape(-1)[0])
        p_basura = 1.0 - p_reciclable
        umbral = args.umbral if args.umbral is not None else 0.5
        decision = "basura" if p_basura >= umbral else "reciclable"
        print(f"P(basura)={p_basura:.4f}  P(reciclable)={p_reciclable:.4f}")
        print(f"Umbral usado: {umbral:.3f} -> {decision}")
        return

    classes = args.clases if args.clases else CLASSES_6
    probs = model.predict(tensor_imagen, verbose=0)[0]

    if len(probs) != len(classes):
        raise SystemExit(
            f"El modelo tiene {len(probs)} salidas pero se dieron {len(classes)} nombres de "
            f"clase (--clases). Pasa --clases con la lista correcta, p.ej. para el modelo "
            f"jerárquico: --clases amarillo azul gris verde"
        )

    orden = np.argsort(probs)[::-1]
    print(f"{'clase':<14}{'probabilidad':>14}")
    print("-" * 28)
    for i in orden:
        marca = "  <-- predicción" if i == orden[0] else ""
        print(f"{classes[i]:<14}{probs[i]:>14.4f}{marca}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True, help="ruta al .keras (scriptsPython/pesos/...)")
    p.add_argument("--imagen", required=True, help="ruta a la imagen a clasificar")
    p.add_argument("--clases", nargs="+", default=None,
                   help="nombres de clase en el orden de salida del modelo (default: las 6 de material)")
    p.add_argument("--binario", action="store_true",
                   help="el modelo es el binario (salida sigmoide P(reciclable))")
    p.add_argument("--umbral", type=float, default=None,
                   help="umbral de P(basura) para el modelo binario (default 0.5)")
    return p.parse_args()


if __name__ == "__main__":
    main()

"""
Clasificación en vivo desde la cámara con OpenCV (punto 4 de la pauta, mandatorio)
— versión TensorFlow.

Abre la webcam, captura frames, y sobre cada frame corre el modelo Keras indicado
(cualquiera de los .keras guardados en scriptsPython/pesos/).

Controles:
    q  -  salir
    c  -  capturar el frame actual y guardarlo en scriptsPython/camara/capturas/

Uso:
    python capturar_camara.py --modelo ../pesos/mobilenetv3_optuna.keras
    python capturar_camara.py --modelo ../pesos/binario_optuna.keras --binario --umbral 0.5
    python capturar_camara.py --modelo ../pesos/jerarquico_optuna.keras --clases amarillo azul gris verde
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# parents[1] = scriptsPython/ (este archivo vive en scriptsPython/camara/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import utils as u

CLASSES_6 = u.ORIGINAL_CLASSES
CAPTURAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capturas")


def preparar_frame(frame_bgr, img_size):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor_imagen = tf.image.resize(frame_rgb, img_size)
    tensor_imagen = tf.cast(tensor_imagen, tf.float32)
    return tf.expand_dims(tensor_imagen, axis=0)


def clasificar_frame(model, frame_bgr, binario):
    tensor_imagen = preparar_frame(frame_bgr, u.IMG_SIZE)
    salida = model.predict(tensor_imagen, verbose=0)
    if binario:
        p_reciclable = float(salida.reshape(-1)[0])
        return np.array([1.0 - p_reciclable, p_reciclable])   # [P(basura), P(reciclable)]
    return salida[0]


def dibujar_resultado(frame, classes, probs, binario, umbral):
    if binario:
        p_basura = probs[0]
        umbral = umbral if umbral is not None else 0.5
        etiqueta = "basura" if p_basura >= umbral else "reciclable"
        texto = f"{etiqueta.upper()}  P(basura)={p_basura:.2f}  umbral={umbral:.2f}"
        color = (0, 0, 255) if etiqueta == "basura" else (0, 200, 0)
    else:
        idx = int(np.argmax(probs))
        texto = f"{classes[idx].upper()}  {probs[idx]:.1%}"
        color = (255, 180, 0)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (30, 30, 30), -1)
    cv2.putText(frame, texto, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def main():
    args = parse_args()

    if not os.path.isdir(args.modelo) and not os.path.isfile(args.modelo):
        raise SystemExit(f"No existe el modelo: {args.modelo}")

    print(f"Cargando modelo: {args.modelo}")
    model = keras.models.load_model(args.modelo)
    classes = args.clases if args.clases else CLASSES_6
    print(f"Clases: {['basura', 'reciclable'] if args.binario else classes}")

    captura = cv2.VideoCapture(args.camara)
    if not captura.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camara}")

    print("Presiona 'q' para salir, 'c' para capturar el frame actual.")
    ultimo_calculo = 0.0
    probs = None

    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                print("No se pudo leer un frame de la cámara.")
                break

            ahora = time.time()
            if ahora - ultimo_calculo >= args.intervalo:
                probs = clasificar_frame(model, frame, args.binario)
                ultimo_calculo = ahora

            if probs is not None:
                frame = dibujar_resultado(frame, classes, probs, args.binario, args.umbral)

            cv2.imshow("TrashNet — clasificación en vivo (TensorFlow)", frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break
            if tecla == ord("c"):
                os.makedirs(CAPTURAS_DIR, exist_ok=True)
                ruta = os.path.join(CAPTURAS_DIR, f"captura_{int(time.time())}.png")
                cv2.imwrite(ruta, frame)
                print(f"Captura guardada: {ruta}")
    finally:
        captura.release()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True, help="ruta al .keras (scriptsPython/pesos/...)")
    p.add_argument("--camara", type=int, default=0, help="índice de la cámara (default 0)")
    p.add_argument("--clases", nargs="+", default=None,
                   help="nombres de clase en el orden de salida del modelo (default: las 6 de material)")
    p.add_argument("--binario", action="store_true", help="el modelo es el binario (salida sigmoide)")
    p.add_argument("--umbral", type=float, default=None, help="umbral de P(basura) para el binario (default 0.5)")
    p.add_argument("--intervalo", type=float, default=0.3,
                   help="segundos entre inferencias, para no saturar la CPU/GPU (default 0.3)")
    return p.parse_args()


if __name__ == "__main__":
    main()

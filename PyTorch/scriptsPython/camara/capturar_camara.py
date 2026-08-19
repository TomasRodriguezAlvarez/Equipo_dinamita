"""
Clasificación en vivo desde la cámara con OpenCV (punto 4 de la pauta, mandatorio).

Abre la webcam, captura frames, y sobre cada frame corre el mismo modelo que
consumo/predecir.py — cualquiera de los .pt guardados en scriptsPython/pesos/.
Muestra la predicción y su probabilidad superpuestas en la ventana de video.

Controles:
    q  -  salir
    c  -  capturar el frame actual y guardarlo en scriptsPython/camara/capturas/

Uso:
    python capturar_camara.py --modelo ../pesos/mobilenetv3_optuna.pt
    python capturar_camara.py --modelo ../pesos/binario_optuna.pt --umbral 0.837
    python capturar_camara.py --modelo ../pesos/jerarquico_optuna.pt --camara 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

# parents[1] = scriptsPython/ (este archivo vive en scriptsPython/camara/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import utils as u

CAPTURAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capturas")


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


def construir_transform(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(u.MEAN, u.STD),
    ])


@torch.no_grad()
def clasificar_frame(model, transform, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor_imagen = transform(frame_rgb).unsqueeze(0).to(u.DEVICE)
    probs = torch.softmax(model(tensor_imagen), dim=1).cpu().numpy()[0]
    return probs


def dibujar_resultado(frame, classes, probs, umbral=None):
    idx_basura = classes.index("basura") if "basura" in classes else None

    if idx_basura is not None and umbral is not None:
        p_basura = probs[idx_basura]
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

    if not os.path.isfile(args.modelo):
        raise SystemExit(f"No existe el archivo de pesos: {args.modelo}")

    print(f"Cargando modelo: {args.modelo}")
    model, classes, img_size = cargar_modelo(args.modelo)
    transform = construir_transform(img_size)
    print(f"Clases: {classes}")

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
                probs = clasificar_frame(model, transform, frame)
                ultimo_calculo = ahora

            if probs is not None:
                frame = dibujar_resultado(frame, classes, probs, args.umbral)

            cv2.imshow("TrashNet — clasificación en vivo", frame)
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
    p.add_argument("--modelo", required=True, help="ruta al .pt (scriptsPython/pesos/...)")
    p.add_argument("--camara", type=int, default=0, help="índice de la cámara (default 0)")
    p.add_argument("--umbral", type=float, default=None,
                   help="umbral de P(basura) para el modelo binario (si no se da, usa argmax)")
    p.add_argument("--intervalo", type=float, default=0.3,
                   help="segundos entre inferencias, para no saturar la CPU/GPU (default 0.3)")
    return p.parse_args()


if __name__ == "__main__":
    main()

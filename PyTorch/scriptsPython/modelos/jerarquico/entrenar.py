"""
Modelo 2 de la segunda entrega — Clasificador jerárquico por contenedor, 4 clases.

Versión script de `PyTorch/jerarquico/TrashNet_PyTorch_Jerarquico_DEFINITIVO.ipynb`.
Mismo backbone y protocolo que el modelo de 6 clases, pero las 6 clases de material se
reasignan a 4 destinos de contenedor con un `target_transform` de ImageFolder (no se
duplican ni mueven imágenes):

    plastic + metal    -> amarillo
    paper + cardboard   -> azul
    trash                -> gris
    glass                -> verde

El reagrupamiento empeora el desbalance ("gris" queda con solo `trash`, ~95 imágenes,
contra ~697 de "azul"), así que la loss usa class_weight.

Uso:
    python entrenar.py
    python entrenar.py --sin-optuna

Genera en scriptsPython/pesos/: jerarquico_sin_optuna.pt, jerarquico_optuna.pt
Y en scriptsPython/resultados/: gráficos, matriz de confusión y reporte con prefijo `jer_`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# parents[2] = scriptsPython/ (este archivo vive en scriptsPython/modelos/jerarquico/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comun import utils as u

MAPEO_NOMBRE = {
    "cardboard": "azul", "glass": "verde", "metal": "amarillo",
    "paper": "azul", "plastic": "amarillo", "trash": "gris",
}
CLASSES = ["amarillo", "azul", "gris", "verde"]     # orden alfabético
N_CLASSES = len(CLASSES)
ORIGINAL_TO_CONTAINER = [CLASSES.index(MAPEO_NOMBRE[c]) for c in u.ORIGINAL_CLASSES]

BATCH_SIZE = 32
EPOCHS_FASE1 = 25
EPOCHS_FASE2 = 15
FINE_TUNE_LR = 1e-5
NUM_WORKERS = 0

BASE_DENSE_UNITS = 128
BASE_DROPOUT = 0.35
BASE_LR = 3e-4
BASE_OPTIMIZER = "adam"


def remapear_a_contenedor(indice_original):
    return ORIGINAL_TO_CONTAINER[indice_original]


def cargar_datos(workers):
    train_tf, eval_tf = u.crear_transforms(u.IMG_SIZE)

    def cargar(ruta, transform):
        ds = datasets.ImageFolder(ruta, transform=transform, target_transform=remapear_a_contenedor)
        assert ds.classes == u.ORIGINAL_CLASSES, f"Orden de clases inesperado en {ruta}: {ds.classes}"
        return ds

    train_ds = cargar(u.TRAIN_DIR, train_tf)
    valid_ds = cargar(u.VALID_DIR, eval_tf)
    test_ds  = cargar(u.TEST_DIR,  eval_tf)

    comun = dict(num_workers=workers, pin_memory=(u.DEVICE.type == "cuda"),
                 persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              generator=u.generador, **comun)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, **comun)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **comun)

    conteos_originales = u.contar_imagenes(u.TRAIN_DIR, u.ORIGINAL_CLASSES)
    conteos = np.zeros(N_CLASSES)
    for c, n in conteos_originales.items():
        conteos[remapear_a_contenedor(u.ORIGINAL_CLASSES.index(c))] += n
    pesos = conteos.sum() / (N_CLASSES * conteos)
    class_weights_tensor = torch.tensor(pesos, dtype=torch.float32, device=u.DEVICE)

    print(f"Clases: {CLASSES}")
    print(f"Pesos por clase: {dict(zip(CLASSES, pesos.round(3)))}")
    print(f"Train: {len(train_ds)} | Validation: {len(valid_ds)} | Test: {len(test_ds)}")
    return train_loader, valid_loader, test_loader, class_weights_tensor


def construir(dense_units, dropout, optimizer_name, lr, nombre, class_weights):
    return u.construir_modelo(dense_units, dropout, optimizer_name, lr,
                              N_CLASSES, nombre, class_weights)


def entrenar_dos_fases(model, optimizer, criterion, train_loader, valid_loader):
    h1 = u.entrenar(model, optimizer, criterion, train_loader, valid_loader,
                    epochs=EPOCHS_FASE1, patience_es=6, patience_lr=3, min_lr=1e-6)
    u.descongelar_ultimos_bloques(model, n_bloques=3)
    optimizer2 = u.construir_optimizador(model, BASE_OPTIMIZER, FINE_TUNE_LR)
    h2 = u.entrenar(model, optimizer2, criterion, train_loader, valid_loader,
                    epochs=EPOCHS_FASE2, patience_es=5, patience_lr=2, min_lr=1e-7)
    return u.unir_historiales(h1, h2)


def buscar_con_optuna(train_loader, valid_loader, class_weights, args):
    import optuna

    def objective(trial):
        u.semilla()
        u.limpiar_memoria()

        dense_units = trial.suggest_categorical("dense_units", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout_rate", 0.20, 0.50, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])

        model, optimizer, criterion = construir(
            dense_units, dropout_rate, optimizer_name, learning_rate,
            f"trial_{trial.number}", class_weights
        )
        history = u.entrenar(model, optimizer, criterion, train_loader, valid_loader,
                             epochs=args.epocas_busqueda, patience_es=3, patience_lr=2,
                             min_lr=1e-7, verbose=0)
        mejor = float(max(history["val_accuracy"]))
        print(f"Trial {trial.number:02d} | val_accuracy={mejor:.4f} | {trial.params}")

        del model, optimizer
        u.limpiar_memoria()
        return mejor

    params_baseline = {
        "dense_units": BASE_DENSE_UNITS, "dropout_rate": BASE_DROPOUT,
        "learning_rate": BASE_LR, "optimizer": BASE_OPTIMIZER,
    }

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=args.startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="TrashNet_Jerarquico_scriptsPython")
    study.enqueue_trial(params_baseline)

    print(f"\n{args.trials} trials ({args.startup_trials} de arranque aleatorio + "
          f"{args.trials - args.startup_trials} informados por el TPE)\n")
    study.optimize(objective, n_trials=args.trials)
    return study, params_baseline


def main():
    args = parse_args()
    u.crear_carpetas_resultados()
    u.semilla()

    print(f"Dispositivo: {u.DEVICE}")
    train_loader, valid_loader, test_loader, class_weights = cargar_datos(args.workers)

    print("\n" + "=" * 70)
    print("PARTE A — Modelo jerárquico base (sin Optuna)")
    print("=" * 70)
    model_base, optimizer_base, criterion = construir(
        BASE_DENSE_UNITS, BASE_DROPOUT, BASE_OPTIMIZER, BASE_LR,
        "jerarquico_sin_optuna", class_weights
    )
    u.resumen_modelo(model_base)
    hist_base = entrenar_dos_fases(model_base, optimizer_base, criterion,
                                   train_loader, valid_loader)

    ruta_base = os.path.join(u.PESOS_DIR, "jerarquico_sin_optuna.pt")
    torch.save({
        "state_dict": model_base.state_dict(), "dense_units": BASE_DENSE_UNITS,
        "dropout_rate": BASE_DROPOUT, "classes": CLASSES, "mapeo": MAPEO_NOMBRE,
        "img_size": u.IMG_SIZE,
    }, ruta_base)
    print(f"\n  {ruta_base}")

    if args.sin_optuna:
        evaluar_y_reportar(model_base, model_base, hist_base, hist_base, test_loader, args)
        return

    print("\n" + "=" * 70)
    print("PARTE B — Búsqueda de hiperparámetros con Optuna")
    print("=" * 70)
    study, params_baseline = buscar_con_optuna(train_loader, valid_loader, class_weights, args)

    def construir_desde_params(params):
        return construir(params["dense_units"], params["dropout_rate"],
                         params["optimizer"], params["learning_rate"],
                         "candidato", class_weights)

    def entrenar_fase(model, optimizer, criterion, epochs, patience_es, patience_lr, min_lr, verbose=1):
        return u.entrenar(model, optimizer, criterion, train_loader, valid_loader,
                          epochs, patience_es, patience_lr, min_lr, verbose=verbose)

    resultados_topk = u.reentrenar_topk(
        study, args.top_k, construir_desde_params, entrenar_fase,
        EPOCHS_FASE1, EPOCHS_FASE2, FINE_TUNE_LR, patience_lr=3,
    )

    ranking = sorted(resultados_topk, key=lambda r: r["valor_completo"], reverse=True)
    ganador = ranking[0]

    print("\nTOP-K REENTRENADO A PRESUPUESTO COMPLETO")
    for r in ranking:
        print(f"  trial {r['trial']:>3}  búsqueda={r['valor_busqueda']:.4f}  "
              f"completo={r['valor_completo']:.4f}  {r['params']}")

    model_optuna = u.TrashNetMobileNetV3(
        ganador["params"]["dense_units"], ganador["params"]["dropout_rate"], N_CLASSES
    ).to(u.DEVICE)
    model_optuna.load_state_dict(ganador["state_dict"])
    model_optuna.nombre = "jerarquico_optuna"
    u.descongelar_ultimos_bloques(model_optuna, n_bloques=3)
    hist_optuna = u.unir_historiales(ganador["history_fase1"], ganador["history_fase2"])

    print(f"\nGANADOR: trial {ganador['trial']} — {ganador['params']}")
    if ganador["params"] == params_baseline:
        print("AVISO: el ganador ES la configuración del baseline.")

    ruta_optuna = os.path.join(u.PESOS_DIR, "jerarquico_optuna.pt")
    torch.save({
        "state_dict": model_optuna.state_dict(), **ganador["params"],
        "trial": ganador["trial"], "classes": CLASSES, "mapeo": MAPEO_NOMBRE,
        "img_size": u.IMG_SIZE,
    }, ruta_optuna)
    print(f"  {ruta_optuna}")

    evaluar_y_reportar(model_base, model_optuna, hist_base, hist_optuna, test_loader, args)


def evaluar_y_reportar(model_base, model_optuna, hist_base, hist_optuna, test_loader, args):
    print("\n" + "=" * 70)
    print("PARTE C — Evaluación en test")
    print("=" * 70)

    eval_base = u.evaluar(model_base, test_loader)
    eval_optuna = u.evaluar(model_optuna, test_loader)
    y_true = eval_base["y_true"]

    def metricas(ev):
        cm = u.confusion_matrix(y_true, ev["y_pred"], N_CLASSES)
        por_clase = u.metricas_por_clase(cm)
        return {
            "Accuracy": u.accuracy_score(y_true, ev["y_pred"]),
            "Precision macro": u.macro(por_clase, "precision"),
            "Recall macro": u.macro(por_clase, "recall"),
            "F1 macro": u.macro(por_clase, "f1"),
            "ROC AUC macro OVR": u.roc_auc_macro(y_true, ev["y_prob"], N_CLASSES),
        }, cm

    m_base, cm_base = metricas(eval_base)
    m_optuna, cm_optuna = metricas(eval_optuna)

    ancho = max(len(k) for k in m_base) + 2
    print(f"{'':<{ancho}}{'Sin Optuna':>14}{'Con Optuna':>14}")
    for k in m_base:
        print(f"{k:<{ancho}}{m_base[k]:>14.4f}{m_optuna[k]:>14.4f}")

    u.graficar_curvas(hist_base, "Jerárquico sin Optuna",
                      os.path.join(u.GRAFICOS_DIR, "jer_curvas_sin_optuna.png"))
    u.graficar_curvas(hist_optuna, "Jerárquico con Optuna",
                      os.path.join(u.GRAFICOS_DIR, "jer_curvas_optuna.png"))

    u.graficar_matriz_confusion(cm_base, CLASSES, "Matriz de confusión — Sin Optuna",
                                os.path.join(u.CM_DIR, "jer_confusion_sin_optuna.png"))
    u.graficar_matriz_confusion(cm_optuna, CLASSES, "Matriz de confusión — Con Optuna",
                                os.path.join(u.CM_DIR, "jer_confusion_optuna.png"))

    u.graficar_roc_por_clase(y_true, eval_base["y_prob"], CLASSES,
                             "Curvas ROC — Sin Optuna",
                             os.path.join(u.GRAFICOS_DIR, "jer_roc_sin_optuna.png"))
    u.graficar_roc_por_clase(y_true, eval_optuna["y_prob"], CLASSES,
                             "Curvas ROC — Con Optuna",
                             os.path.join(u.GRAFICOS_DIR, "jer_roc_optuna.png"))

    reporte = ["TrashNet — Jerárquico por contenedor (PyTorch, scriptsPython) — Reporte",
              "=" * 70, f"Dispositivo: {u.DEVICE}", f"Mapeo: {MAPEO_NOMBRE}", ""]
    for nombre, m in [("SIN OPTUNA", m_base), ("CON OPTUNA", m_optuna)]:
        reporte.append(nombre)
        reporte += [f"  {k:<20}: {v:.4f}" for k, v in m.items()]
        reporte.append("")
    reporte.append(u.classification_report(y_true, eval_optuna["y_pred"], CLASSES))

    ruta_reporte = os.path.join(u.REPORTE_DIR, "jer_reporte.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write("\n".join(reporte) + "\n")
    print(f"\n  {ruta_reporte}")
    print("\nListo. Resultados en scriptsPython/resultados/ y pesos en scriptsPython/pesos/")


def parse_args():
    p = argparse.ArgumentParser(description="Entrena el modelo jerárquico de 4 clases (contenedor)")
    p.add_argument("--sin-optuna", action="store_true", help="entrenar solo el baseline")
    p.add_argument("--trials", type=int, default=30, help="trials de Optuna (default 30)")
    p.add_argument("--startup-trials", type=int, default=10, help="trials de arranque aleatorio (default 10)")
    p.add_argument("--epocas-busqueda", type=int, default=8, help="épocas por trial (default 8)")
    p.add_argument("--top-k", type=int, default=3, help="mejores trials a reentrenar (default 3)")
    p.add_argument("--workers", type=int, default=NUM_WORKERS, help="num_workers del DataLoader")
    return p.parse_args()


if __name__ == "__main__":
    main()

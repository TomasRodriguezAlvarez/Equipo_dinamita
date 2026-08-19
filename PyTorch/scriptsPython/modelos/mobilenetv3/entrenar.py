"""
Modelo 1 de la segunda entrega — MobileNetV3Small, 6 clases de material
(cardboard, glass, metal, paper, plastic, trash).

Versión script del notebook `PyTorch/TrashNet_PyTorch_MobileNetV3_DEFINITIVO.ipynb`.
Transfer learning en 2 fases (backbone congelado -> fine-tuning de los últimos 3
bloques) + búsqueda de hiperparámetros con Optuna, siguiendo la metodología corregida
de PyTorch/CONTEXT.md §9 (baseline encolado como trial 0, top-K reentrenado a
presupuesto completo).

Uso:
    python entrenar.py                    # baseline + búsqueda Optuna + comparación
    python entrenar.py --sin-optuna       # solo el baseline, sin búsqueda
    python entrenar.py --trials 20 --epocas-busqueda 6

Genera en scriptsPython/pesos/:
    mobilenetv3_sin_optuna.pt
    mobilenetv3_optuna.pt

Y en scriptsPython/resultados/:
    graficos/mnv3_curvas_{sin_optuna,optuna}.png
    graficos/mnv3_optuna_historia.png
    graficos/mnv3_roc_{sin_optuna,optuna}.png
    matriz_de_confusion/mnv3_confusion_{sin_optuna,optuna}.png
    reporte/mnv3_reporte.txt
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

# parents[2] = scriptsPython/ (este archivo vive en scriptsPython/modelos/mobilenetv3/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comun import utils as u

CLASSES = u.ORIGINAL_CLASSES
N_CLASSES = len(CLASSES)

BATCH_SIZE = 32
EPOCHS_FASE1 = 25
EPOCHS_FASE2 = 15
FINE_TUNE_LR = 1e-5
NUM_WORKERS = 0

BASE_DENSE_UNITS = 128
BASE_DROPOUT = 0.35
BASE_LR = 3e-4
BASE_OPTIMIZER = "adam"


def cargar_datos(workers):
    train_tf, eval_tf = u.crear_transforms(u.IMG_SIZE)

    train_ds = datasets.ImageFolder(u.TRAIN_DIR, transform=train_tf)
    valid_ds = datasets.ImageFolder(u.VALID_DIR, transform=eval_tf)
    test_ds  = datasets.ImageFolder(u.TEST_DIR,  transform=eval_tf)
    assert train_ds.classes == CLASSES, f"Orden de clases inesperado: {train_ds.classes}"

    comun = dict(num_workers=workers, pin_memory=(u.DEVICE.type == "cuda"),
                 persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              generator=u.generador, **comun)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, **comun)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **comun)

    conteos_por_clase = u.contar_imagenes(u.TRAIN_DIR, CLASSES)
    conteos = np.array([conteos_por_clase[c] for c in CLASSES], dtype=np.float64)
    pesos = conteos.sum() / (N_CLASSES * conteos)
    class_weights_tensor = torch.tensor(pesos, dtype=torch.float32, device=u.DEVICE)

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
                                study_name="TrashNet_MobileNetV3_scriptsPython")
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

    # ── Parte A: baseline sin Optuna ────────────────────────────────
    print("\n" + "=" * 70)
    print("PARTE A — Modelo base (sin Optuna)")
    print("=" * 70)
    model_base, optimizer_base, criterion = construir(
        BASE_DENSE_UNITS, BASE_DROPOUT, BASE_OPTIMIZER, BASE_LR,
        "mobilenetv3_sin_optuna", class_weights
    )
    u.resumen_modelo(model_base)
    hist_base = entrenar_dos_fases(model_base, optimizer_base, criterion,
                                   train_loader, valid_loader)

    ruta_base = os.path.join(u.PESOS_DIR, "mobilenetv3_sin_optuna.pt")
    torch.save({
        "state_dict": model_base.state_dict(), "dense_units": BASE_DENSE_UNITS,
        "dropout_rate": BASE_DROPOUT, "classes": CLASSES, "img_size": u.IMG_SIZE,
    }, ruta_base)
    print(f"\n  {ruta_base}")

    if args.sin_optuna:
        evaluar_y_reportar(model_base, model_base, hist_base, hist_base, test_loader)
        return

    # ── Parte B: búsqueda con Optuna ────────────────────────────────
    print("\n" + "=" * 70)
    print("PARTE B — Búsqueda de hiperparámetros con Optuna")
    print("=" * 70)
    study, params_baseline = buscar_con_optuna(train_loader, valid_loader, class_weights, args)

    _graficar_historia_optuna(study, args)

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
    model_optuna.nombre = "mobilenetv3_optuna"
    u.descongelar_ultimos_bloques(model_optuna, n_bloques=3)
    hist_optuna = u.unir_historiales(ganador["history_fase1"], ganador["history_fase2"])

    print(f"\nGANADOR: trial {ganador['trial']} — {ganador['params']}")
    if ganador["params"] == params_baseline:
        print("AVISO: el ganador ES la configuración del baseline.")

    ruta_optuna = os.path.join(u.PESOS_DIR, "mobilenetv3_optuna.pt")
    torch.save({
        "state_dict": model_optuna.state_dict(), **ganador["params"],
        "trial": ganador["trial"], "classes": CLASSES, "img_size": u.IMG_SIZE,
    }, ruta_optuna)
    print(f"  {ruta_optuna}")

    evaluar_y_reportar(model_base, model_optuna, hist_base, hist_optuna, test_loader)


def _graficar_historia_optuna(study, args):
    import optuna
    import matplotlib.pyplot as plt

    completos = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    numeros = [t.number for t in completos]
    valores = [t.value for t in completos]

    plt.figure(figsize=(10, 5))
    plt.scatter(numeros, valores, label="val_accuracy del trial", zorder=3)
    plt.plot(numeros, np.maximum.accumulate(valores), label="Mejor hasta el momento", zorder=2)
    plt.axvline(args.startup_trials - 0.5, linestyle="--", color="gray",
               label=f"Fin del arranque aleatorio ({args.startup_trials} trials)")
    plt.xlabel("Trial"); plt.ylabel("val_accuracy")
    plt.title("Historia de la optimización — MobileNetV3")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(u.GRAFICOS_DIR, "mnv3_optuna_historia.png")
    plt.savefig(ruta, dpi=120)
    plt.close()
    print(f"  {ruta}")


def evaluar_y_reportar(model_base, model_optuna, hist_base, hist_optuna, test_loader):
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

    u.graficar_curvas(hist_base, "MobileNetV3 sin Optuna",
                      os.path.join(u.GRAFICOS_DIR, "mnv3_curvas_sin_optuna.png"))
    u.graficar_curvas(hist_optuna, "MobileNetV3 con Optuna",
                      os.path.join(u.GRAFICOS_DIR, "mnv3_curvas_optuna.png"))

    u.graficar_matriz_confusion(cm_base, CLASSES, "Matriz de confusión — Sin Optuna",
                                os.path.join(u.CM_DIR, "mnv3_confusion_sin_optuna.png"))
    u.graficar_matriz_confusion(cm_optuna, CLASSES, "Matriz de confusión — Con Optuna",
                                os.path.join(u.CM_DIR, "mnv3_confusion_optuna.png"))

    u.graficar_roc_por_clase(y_true, eval_base["y_prob"], CLASSES,
                             "Curvas ROC — Sin Optuna",
                             os.path.join(u.GRAFICOS_DIR, "mnv3_roc_sin_optuna.png"))
    u.graficar_roc_por_clase(y_true, eval_optuna["y_prob"], CLASSES,
                             "Curvas ROC — Con Optuna",
                             os.path.join(u.GRAFICOS_DIR, "mnv3_roc_optuna.png"))

    reporte = ["TrashNet — MobileNetV3Small (PyTorch, scriptsPython) — Reporte", "=" * 70,
              f"Dispositivo: {u.DEVICE}", ""]
    for nombre, m in [("SIN OPTUNA", m_base), ("CON OPTUNA", m_optuna)]:
        reporte.append(nombre)
        reporte += [f"  {k:<20}: {v:.4f}" for k, v in m.items()]
        reporte.append("")
    reporte.append(u.classification_report(y_true, eval_optuna["y_pred"], CLASSES))

    ruta_reporte = os.path.join(u.REPORTE_DIR, "mnv3_reporte.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write("\n".join(reporte) + "\n")
    print(f"\n  {ruta_reporte}")
    print("\nListo. Resultados en scriptsPython/resultados/ y pesos en scriptsPython/pesos/")


def parse_args():
    p = argparse.ArgumentParser(description="Entrena el modelo MobileNetV3 de 6 clases")
    p.add_argument("--sin-optuna", action="store_true", help="entrenar solo el baseline")
    p.add_argument("--trials", type=int, default=30, help="trials de Optuna (default 30)")
    p.add_argument("--startup-trials", type=int, default=10, help="trials de arranque aleatorio (default 10)")
    p.add_argument("--epocas-busqueda", type=int, default=8, help="épocas por trial (default 8)")
    p.add_argument("--top-k", type=int, default=3, help="mejores trials a reentrenar (default 3)")
    p.add_argument("--workers", type=int, default=NUM_WORKERS, help="num_workers del DataLoader")
    return p.parse_args()


if __name__ == "__main__":
    main()

"""
Reentrenamiento final del modelo TrashNet (PyTorch) con los mejores hiperparámetros
encontrados por Optuna, y evaluación completa sobre el conjunto de TEST.

Flujo:
  1. Lee PyTorch/optuna/mejores_params.json (lo escribe optuna_trashnet.py).
  2. Entrena el modelo más épocas que en la búsqueda, con EarlyStopping sobre
     validación y quedándose con los pesos de la mejor época.
  3. Evalúa UNA sola vez en test: accuracy, matriz de confusión, precision/recall/F1
     por clase y curvas ROC one-vs-rest con AUC.
  4. Guarda el modelo, los gráficos y un reporte de texto.

Por qué se separa de la búsqueda: durante los trials solo se mira validación. El
test se reserva para esta única evaluación final, si no la métrica reportada estaría
contaminada por la propia selección de hiperparámetros.

Uso:
    python entrenar_mejor.py                        # usa mejores_params.json
    python entrenar_mejor.py --epocas 60 --patience 12
    python entrenar_mejor.py --params otro.json
    python entrenar_mejor.py --sin-optuna           # entrena con los HP del notebook (baseline)

Genera:
    PyTorch/modelos/trashnet_pytorch_optuna.pt                        — pesos + class_names + hiperparámetros
    PyTorch/resultados/matriz_de_confusion/confusion_matrix_optuna.png
    PyTorch/resultados/graficos/roc_curves_optuna.png
    PyTorch/resultados/graficos/curvas_entrenamiento_optuna.png
    PyTorch/resultados/reporte/reporte_optuna.txt

(con --sin-optuna el sufijo pasa a ser _baseline, así los dos quedan lado a lado)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from trashnet_comun import (
    DEVICE, MODELOS_DIR, OPTUNA_DIR, GRAFICOS_DIR, CM_DIR, REPORTE_DIR,
    crear_carpetas_resultados,
    construir_loaders, construir_modelo, construir_optimizador,
    correr_epoca, obtener_predicciones,
    calcular_cm, calcular_metricas_por_clase,
    plot_matriz_confusion, plot_roc, plot_curvas_entrenamiento,
    pesos_de_clase, semilla,
)

SEED = 42

# Hiperparámetros del notebook original: sirven de baseline para comparar
# contra lo que encuentre Optuna.
PARAMS_BASELINE = {
    "n_bloques": 3, "filtros_base": 16, "units_fc": 128, "activacion": "relu",
    "dropout": 0.3, "usar_bn": False, "weight_decay": 0.0, "aug": "media",
    "optimizer": "adam", "lr": 1e-3, "batch_size": 16, "img_size": 150,
    "class_weight": False,
}


def cargar_params(ruta, usar_baseline):
    if usar_baseline:
        print("  Usando hiperparámetros BASELINE (los del notebook original)")
        return dict(PARAMS_BASELINE), None

    if not os.path.exists(ruta):
        raise SystemExit(
            f"No existe {ruta}.\n"
            "Ejecuta primero:  python optuna_trashnet.py\n"
            "O entrena el baseline con:  python entrenar_mejor.py --sin-optuna"
        )
    with open(ruta, encoding="utf-8") as f:
        info = json.load(f)
    print(f"  Hiperparámetros de Optuna (trial {info['mejor_trial']}, "
          f"{info['metrica']}={info['mejor_valor']:.4f} en validación)")
    return info["params"], info


def entrenar(params, epocas, patience, workers):
    """Entrena con EarlyStopping sobre val_accuracy y restaura la mejor época."""
    semilla(SEED)

    train_loader, valid_loader, test_loader, class_names = construir_loaders(
        img_size    = params["img_size"],
        batch_size  = params["batch_size"],
        aug         = params["aug"],
        num_workers = workers,
        seed        = SEED,
    )
    print(f"  Imágenes: {len(train_loader.dataset)} train / "
          f"{len(valid_loader.dataset)} val / {len(test_loader.dataset)} test")
    print(f"  Clases: {dict(enumerate(class_names))}")

    model = construir_modelo(len(class_names), params)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros entrenables: {n_params:,}")

    pesos = pesos_de_clase(class_names).to(DEVICE) if params["class_weight"] else None
    criterion = nn.CrossEntropyLoss(weight=pesos)
    optimizer = construir_optimizador(model, params)

    # Reduce el LR cuando la validación se estanca: ayuda a afinar el último tramo
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, patience // 3)
    )

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    mejor_acc, mejor_epoca, sin_mejora = 0.0, 0, 0
    mejores_pesos = copy.deepcopy(model.state_dict())

    print(f"\n  Entrenando hasta {epocas} épocas (EarlyStopping patience={patience})")
    print("  " + "─" * 74)
    t0 = time.time()

    for epoca in range(1, epocas + 1):
        tr_loss, tr_acc = correr_epoca(model, train_loader, criterion, optimizer)
        va_loss, va_acc = correr_epoca(model, valid_loader, criterion)

        history["loss"].append(tr_loss);         history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss);     history["val_accuracy"].append(va_acc)
        scheduler.step(va_acc)

        marca = ""
        if va_acc > mejor_acc:
            mejor_acc, mejor_epoca, sin_mejora = va_acc, epoca, 0
            mejores_pesos = copy.deepcopy(model.state_dict())
            marca = "  ★ mejor"
        else:
            sin_mejora += 1

        print(f"  Época {epoca:>3}/{epocas}  loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  |  "
              f"val_loss: {va_loss:.4f}  val_acc: {va_acc:.4f}{marca}", flush=True)

        if sin_mejora >= patience:
            print(f"  EarlyStopping: {patience} épocas sin mejorar.")
            break

    model.load_state_dict(mejores_pesos)     # restaurar la mejor época
    print(f"  " + "─" * 74)
    print(f"  Mejor val_accuracy: {mejor_acc:.4f} (época {mejor_epoca})  ·  "
          f"{(time.time() - t0) / 60:.1f} min")

    return model, history, class_names, test_loader, {
        "n_params": n_params, "mejor_val_accuracy": mejor_acc,
        "mejor_epoca": mejor_epoca, "epocas_corridas": len(history["accuracy"]),
    }


def evaluar(model, test_loader, class_names, params, info_train, info_optuna, sufijo):
    """Evaluación en test + gráficos + reporte."""
    n = len(class_names)
    y_true, y_pred, y_prob = obtener_predicciones(model, test_loader)

    cm       = calcular_cm(y_true, y_pred, n)
    metricas = calcular_metricas_por_clase(cm)
    acc      = float(np.trace(cm) / np.sum(cm))
    macro_f1 = float(np.mean([metricas[i]["f1"] for i in range(n)]))

    print("\n" + "═" * 78)
    print("  EVALUACIÓN EN TEST")
    print("═" * 78)
    print(f"{'Clase':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'N':>6}")
    print("─" * 48)
    for i, clase in enumerate(class_names):
        m = metricas[i]
        print(f"{clase:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>8.3f} {m['support']:>6}")
    print("─" * 48)
    print(f"Accuracy: {acc:.4f} ({acc:.1%})  |  Macro F1: {macro_f1:.3f}")

    # ── Gráficos ──────────────────────────────────────────────────────
    crear_carpetas_resultados()

    print("\nGenerando gráficos...")
    titulo = "TrashNet (PyTorch) — " + ("hiperparámetros de Optuna" if info_optuna else "baseline")

    plot_matriz_confusion(
        cm, metricas, class_names, acc,
        os.path.join(CM_DIR, f"confusion_matrix{sufijo}.png"),
        titulo=f"{titulo} — test",
    )

    auc_por_clase = plot_roc(
        y_true, y_prob, class_names,
        os.path.join(GRAFICOS_DIR, f"roc_curves{sufijo}.png"),
        titulo=f"{titulo} — Curvas ROC (One-vs-Rest)",
    )

    m_auc = float(np.mean(list(auc_por_clase.values())))
    print("\nAUC por clase (test):")
    for clase, a in auc_por_clase.items():
        print(f"  {clase:<12}: {a:.4f}")
    print(f"  {'macro avg':<12}: {m_auc:.4f}")

    return {
        "accuracy": acc, "macro_f1": macro_f1, "macro_auc": m_auc,
        "auc_por_clase": auc_por_clase,
        "cm": cm.tolist(),
        "metricas_por_clase": {class_names[i]: metricas[i] for i in range(n)},
    }


def guardar_reporte(ruta, params, info_train, info_optuna, resultados, class_names):
    lineas = []
    add = lineas.append
    add("TrashNet — PyTorch + Optuna · Reporte de evaluación")
    add("=" * 70)
    add(f"Device: {DEVICE}")
    if info_optuna:
        add(f"Optuna: trial {info_optuna['mejor_trial']} de {info_optuna['n_trials']} · "
            f"{info_optuna['metrica']} = {info_optuna['mejor_valor']:.4f} (validación, "
            f"{info_optuna['epocas_trial']} épocas)")
    else:
        add("Optuna: no usado (hiperparámetros baseline del notebook)")
    add("")
    add("Hiperparámetros")
    add("-" * 70)
    for k, v in params.items():
        add(f"  {k:<14}: {v}")
    add("")
    add("Entrenamiento final")
    add("-" * 70)
    add(f"  Parámetros del modelo : {info_train['n_params']:,}")
    add(f"  Épocas corridas       : {info_train['epocas_corridas']}")
    add(f"  Mejor época           : {info_train['mejor_epoca']}")
    add(f"  Mejor val_accuracy    : {info_train['mejor_val_accuracy']:.4f}")
    add("")
    add("Resultados en TEST")
    add("-" * 70)
    add(f"  Accuracy  : {resultados['accuracy']:.4f}  ({resultados['accuracy']:.1%})")
    add(f"  Macro F1  : {resultados['macro_f1']:.4f}")
    add(f"  Macro AUC : {resultados['macro_auc']:.4f}")
    add("")
    add(f"  {'Clase':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} {'N':>6}")
    for clase in class_names:
        m = resultados["metricas_por_clase"][clase]
        add(f"  {clase:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} "
            f"{m['f1']:>8.3f} {resultados['auc_por_clase'][clase]:>8.3f} {m['support']:>6}")
    add("")
    add("Matriz de confusión (filas = real, columnas = predicho)")
    add(f"  {'':<12}" + "".join(f"{c[:8]:>9}" for c in class_names))
    for i, clase in enumerate(class_names):
        add(f"  {clase:<12}" + "".join(f"{v:>9}" for v in resultados["cm"][i]))

    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    print(f"  ✓ {ruta}")


def parse_args():
    p = argparse.ArgumentParser(description="Reentrena TrashNet con los mejores hiperparámetros de Optuna")
    p.add_argument("--params", default=os.path.join(OPTUNA_DIR, "mejores_params.json"),
                   help="JSON con los hiperparámetros (default: PyTorch/optuna/mejores_params.json)")
    p.add_argument("--sin-optuna", action="store_true",
                   help="entrenar con los hiperparámetros del notebook (baseline para comparar)")
    p.add_argument("--epocas",   type=int, default=50, help="épocas máximas (default 50)")
    p.add_argument("--patience", type=int, default=10, help="paciencia de EarlyStopping (default 10)")
    p.add_argument("--workers",  type=int, default=2,  help="num_workers de los DataLoader")
    return p.parse_args()


def main():
    args = parse_args()
    sufijo = "_baseline" if args.sin_optuna else "_optuna"

    print("─" * 78)
    print("  Reentrenamiento final — TrashNet (PyTorch)")
    print("─" * 78)
    params, info_optuna = cargar_params(args.params, args.sin_optuna)

    model, history, class_names, test_loader, info_train = entrenar(
        params, args.epocas, args.patience, args.workers
    )

    resultados = evaluar(
        model, test_loader, class_names, params, info_train, info_optuna, sufijo
    )

    plot_curvas_entrenamiento(
        history, os.path.join(GRAFICOS_DIR, f"curvas_entrenamiento{sufijo}.png")
    )

    # ── Guardar modelo ────────────────────────────────────────────────
    os.makedirs(MODELOS_DIR, exist_ok=True)
    ruta_modelo = os.path.join(MODELOS_DIR, f"trashnet_pytorch{sufijo}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names"     : class_names,
        "img_size"        : (params["img_size"], params["img_size"]),
        "hiperparametros" : params,
        "info_entrenamiento": info_train,
        "test": {k: resultados[k] for k in ("accuracy", "macro_f1", "macro_auc")},
    }, ruta_modelo)
    print(f"\n  ✓ {ruta_modelo}")

    guardar_reporte(
        os.path.join(REPORTE_DIR, f"reporte{sufijo}.txt"),
        params, info_train, info_optuna, resultados, class_names
    )

    print("\n✓ Listo. Resultados en PyTorch/resultados/")


if __name__ == "__main__":
    main()

"""
Modelo 1 de la segunda entrega (TensorFlow) — MobileNetV3Small, 6 clases de material.

Versión script de `tensorflow/TrashNet_TensorFlow_MobileNetV3_DEFINITIVO.ipynb`.
Transfer learning en 2 fases (backbone congelado -> fine-tuning de las últimas 20
capas) + búsqueda de hiperparámetros con Optuna (10 trials, tal como en el notebook
original — no se aplicó la corrección metodológica de la rama PyTorch, ver
`PyTorch/CONTEXT.md` §9 si se quiere igualar).

Uso:
    python entrenar.py
    python entrenar.py --sin-optuna
    python entrenar.py --trials 20 --epocas-busqueda 8

Genera en scriptsPython/pesos/: mobilenetv3_sin_optuna.keras, mobilenetv3_optuna.keras
Y en scriptsPython/resultados/: curvas, matrices de confusión, ROC y reporte con
prefijo `mnv3_`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# parents[2] = scriptsPython/ (este archivo vive en scriptsPython/modelos/mobilenetv3/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comun import utils as u

CLASSES = u.ORIGINAL_CLASSES
N_CLASSES = len(CLASSES)

BATCH_SIZE = 32
EPOCHS_FASE1 = 25
EPOCHS_FASE2 = 15
FINE_TUNE_LR = 1e-5
N_CAPAS_FINE_TUNE = 20

BASE_DENSE_UNITS = 128
BASE_DROPOUT = 0.35
BASE_LR = 3e-4
BASE_OPTIMIZER = "adam"


def cargar_datos():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        u.TRAIN_DIR, labels="inferred", label_mode="categorical", class_names=CLASSES,
        image_size=u.IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True, seed=42,
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        u.VALID_DIR, labels="inferred", label_mode="categorical", class_names=CLASSES,
        image_size=u.IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        u.TEST_DIR, labels="inferred", label_mode="categorical", class_names=CLASSES,
        image_size=u.IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds, valid_ds, test_ds = (ds.prefetch(autotune) for ds in (train_ds, valid_ds, test_ds))

    conteos = {
        c: len([f for f in os.listdir(os.path.join(u.TRAIN_DIR, c)) if f.lower().endswith(u.EXTENSIONES)])
        for c in CLASSES
    }
    y_train = np.concatenate([[i] * conteos[c] for i, c in enumerate(CLASSES)])
    pesos = compute_class_weight(class_weight="balanced", classes=np.arange(N_CLASSES), y=y_train)
    class_weights = {i: float(p) for i, p in enumerate(pesos)}

    print(f"Train: {sum(conteos.values())} imágenes")
    return train_ds, valid_ds, test_ds, class_weights


def entrenar_dos_fases(dense_units, dropout, optimizer_name, lr, nombre, class_weights,
                       train_ds, valid_ds, epocas_fase1=EPOCHS_FASE1, epocas_fase2=EPOCHS_FASE2,
                       verbose=1):
    u.limpiar_memoria()
    model, backbone = u.construir_modelo(dense_units, dropout, optimizer_name, lr,
                                         N_CLASSES, nombre)

    h1 = model.fit(train_ds, validation_data=valid_ds, epochs=epocas_fase1,
                   class_weight=class_weights,
                   callbacks=u.callbacks_estandar(patience_es=6, patience_lr=3, min_lr=1e-6, verbose=verbose),
                   verbose=verbose)

    u.descongelar_ultimas_capas(backbone, N_CAPAS_FINE_TUNE)
    model.compile(optimizer=u.construir_optimizador(optimizer_name, FINE_TUNE_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    h2 = model.fit(train_ds, validation_data=valid_ds, epochs=epocas_fase2,
                   class_weight=class_weights,
                   callbacks=u.callbacks_estandar(patience_es=5, patience_lr=2, min_lr=1e-7, verbose=verbose),
                   verbose=verbose)

    return model, u.unir_historiales(h1, h2)


def buscar_con_optuna(train_ds, valid_ds, class_weights, args):
    import optuna

    def objective(trial):
        u.limpiar_memoria()
        dense_units = trial.suggest_categorical("dense_units", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout_rate", 0.20, 0.50, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])

        model, _ = u.construir_modelo(dense_units, dropout_rate, optimizer_name, learning_rate,
                                      N_CLASSES, f"trial_{trial.number}")
        history = model.fit(
            train_ds, validation_data=valid_ds, epochs=args.epocas_busqueda,
            class_weight=class_weights,
            callbacks=u.callbacks_estandar(patience_es=3, patience_lr=2, min_lr=1e-7, verbose=0),
            verbose=0,
        )
        mejor = float(max(history.history["val_accuracy"]))
        print(f"Trial {trial.number:02d} | val_accuracy={mejor:.4f} | {trial.params}")

        del model
        u.limpiar_memoria()
        return mejor

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="TrashNet_MobileNetV3_scriptsPython_TF")
    study.optimize(objective, n_trials=args.trials)
    return study


def evaluar_y_reportar(model_base, model_optuna, hist_base, hist_optuna, test_ds, args):
    print("\n" + "=" * 70)
    print("PARTE C — Evaluación en test")
    print("=" * 70)

    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    y_prob_base = model_base.predict(test_ds, verbose=0)
    y_prob_optuna = model_optuna.predict(test_ds, verbose=0)
    y_pred_base = np.argmax(y_prob_base, axis=1)
    y_pred_optuna = np.argmax(y_prob_optuna, axis=1)

    m_base = u.metricas_multiclase(y_true, y_pred_base, y_prob_base)
    m_optuna = u.metricas_multiclase(y_true, y_pred_optuna, y_prob_optuna)

    ancho = max(len(k) for k in m_base) + 2
    print(f"{'':<{ancho}}{'Sin Optuna':>14}{'Con Optuna':>14}")
    for k in m_base:
        print(f"{k:<{ancho}}{m_base[k]:>14.4f}{m_optuna[k]:>14.4f}")

    from sklearn.metrics import confusion_matrix, classification_report
    cm_base = confusion_matrix(y_true, y_pred_base)
    cm_optuna = confusion_matrix(y_true, y_pred_optuna)

    u.graficar_curvas(hist_base, "MobileNetV3 sin Optuna",
                      os.path.join(u.GRAFICOS_DIR, "mnv3_curvas_sin_optuna.png"))
    u.graficar_curvas(hist_optuna, "MobileNetV3 con Optuna",
                      os.path.join(u.GRAFICOS_DIR, "mnv3_curvas_optuna.png"))
    u.graficar_matriz_confusion(cm_base, CLASSES, "Matriz de confusión — Sin Optuna",
                                os.path.join(u.CM_DIR, "mnv3_confusion_sin_optuna.png"))
    u.graficar_matriz_confusion(cm_optuna, CLASSES, "Matriz de confusión — Con Optuna",
                                os.path.join(u.CM_DIR, "mnv3_confusion_optuna.png"))
    u.graficar_roc_por_clase(y_true, y_prob_base, CLASSES, "Curvas ROC — Sin Optuna",
                             os.path.join(u.GRAFICOS_DIR, "mnv3_roc_sin_optuna.png"))
    u.graficar_roc_por_clase(y_true, y_prob_optuna, CLASSES, "Curvas ROC — Con Optuna",
                             os.path.join(u.GRAFICOS_DIR, "mnv3_roc_optuna.png"))

    reporte = ["TrashNet — MobileNetV3Small (TensorFlow, scriptsPython) — Reporte",
              "=" * 70, ""]
    for nombre, m in [("SIN OPTUNA", m_base), ("CON OPTUNA", m_optuna)]:
        reporte.append(nombre)
        reporte += [f"  {k:<20}: {v:.4f}" for k, v in m.items()]
        reporte.append("")
    reporte.append(classification_report(y_true, y_pred_optuna, target_names=CLASSES,
                                         digits=4, zero_division=0))

    ruta_reporte = os.path.join(u.REPORTE_DIR, "mnv3_reporte.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write("\n".join(reporte) + "\n")
    print(f"\n  {ruta_reporte}")
    print("\nListo. Resultados en scriptsPython/resultados/ y pesos en scriptsPython/pesos/")


def main():
    args = parse_args()
    u.crear_carpetas_resultados()
    u.configurar_gpu()
    u.semilla()

    train_ds, valid_ds, test_ds, class_weights = cargar_datos()

    print("\n" + "=" * 70)
    print("PARTE A — Modelo base (sin Optuna)")
    print("=" * 70)
    model_base, hist_base = entrenar_dos_fases(
        BASE_DENSE_UNITS, BASE_DROPOUT, BASE_OPTIMIZER, BASE_LR,
        "mobilenetv3_sin_optuna", class_weights, train_ds, valid_ds,
    )
    ruta_base = os.path.join(u.PESOS_DIR, "mobilenetv3_sin_optuna.keras")
    model_base.save(ruta_base)
    print(f"\n  {ruta_base}")

    if args.sin_optuna:
        evaluar_y_reportar(model_base, model_base, hist_base, hist_base, test_ds, args)
        return

    print("\n" + "=" * 70)
    print(f"PARTE B — Búsqueda de hiperparámetros con Optuna ({args.trials} trials)")
    print("=" * 70)
    study = buscar_con_optuna(train_ds, valid_ds, class_weights, args)

    print(f"\nMejor trial: {study.best_trial.number} — val_accuracy {study.best_value:.4f}")
    print(f"Mejores hiperparámetros: {study.best_params}")

    model_optuna, hist_optuna = entrenar_dos_fases(
        study.best_params["dense_units"], study.best_params["dropout_rate"],
        study.best_params["optimizer"], study.best_params["learning_rate"],
        "mobilenetv3_optuna", class_weights, train_ds, valid_ds,
    )
    ruta_optuna = os.path.join(u.PESOS_DIR, "mobilenetv3_optuna.keras")
    model_optuna.save(ruta_optuna)
    print(f"  {ruta_optuna}")

    evaluar_y_reportar(model_base, model_optuna, hist_base, hist_optuna, test_ds, args)


def parse_args():
    p = argparse.ArgumentParser(description="Entrena el modelo MobileNetV3 de 6 clases (TensorFlow)")
    p.add_argument("--sin-optuna", action="store_true", help="entrenar solo el baseline")
    p.add_argument("--trials", type=int, default=10, help="trials de Optuna (default 10, como el notebook)")
    p.add_argument("--epocas-busqueda", type=int, default=8, help="épocas por trial (default 8)")
    return p.parse_args()


if __name__ == "__main__":
    main()

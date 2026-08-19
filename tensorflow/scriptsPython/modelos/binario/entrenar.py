"""
Modelo 3 de la segunda entrega (TensorFlow) — Clasificador binario reciclable / basura.
Es el modelo DESEQUILIBRADO que pide la pauta (punto 5).

Versión script de `tensorflow/TrashNet_TensorFlow_Binario_DEFINITIVO.ipynb`.

Diferencia importante frente al binario de PyTorch (`PyTorch/scriptsPython/modelos/
binario/entrenar.py`): aquí el Garbage Dataset (`dataset_gd/trash`) **se incorpora al
entrenamiento** en vez de reservarse solo para evaluación fuera de distribución — 70%
a train, 15% a validation, 15% a test, con una única partición aleatoria (semilla fija)
para que ningún split se filtre a otro. La idea es que el modelo vea ejemplos reales
de basura (fondo y contexto), no solo las fotos de estudio de TrashNet.

Salida sigmoide de 1 neurona: P(reciclable). P(basura) = 1 - P(reciclable).

Uso:
    python entrenar.py
    python entrenar.py --sin-optuna
    python entrenar.py --sin-gd            # ignora dataset_gd/, solo TrashNet

Genera en scriptsPython/pesos/: binario_sin_optuna.keras, binario_optuna.keras
Y en scriptsPython/resultados/: matrices, curvas ROC/PR y reporte con prefijo `bin_`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# parents[2] = scriptsPython/ (este archivo vive en scriptsPython/modelos/binario/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comun import utils as u

CLASSES = ["basura", "reciclable"]
IDX_BASURA = 0
IDX_RECICLABLE = 1

BATCH_SIZE = 32
EPOCHS_FASE1 = 20
EPOCHS_FASE2 = 12
FINE_TUNE_LR = 1e-5
N_CAPAS_FINE_TUNE = 30

BASE_DENSE_UNITS = 128
BASE_DROPOUT = 0.30
BASE_LR = 1e-3
BASE_OPTIMIZER = "adam"

GD_TRAIN_RATIO, GD_VALID_RATIO, GD_TEST_RATIO = 0.70, 0.15, 0.15


def archivos_split_trashnet(ruta_split):
    rutas, etiquetas = [], []
    for clase in u.ORIGINAL_CLASSES:
        carpeta = Path(ruta_split) / clase
        archivos = sorted(str(carpeta / f) for f in os.listdir(carpeta)
                          if f.lower().endswith(u.EXTENSIONES))
        etiqueta = IDX_BASURA if clase == "trash" else IDX_RECICLABLE
        rutas.extend(archivos)
        etiquetas.extend([etiqueta] * len(archivos))
    return rutas, etiquetas


def dividir_garbage_dataset(usar_gd):
    gd_trash_dir = os.path.join(u.GARBAGE_DIR, "trash")
    if not usar_gd or not os.path.isdir(gd_trash_dir):
        return [], [], []

    archivos = np.array(sorted(
        str(Path(gd_trash_dir) / f) for f in os.listdir(gd_trash_dir)
        if f.lower().endswith(u.EXTENSIONES)
    ))
    rng = np.random.default_rng(42)
    archivos = archivos[rng.permutation(len(archivos))]

    n_train = int(round(len(archivos) * GD_TRAIN_RATIO))
    n_valid = int(round(len(archivos) * GD_VALID_RATIO))
    return (list(archivos[:n_train]), list(archivos[n_train:n_train + n_valid]),
            list(archivos[n_train + n_valid:]))


def cargar_imagen(ruta, etiqueta):
    datos = tf.io.read_file(ruta)
    imagen = tf.io.decode_image(datos, channels=3, expand_animations=False)
    imagen.set_shape([None, None, 3])
    imagen = tf.image.resize(imagen, u.IMG_SIZE)
    imagen = tf.cast(imagen, tf.float32)
    return imagen, tf.cast(etiqueta, tf.int32)


def crear_dataset(paths, labels, shuffle):
    autotune = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(cargar_imagen, num_parallel_calls=autotune)
    return ds.batch(BATCH_SIZE).prefetch(autotune)


def cargar_datos(usar_gd):
    train_p, train_l = archivos_split_trashnet(u.TRAIN_DIR)
    valid_p, valid_l = archivos_split_trashnet(u.VALID_DIR)
    test_p, test_l = archivos_split_trashnet(u.TEST_DIR)

    gd_train, gd_valid, gd_test = dividir_garbage_dataset(usar_gd)
    if gd_train or gd_valid or gd_test:
        print(f"Garbage Dataset incorporado: {len(gd_train)} train / "
              f"{len(gd_valid)} valid / {len(gd_test)} test (todas 'basura')")

    train_paths = train_p + gd_train
    train_labels = train_l + [IDX_BASURA] * len(gd_train)
    valid_paths = valid_p + gd_valid
    valid_labels = valid_l + [IDX_BASURA] * len(gd_valid)
    test_paths = test_p + gd_test
    test_labels = test_l + [IDX_BASURA] * len(gd_test)

    def contar(labels):
        labels = np.array(labels)
        return {"basura": int((labels == IDX_BASURA).sum()), "reciclable": int((labels == IDX_RECICLABLE).sum())}

    for nombre, labels in [("Train", train_labels), ("Validation", valid_labels), ("Test", test_labels)]:
        c = contar(labels)
        total = sum(c.values())
        print(f"{nombre}: {total} imágenes | basura={c['basura']} | reciclable={c['reciclable']} "
              f"| basura={c['basura'] / total * 100:.2f}%")

    n_basura, n_reciclable = contar(train_labels)["basura"], contar(train_labels)["reciclable"]
    n_total = n_basura + n_reciclable
    class_weights = {IDX_BASURA: n_total / (2 * n_basura), IDX_RECICLABLE: n_total / (2 * n_reciclable)}

    train_ds = crear_dataset(train_paths, train_labels, shuffle=True)
    valid_ds = crear_dataset(valid_paths, valid_labels, shuffle=False)
    test_ds = crear_dataset(test_paths, test_labels, shuffle=False)

    return train_ds, valid_ds, test_ds, class_weights, np.array(valid_labels), np.array(test_labels)


def obtener_probabilidades(model, dataset):
    """Devuelve (y_true, p_reciclable)."""
    y_true, p_reciclable = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0).reshape(-1)
        y_true.extend(labels.numpy().reshape(-1))
        p_reciclable.extend(probs)
    return np.array(y_true, dtype=int), np.array(p_reciclable, dtype=float)


def entrenar_dos_fases(dense_units, dropout, optimizer_name, lr, nombre, class_weights,
                       train_ds, valid_ds, verbose=1):
    u.limpiar_memoria()
    model, backbone = u.construir_modelo(
        dense_units, dropout, optimizer_name, lr, n_classes=2, nombre_modelo=nombre,
        activacion_salida="sigmoid", unidades_salida=1, loss="binary_crossentropy",
        incluir_contraste=False,
    )

    h1 = model.fit(
        train_ds, validation_data=valid_ds, epochs=EPOCHS_FASE1, class_weight=class_weights,
        callbacks=u.callbacks_estandar(monitor="val_loss", modo="min", patience_es=4,
                                       patience_lr=2, min_lr=1e-6, factor=0.3, verbose=verbose),
        verbose=verbose,
    )

    u.descongelar_ultimas_capas(backbone, N_CAPAS_FINE_TUNE)
    model.compile(optimizer=u.construir_optimizador("adam", FINE_TUNE_LR),
                  loss="binary_crossentropy", metrics=["accuracy"])

    h2 = model.fit(
        train_ds, validation_data=valid_ds, epochs=EPOCHS_FASE2, class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=verbose)],
        verbose=verbose,
    )

    return model, u.unir_historiales(h1, h2)


def buscar_con_optuna(train_ds, valid_ds, class_weights, y_valid_basura, args):
    import optuna
    from sklearn.metrics import average_precision_score

    def objective(trial):
        u.limpiar_memoria()
        dense_units = trial.suggest_categorical("dense_units", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout_rate", 0.20, 0.50, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])

        model, _ = u.construir_modelo(
            dense_units, dropout_rate, optimizer_name, learning_rate, n_classes=2,
            nombre_modelo=f"trial_{trial.number}", activacion_salida="sigmoid",
            unidades_salida=1, loss="binary_crossentropy", incluir_contraste=False,
        )
        model.fit(
            train_ds, validation_data=valid_ds, epochs=args.epocas_busqueda,
            class_weight=class_weights,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                         restore_best_weights=True, verbose=0)],
            verbose=0,
        )

        _, p_reciclable = obtener_probabilidades(model, valid_ds)
        ap = average_precision_score(y_valid_basura, 1.0 - p_reciclable)
        print(f"Trial {trial.number:02d} | AP basura={ap:.4f} | {trial.params}")

        del model
        u.limpiar_memoria()
        return ap

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="TrashNet_Binario_scriptsPython_TF")
    study.optimize(objective, n_trials=args.trials)
    return study


def calibrar_umbral(y_valid_basura, p_valid_basura):
    from sklearn.metrics import f1_score
    umbrales = np.linspace(0.05, 0.95, 181)
    resultados = [(u_, f1_score(y_valid_basura, (p_valid_basura >= u_).astype(int), zero_division=0))
                  for u_ in umbrales]
    return max(resultados, key=lambda x: x[1])


def main():
    args = parse_args()
    u.crear_carpetas_resultados()
    u.configurar_gpu()
    u.semilla()

    train_ds, valid_ds, test_ds, class_weights, y_valid_all, y_test_all = cargar_datos(not args.sin_gd)
    y_valid_basura = (y_valid_all == IDX_BASURA).astype(int)
    y_test_basura = (y_test_all == IDX_BASURA).astype(int)

    print("\n" + "=" * 70)
    print("PARTE A — Modelo binario base (sin Optuna)")
    print("=" * 70)
    model_base, hist_base = entrenar_dos_fases(
        BASE_DENSE_UNITS, BASE_DROPOUT, BASE_OPTIMIZER, BASE_LR,
        "binario_sin_optuna", class_weights, train_ds, valid_ds,
    )
    ruta_base = os.path.join(u.PESOS_DIR, "binario_sin_optuna.keras")
    model_base.save(ruta_base)
    print(f"\n  {ruta_base}")

    model_optuna, hist_optuna = model_base, hist_base

    if not args.sin_optuna:
        print("\n" + "=" * 70)
        print(f"PARTE B — Búsqueda con Optuna ({args.trials} trials, métrica: AP sobre basura)")
        print("=" * 70)
        study = buscar_con_optuna(train_ds, valid_ds, class_weights, y_valid_basura, args)
        print(f"\nMejor trial: {study.best_trial.number} — AP basura {study.best_value:.4f}")
        print(f"Mejores hiperparámetros: {study.best_params}")

        model_optuna, hist_optuna = entrenar_dos_fases(
            study.best_params["dense_units"], study.best_params["dropout_rate"],
            study.best_params["optimizer"], study.best_params["learning_rate"],
            "binario_optuna", class_weights, train_ds, valid_ds,
        )
        ruta_optuna = os.path.join(u.PESOS_DIR, "binario_optuna.keras")
        model_optuna.save(ruta_optuna)
        print(f"  {ruta_optuna}")

    evaluar_y_reportar(model_base, model_optuna, hist_base, hist_optuna,
                       valid_ds, test_ds, y_valid_basura, y_test_all, y_test_basura, args)


def evaluar_y_reportar(model_base, model_optuna, hist_base, hist_optuna,
                       valid_ds, test_ds, y_valid_basura, y_test_all, y_test_basura, args):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score, confusion_matrix,
        classification_report, roc_curve, precision_recall_curve,
    )

    print("\n" + "=" * 70)
    print("PARTE C — Calibración del umbral y evaluación en test")
    print("=" * 70)

    _, p_valid_reciclable = obtener_probabilidades(model_optuna, valid_ds)
    p_valid_basura = 1.0 - p_valid_reciclable
    UMBRAL, f1_calibrado = calibrar_umbral(y_valid_basura, p_valid_basura)
    print(f"Umbral calibrado en validación (máximo F1): {UMBRAL:.3f} (F1={f1_calibrado:.4f})")

    y_true, p_base_reciclable = obtener_probabilidades(model_base, test_ds)
    _, p_optuna_reciclable = obtener_probabilidades(model_optuna, test_ds)
    p_base_basura = 1.0 - p_base_reciclable
    p_optuna_basura = 1.0 - p_optuna_reciclable

    y_pred_base = np.where(p_base_basura >= 0.5, IDX_BASURA, IDX_RECICLABLE)
    y_pred_optuna = np.where(p_optuna_basura >= UMBRAL, IDX_BASURA, IDX_RECICLABLE)

    def metricas(y_real, y_pred, p_basura):
        y_real_b = (y_real == IDX_BASURA).astype(int)
        y_pred_b = (y_pred == IDX_BASURA).astype(int)
        return {
            "Accuracy": accuracy_score(y_real, y_pred),
            "Balanced accuracy": balanced_accuracy_score(y_real, y_pred),
            "Precision basura": precision_score(y_real_b, y_pred_b, zero_division=0),
            "Recall basura": recall_score(y_real_b, y_pred_b, zero_division=0),
            "F1 basura": f1_score(y_real_b, y_pred_b, zero_division=0),
            "ROC AUC basura": roc_auc_score(y_real_b, p_basura),
            "PR AUC / AP basura": average_precision_score(y_real_b, p_basura),
        }

    m_base = metricas(y_true, y_pred_base, p_base_basura)
    m_optuna = metricas(y_true, y_pred_optuna, p_optuna_basura)

    ancho = max(len(k) for k in m_base) + 2
    print(f"{'':<{ancho}}{'Sin Optuna':>14}{'Con Optuna':>14}")
    for k in m_base:
        print(f"{k:<{ancho}}{m_base[k]:>14.4f}{m_optuna[k]:>14.4f}")

    cm_base = confusion_matrix(y_true, y_pred_base)
    cm_optuna = confusion_matrix(y_true, y_pred_optuna)

    u.graficar_curvas(hist_base, "Binario sin Optuna",
                      os.path.join(u.GRAFICOS_DIR, "bin_curvas_sin_optuna.png"))
    u.graficar_curvas(hist_optuna, "Binario con Optuna",
                      os.path.join(u.GRAFICOS_DIR, "bin_curvas_optuna.png"))
    u.graficar_matriz_confusion(cm_base, CLASSES, "Matriz de confusión — Sin Optuna",
                                os.path.join(u.CM_DIR, "bin_confusion_sin_optuna.png"))
    u.graficar_matriz_confusion(cm_optuna, CLASSES, "Matriz de confusión — Con Optuna (umbral calibrado)",
                                os.path.join(u.CM_DIR, "bin_confusion_optuna.png"))

    import matplotlib.pyplot as plt
    fpr_b, tpr_b, _ = roc_curve(y_test_basura, p_base_basura)
    fpr_o, tpr_o, _ = roc_curve(y_test_basura, p_optuna_basura)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_b, tpr_b, label=f"Sin Optuna (AUC={m_base['ROC AUC basura']:.3f})")
    plt.plot(fpr_o, tpr_o, label=f"Con Optuna (AUC={m_optuna['ROC AUC basura']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    plt.xlabel("Tasa de falsos positivos"); plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curva ROC — clase basura (test)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    ruta_roc = os.path.join(u.GRAFICOS_DIR, "bin_roc_test.png")
    plt.savefig(ruta_roc, dpi=120); plt.close()
    print(f"  {ruta_roc}")

    prec_b, rec_b, _ = precision_recall_curve(y_test_basura, p_base_basura)
    prec_o, rec_o, _ = precision_recall_curve(y_test_basura, p_optuna_basura)
    plt.figure(figsize=(8, 6))
    plt.plot(rec_b, prec_b, label=f"Sin Optuna (AP={m_base['PR AUC / AP basura']:.3f})")
    plt.plot(rec_o, prec_o, label=f"Con Optuna (AP={m_optuna['PR AUC / AP basura']:.3f})")
    plt.axhline(y_test_basura.mean(), linestyle="--", color="gray", label="Azar (prevalencia)")
    plt.xlabel("Recall (basura)"); plt.ylabel("Precision (basura)")
    plt.title("Curva precision-recall — clase basura (test)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    ruta_pr = os.path.join(u.GRAFICOS_DIR, "bin_pr_test.png")
    plt.savefig(ruta_pr, dpi=120); plt.close()
    print(f"  {ruta_pr}")

    reporte = [
        "TrashNet — Binario reciclable/basura (TensorFlow, scriptsPython) — Reporte",
        "=" * 70, f"Umbral calibrado en validación: {UMBRAL:.3f}", "",
    ]
    for nombre, m in [("SIN OPTUNA", m_base), ("CON OPTUNA (umbral calibrado)", m_optuna)]:
        reporte.append(nombre)
        reporte += [f"  {k:<20}: {v:.4f}" for k, v in m.items()]
        reporte.append("")
    reporte.append(classification_report(y_true, y_pred_optuna, target_names=CLASSES, zero_division=0))

    ruta_reporte = os.path.join(u.REPORTE_DIR, "bin_reporte.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write("\n".join(reporte) + "\n")
    print(f"\n  {ruta_reporte}")
    print("\nListo. Resultados en scriptsPython/resultados/ y pesos en scriptsPython/pesos/")


def parse_args():
    p = argparse.ArgumentParser(description="Entrena el modelo binario reciclable/basura (TensorFlow, desequilibrado)")
    p.add_argument("--sin-optuna", action="store_true", help="entrenar solo el baseline")
    p.add_argument("--sin-gd", action="store_true", help="no incorporar dataset_gd/trash al entrenamiento")
    p.add_argument("--trials", type=int, default=15, help="trials de Optuna (default 15, como el notebook)")
    p.add_argument("--epocas-busqueda", type=int, default=8, help="épocas por trial (default 8)")
    return p.parse_args()


if __name__ == "__main__":
    main()

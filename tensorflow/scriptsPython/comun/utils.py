"""
Módulo común de la segunda entrega (TensorFlow/Keras).

Equivalente al `comun/utils.py` de la rama PyTorch, pero con las herramientas de
Keras: aquí el entorno sí tiene scikit-learn, así que las métricas se calculan con
`sklearn.metrics` directamente en vez de reimplementarlas a mano.

Comparte:
  - Rutas del proyecto (dataset, pesos, resultados)
  - construir_modelo: MobileNetV3Small (backbone congelado) + aumento de datos +
    cabezal denso, igual que en los tres notebooks de `tensorflow/`
  - Callbacks estándar (EarlyStopping + ReduceLROnPlateau)
  - Gráficos: curvas de entrenamiento, matriz de confusión, ROC por clase

No se ejecuta por sí solo, se importa.
"""

from __future__ import annotations

import gc
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── RUTAS ─────────────────────────────────────────────────────────────
COMUN_DIR      = os.path.dirname(os.path.abspath(__file__))
SCRIPTSPY_DIR  = os.path.dirname(COMUN_DIR)
TF_DIR         = os.path.dirname(SCRIPTSPY_DIR)
PROJECT_DIR    = os.path.dirname(TF_DIR)

DATASET_DIR    = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR      = os.path.join(DATASET_DIR, "train")
VALID_DIR      = os.path.join(DATASET_DIR, "validation")
TEST_DIR       = os.path.join(DATASET_DIR, "test")
GARBAGE_DIR    = os.path.join(PROJECT_DIR, "dataset_gd")

PESOS_DIR      = os.path.join(SCRIPTSPY_DIR, "pesos")
RESULTADOS_DIR = os.path.join(SCRIPTSPY_DIR, "resultados")
GRAFICOS_DIR   = os.path.join(RESULTADOS_DIR, "graficos")
CM_DIR         = os.path.join(RESULTADOS_DIR, "matriz_de_confusion")
REPORTE_DIR    = os.path.join(RESULTADOS_DIR, "reporte")

ORIGINAL_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (224, 224)
EXTENSIONES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def crear_carpetas_resultados():
    for d in (PESOS_DIR, GRAFICOS_DIR, CM_DIR, REPORTE_DIR):
        os.makedirs(d, exist_ok=True)


def semilla(valor: int = 42):
    random.seed(valor)
    np.random.seed(valor)
    tf.keras.utils.set_random_seed(valor)


def configurar_gpu():
    """Uso progresivo de memoria GPU, para no reservarla toda de golpe."""
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print("No se pudo configurar memory growth:", error)


def limpiar_memoria():
    keras.backend.clear_session()
    gc.collect()


# ── DATA AUGMENTATION ────────────────────────────────────────────────
def crear_data_augmentation(nombre="data_augmentation", incluir_contraste=True):
    """
    Igual en los tres notebooks: flip horizontal, rotación, zoom, traslación, y
    contraste (el binario no usa RandomContrast).
    """
    capas = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
    ]
    if incluir_contraste:
        capas.append(layers.RandomContrast(0.10))
    capas.append(layers.RandomTranslation(height_factor=0.05, width_factor=0.05))
    return keras.Sequential(capas, name=nombre)


# ── MODELO ────────────────────────────────────────────────────────────
def construir_modelo(dense_units, dropout_rate, optimizer_name, learning_rate,
                     n_classes, nombre_modelo, activacion_salida="softmax",
                     unidades_salida=None, loss="categorical_crossentropy",
                     incluir_contraste=True, img_size=IMG_SIZE):
    """
    MobileNetV3Small(include_top=False, imagenet) -> aumento de datos ->
    GlobalAveragePooling2D -> Dense(dense_units, relu) -> Dropout -> Dense(salida).

    `unidades_salida` y `activacion_salida` permiten reusar esto para el modelo
    binario, que usa una salida sigmoide de 1 neurona en vez de softmax de N clases.
    """
    if unidades_salida is None:
        unidades_salida = n_classes

    base_model = keras.applications.MobileNetV3Small(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,     # el backbone ya espera píxeles en [0, 255]
    )
    base_model.trainable = False        # Fase 1: backbone congelado

    inputs = keras.Input(shape=img_size + (3,))
    x = crear_data_augmentation(incluir_contraste=incluir_contraste)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(unidades_salida, activation=activacion_salida)(x)

    model = keras.Model(inputs, outputs, name=nombre_modelo)
    model.compile(optimizer=construir_optimizador(optimizer_name, learning_rate),
                  loss=loss, metrics=["accuracy"])

    return model, base_model


def construir_optimizador(optimizer_name, learning_rate):
    nombre = optimizer_name.lower()
    if nombre == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if nombre == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    if nombre == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    raise ValueError(f"Optimizador no soportado: {optimizer_name}")


def descongelar_ultimas_capas(backbone, n_capas=20):
    """
    Fase 2: descongela las últimas `n_capas` del backbone. BatchNormalization queda
    siempre congelada (parámetros y estadísticas) para que el fine-tuning sea estable.
    """
    backbone.trainable = True
    for layer in backbone.layers[:-n_capas]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def callbacks_estandar(monitor="val_accuracy", modo="max", patience_es=6,
                       patience_lr=3, min_lr=1e-6, factor=0.5, verbose=1):
    return [
        keras.callbacks.EarlyStopping(monitor=monitor, mode=modo, patience=patience_es,
                                      restore_best_weights=True, verbose=verbose),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=factor,
                                          patience=patience_lr, min_lr=min_lr, verbose=verbose),
    ]


def unir_historiales(history_fase1, history_fase2, claves=("accuracy", "val_accuracy", "loss", "val_loss")):
    resultado = {k: list(history_fase1.history[k]) + list(history_fase2.history[k]) for k in claves}
    resultado["inicio_fine_tuning"] = len(history_fase1.history[claves[0]])
    return resultado


# ── MÉTRICAS (con scikit-learn) ─────────────────────────────────────────
def metricas_multiclase(y_true, y_pred, y_prob):
    from sklearn.metrics import roc_auc_score
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC AUC macro OVR": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"),
    }


# ── GRÁFICOS ──────────────────────────────────────────────────────────
def graficar_curvas(hist, titulo, archivo, extra_paneles=None):
    paneles = [("loss", "val_loss", "Loss"), ("accuracy", "val_accuracy", "Accuracy")]
    if extra_paneles:
        paneles += extra_paneles

    cols = 2
    filas = int(np.ceil(len(paneles) / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(6.5 * cols, 4.2 * filas))
    axes_flat = list(np.array(axes).flat) if len(paneles) > 1 else [axes]

    for ax, (tr, va, etiqueta) in zip(axes_flat, paneles):
        if tr:
            ax.plot(hist[tr], label="Train")
        ax.plot(hist[va], label="Validation")
        ax.axvline(hist["inicio_fine_tuning"] - 1, linestyle="--", color="gray",
                   label="Inicio fine-tuning")
        ax.set_title(etiqueta)
        ax.set_xlabel("Época")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(len(paneles), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(archivo, dpi=120)
    plt.close(fig)
    print(f"  {archivo}")


def graficar_matriz_confusion(cm, class_names, titulo, archivo):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6.5, n * 1.4), max(5.5, n * 1.2)))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=40, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")

    umbral = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > umbral else "black")

    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(archivo, dpi=120)
    plt.close(fig)
    print(f"  {archivo}")


def graficar_roc_por_clase(y_true, y_prob, class_names, titulo, archivo):
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(n))

    plt.figure(figsize=(9, 7))
    aucs = {}
    for i, clase in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[clase] = roc_auc
        plt.plot(fpr, tpr, label=f"{clase} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title(titulo)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(archivo, dpi=120)
    plt.close()
    print(f"  {archivo}")
    return aucs

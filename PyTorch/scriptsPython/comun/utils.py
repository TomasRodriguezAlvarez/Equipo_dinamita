"""
Módulo común de la segunda entrega (PyTorch).

Todo lo que comparten los tres scripts de entrenamiento
(modelos/mobilenetv3, modelos/jerarquico, modelos/binario):

  - Rutas del proyecto (dataset, pesos, resultados)
  - TrashNetMobileNetV3: backbone MobileNetV3Small + cabezal denso
  - Congelamiento / descongelamiento de capas para transfer learning en 2 fases
  - Bucle de entrenamiento con EarlyStopping + ReduceLROnPlateau (PyTorch no trae
    callbacks, así que se reimplementan a mano)
  - Métricas de clasificación calculadas con NumPy (matriz de confusión, precision,
    recall, F1, ROC/AUC, precision-recall/AP) porque el entorno del proyecto no tiene
    scikit-learn instalado (ver PyTorch/CONTEXT.md §10)
  - Utilidades de Optuna: encolar el baseline como trial 0 y reentrenar el top-K a
    presupuesto completo, la corrección metodológica documentada en CONTEXT.md §9
  - Gráficos: curvas de entrenamiento, matriz de confusión, ROC por clase

Es la versión "script" de lo que en los notebooks estaba repetido tres veces casi
palabra por palabra. No se ejecuta por sí solo, se importa.
"""

from __future__ import annotations

import copy
import gc
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

import matplotlib
matplotlib.use("Agg")            # backend sin ventana: los scripts guardan PNG
import matplotlib.pyplot as plt


# ── RUTAS ─────────────────────────────────────────────────────────────
# Resueltas desde este archivo, así los scripts funcionan sin importar desde qué
# carpeta se lancen. scriptsPython/ vive en PyTorch/, dos niveles sobre este módulo.
COMUN_DIR        = os.path.dirname(os.path.abspath(__file__))
SCRIPTSPY_DIR    = os.path.dirname(COMUN_DIR)
PYTORCH_DIR      = os.path.dirname(SCRIPTSPY_DIR)
PROJECT_DIR      = os.path.dirname(PYTORCH_DIR)

DATASET_DIR      = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR        = os.path.join(DATASET_DIR, "train")
VALID_DIR        = os.path.join(DATASET_DIR, "validation")
TEST_DIR         = os.path.join(DATASET_DIR, "test")
GARBAGE_DIR      = os.path.join(PROJECT_DIR, "dataset_gd")     # solo para el modelo binario

PESOS_DIR        = os.path.join(SCRIPTSPY_DIR, "pesos")
RESULTADOS_DIR   = os.path.join(SCRIPTSPY_DIR, "resultados")
GRAFICOS_DIR     = os.path.join(RESULTADOS_DIR, "graficos")
CM_DIR           = os.path.join(RESULTADOS_DIR, "matriz_de_confusion")
REPORTE_DIR      = os.path.join(RESULTADOS_DIR, "reporte")

ORIGINAL_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
EXTENSIONES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

IMG_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]      # normalización ImageNet, la que espera el backbone
STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generador propio del DataLoader: se reinicia dentro de semilla() para que el orden
# del shuffle sea el mismo en cada trial y los trials sean comparables entre sí.
generador = torch.Generator()


def crear_carpetas_resultados():
    for d in (PESOS_DIR, GRAFICOS_DIR, CM_DIR, REPORTE_DIR):
        os.makedirs(d, exist_ok=True)


def semilla(valor: int = 42) -> None:
    """
    Reinicia TODAS las fuentes de aleatoriedad del entrenamiento.

    Se llama al inicio de cada trial de Optuna y de cada reentrenamiento. Sin esto los
    trials arrastran el estado del RNG del anterior, así que la misma configuración da
    resultados distintos según en qué posición de la búsqueda le toque correr — y esa
    varianza se confunde con diferencias reales entre hiperparámetros.
    """
    random.seed(valor)
    np.random.seed(valor)
    torch.manual_seed(valor)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(valor)
    generador.manual_seed(valor)


def limpiar_memoria():
    """Equivalente a keras.backend.clear_session() + gc.collect()."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def contar_imagenes(directorio_split, clases):
    """
    Cuenta las imágenes por clase filtrando por extensión.

    Importa filtrar: `len(os.listdir(...))` contaría también cualquier archivo suelto
    que haya en la carpeta (hoy hay un `hola.md` en `dataset/train/metal/`), y esos
    conteos alimentan los pesos de clase. `ImageFolder` sí ignora los no-imagen, así
    que sin el filtro los pesos no corresponderían al dataset realmente cargado.
    """
    return {
        clase: sum(
            1 for archivo in os.listdir(os.path.join(directorio_split, clase))
            if archivo.lower().endswith(EXTENSIONES)
        )
        for clase in clases
    }


# ── TRANSFORMS / AUMENTO DE DATOS ────────────────────────────────────────
def crear_transforms(img_size=IMG_SIZE):
    """
    train_tf incluye aumento de datos (equivalente a las capas RandomFlip/Rotation/
    Zoom/Contrast/Translation de Keras); eval_tf solo redimensiona y normaliza.
    """
    aumento = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=36),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10)),
        transforms.ColorJitter(contrast=0.10),
    ]
    normalizacion = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]

    train_tf = transforms.Compose([transforms.Resize(img_size), *aumento, *normalizacion])
    eval_tf  = transforms.Compose([transforms.Resize(img_size), *normalizacion])
    return train_tf, eval_tf


# ── MODELO ────────────────────────────────────────────────────────────
class TrashNetMobileNetV3(nn.Module):
    """
    MobileNetV3Small(preentrenada en ImageNet) -> GlobalAveragePooling2D
    -> Dense(dense_units, relu) -> Dropout -> Dense(n_classes)

    La softmax NO va en el modelo: CrossEntropyLoss la aplica internamente sobre los
    logits. Para obtener probabilidades se usa torch.softmax en la evaluación.
    """

    def __init__(self, dense_units, dropout_rate, n_classes):
        super().__init__()

        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        self.features = base.features                    # backbone preentrenado
        n_features = base.classifier[0].in_features       # 576 en MobileNetV3Small

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


def congelar_backbone(model):
    """Fase 1: el backbone completo queda congelado."""
    for p in model.features.parameters():
        p.requires_grad = False


def descongelar_ultimos_bloques(model, n_bloques=3):
    """
    Fase 2: se descongelan solo los últimos bloques del backbone.

    torchvision agrupa el backbone en 13 bloques (features[0..12]); descongelar los
    últimos 3 es el equivalente razonable a "las últimas 20 capas" de Keras. Las
    BatchNorm quedan siempre congeladas (parámetros y estadísticas), igual que
    `layer.trainable = False` sobre una BatchNorm en Keras.
    """
    for p in model.features.parameters():
        p.requires_grad = False

    for bloque in model.features[-n_bloques:]:
        for p in bloque.parameters():
            p.requires_grad = True

    for m in model.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                p.requires_grad = False


def construir_optimizador(model, optimizer_name, learning_rate):
    entrenables = [p for p in model.parameters() if p.requires_grad]
    nombre = optimizer_name.lower()
    if nombre == "adam":
        return torch.optim.Adam(entrenables, lr=learning_rate)
    if nombre == "rmsprop":
        return torch.optim.RMSprop(entrenables, lr=learning_rate)
    raise ValueError(f"Optimizador no soportado: {optimizer_name}")


def construir_modelo(dense_units, dropout_rate, optimizer_name, learning_rate,
                     n_classes, nombre_modelo, class_weights_tensor=None):
    model = TrashNetMobileNetV3(dense_units, dropout_rate, n_classes).to(DEVICE)
    model.nombre = nombre_modelo

    congelar_backbone(model)                     # Fase 1: backbone congelado
    optimizer = construir_optimizador(model, optimizer_name, learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    return model, optimizer, criterion


def resumen_modelo(model):
    total = sum(p.numel() for p in model.parameters())
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo: {getattr(model, 'nombre', model.__class__.__name__)}")
    print(f"Parámetros totales:     {total:,}")
    print(f"Parámetros entrenables: {entrenables:,}")
    print(f"Parámetros congelados:  {total - entrenables:,}")


# ── ENTRENAMIENTO ─────────────────────────────────────────────────────
def entrenar_epoca(model, loader, criterion, optimizer):
    model.train()
    model.features.eval()   # backbone siempre en inferencia: deja las BatchNorm con
                             # sus estadísticas de ImageNet, en las dos fases

    perdida, aciertos, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        perdida += loss.item() * labels.size(0)
        aciertos += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return perdida / total, aciertos / total


@torch.no_grad()
def evaluar(model, loader, criterion=None):
    """Devuelve loss, accuracy, y_true (numpy) e y_prob (numpy, softmax)."""
    model.eval()

    perdida, total = 0.0, 0
    y_true_all, y_prob_all = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels_dev = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)
        if criterion is not None:
            perdida += criterion(outputs, labels_dev).item() * labels.size(0)
        total += labels.size(0)

        y_prob_all.append(torch.softmax(outputs, dim=1).cpu().numpy())
        y_true_all.append(labels.numpy())

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    y_pred = y_prob.argmax(1)

    return {
        "loss": perdida / total if criterion is not None else float("nan"),
        "accuracy": accuracy_score(y_true, y_pred),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def entrenar(model, optimizer, criterion, train_loader, valid_loader, epochs,
            patience_es, patience_lr, min_lr, metrica_fn=None, verbose=1):
    """
    Bucle de entrenamiento con EarlyStopping + ReduceLROnPlateau reimplementados a
    mano (PyTorch no trae callbacks).

    - EarlyStopping: monitoriza `metrica_fn(val_evaluacion)` (por defecto val_accuracy),
      modo maximizar, restaura los mejores pesos al terminar.
    - ReduceLROnPlateau: sobre val_loss, factor 0.5.

    `metrica_fn` permite reusar este mismo bucle para el modelo binario, que
    monitoriza average precision en vez de accuracy (ver modelos/binario/entrenar.py).

    Devuelve un history con accuracy, val_accuracy, loss, val_loss y val_metrica.
    """
    if metrica_fn is None:
        metrica_fn = lambda val: val["accuracy"]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience_lr, min_lr=min_lr
    )

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": [], "val_metrica": []}

    mejor_metrica = -np.inf
    mejores_pesos = copy.deepcopy(model.state_dict())
    sin_mejora = 0
    inicio = time.time()

    for epoca in range(epochs):
        lr_antes = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = entrenar_epoca(model, train_loader, criterion, optimizer)
        val = evaluar(model, valid_loader, criterion)
        val_metrica = metrica_fn(val)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        history["val_metrica"].append(val_metrica)

        scheduler.step(val["loss"])
        lr_despues = optimizer.param_groups[0]["lr"]

        if val_metrica > mejor_metrica:
            mejor_metrica = val_metrica
            mejores_pesos = copy.deepcopy(model.state_dict())
            sin_mejora = 0
        else:
            sin_mejora += 1

        if verbose:
            print(
                f"Época {epoca + 1:02d}/{epochs} | loss {train_loss:.4f} "
                f"acc {train_acc:.4f} | val_loss {val['loss']:.4f} "
                f"val_acc {val['accuracy']:.4f} val_metrica {val_metrica:.4f} | "
                f"lr {lr_despues:.2e}"
            )
            if lr_despues < lr_antes:
                print(f"  ReduceLROnPlateau: {lr_antes:.2e} -> {lr_despues:.2e}")

        if sin_mejora >= patience_es:
            if verbose:
                print(f"EarlyStopping en la época {epoca + 1}: "
                      f"{patience_es} épocas sin mejorar")
            break

    model.load_state_dict(mejores_pesos)

    if verbose:
        print(f"Mejor métrica en validación: {mejor_metrica:.4f} "
              f"(pesos restaurados) — {time.time() - inicio:.1f}s")

    return history


def unir_historiales(h1, h2):
    """Concatena las dos fases de entrenamiento en un solo history."""
    resultado = {clave: h1[clave] + h2[clave] for clave in h1}
    resultado["inicio_fine_tuning"] = len(h1["accuracy"])
    return resultado


# ── MÉTRICAS MANUALES (sin scikit-learn, ver CONTEXT.md §10) ──────────
def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for real, predicho in zip(y_true, y_pred):
        cm[real][predicho] += 1
    return cm


def metricas_por_clase(cm):
    metricas = {}
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metricas[i] = {
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "support": int(cm[i, :].sum()),
        }
    return metricas


def accuracy_score(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def balanced_accuracy(y_true, y_pred, n):
    """Media de los recalls por clase: el modelo trivial que siempre predice la clase
    mayoritaria saca 0.5 en un problema binario, no ~0.94 como el accuracy normal."""
    cm = confusion_matrix(y_true, y_pred, n)
    m = metricas_por_clase(cm)
    return float(np.mean([m[i]["recall"] for i in range(n)]))


def macro(metricas, clave):
    return float(np.mean([m[clave] for m in metricas.values()]))


def label_binarize(y, n_clases):
    return np.eye(n_clases, dtype=int)[np.asarray(y)]


def roc_curve(y_true_bin, y_score):
    """ROC one-vs-rest recorriendo todos los umbrales presentes en y_score. AUC por
    regla del trapecio."""
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    P = int(y_true_bin.sum())
    N = len(y_true_bin) - P
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    orden = np.argsort(-y_score, kind="mergesort")
    y, s = y_true_bin[orden], y_score[orden]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    ultimos = np.r_[np.diff(s) != 0, True]

    tpr = np.r_[0.0, tp[ultimos] / P, 1.0]
    fpr = np.r_[0.0, fp[ultimos] / N, 1.0]

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return fpr, tpr, abs(float(_trapz(tpr, fpr)))


def roc_auc_macro(y_true, y_prob, n_clases):
    """Macro-average AUC one-vs-rest, equivalente a roc_auc_score(multi_class='ovr')."""
    y_bin = label_binarize(y_true, n_clases)
    aucs = [roc_curve(y_bin[:, i], y_prob[:, i])[2] for i in range(n_clases)]
    return float(np.mean(aucs))


def precision_recall_curve(y_true_bin, y_score):
    """
    Curva precision-recall de la clase de interés y su average precision (AP).
    AP = suma de (R_n - R_{n-1}) * P_n, sin interpolación, igual que
    sklearn.metrics.average_precision_score. Se usa en el modelo binario, donde la
    ROC es engañosamente optimista bajo desbalance fuerte (ver CONTEXT.md §10).
    """
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    P = int(y_true_bin.sum())
    if P == 0:
        return np.array([1.0]), np.array([0.0]), np.array([]), 0.0

    orden = np.argsort(-y_score, kind="mergesort")
    y, s = y_true_bin[orden], y_score[orden]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    ultimos = np.r_[np.diff(s) != 0, True]
    tp, fp, umbrales = tp[ultimos], fp[ultimos], s[ultimos]

    precision = tp / (tp + fp)
    recall = tp / P
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    return precision, recall, umbrales, ap


def classification_report(y_true, y_pred, target_names, digits=4):
    """Reporte con el mismo formato que sklearn.metrics.classification_report."""
    cm = confusion_matrix(y_true, y_pred, len(target_names))
    metricas = metricas_por_clase(cm)
    total = int(cm.sum())
    ancho = max(len(n) for n in target_names) + 2

    lineas = [
        f"{'':>{ancho}}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}",
        "",
    ]
    for i, nombre in enumerate(target_names):
        m = metricas[i]
        lineas.append(
            f"{nombre:>{ancho}}{m['precision']:>12.{digits}f}"
            f"{m['recall']:>12.{digits}f}{m['f1']:>12.{digits}f}{m['support']:>12d}"
        )

    lineas.append("")
    lineas.append(
        f"{'accuracy':>{ancho}}{'':>12}{'':>12}"
        f"{accuracy_score(y_true, y_pred):>12.{digits}f}{total:>12d}"
    )
    lineas.append(
        f"{'macro avg':>{ancho}}{macro(metricas, 'precision'):>12.{digits}f}"
        f"{macro(metricas, 'recall'):>12.{digits}f}{macro(metricas, 'f1'):>12.{digits}f}{total:>12d}"
    )

    pesos = np.array([metricas[i]["support"] for i in range(len(target_names))])
    lineas.append(
        f"{'weighted avg':>{ancho}}"
        + "".join(
            f"{float(np.average([metricas[i][k] for i in range(len(target_names))], weights=pesos)):>12.{digits}f}"
            for k in ("precision", "recall", "f1")
        )
        + f"{total:>12d}"
    )
    return "\n".join(lineas)


# ── OPTUNA: baseline como trial 0 + reentrenamiento del top-K ─────────
# Metodología corregida documentada en PyTorch/CONTEXT.md §9: con solo 10 trials
# TPESampler nunca sale de su fase de arranque aleatorio (n_startup_trials=10 por
# defecto), el baseline nunca se compara contra nada, y el ranking de una búsqueda
# corta no predice el resultado a presupuesto completo. Las tres correcciones:
#   1. trials > startup_trials, para que el TPE se use de verdad
#   2. encolar el baseline como trial 0 con study.enqueue_trial
#   3. reentrenar el top-K a presupuesto completo y elegir con eso, no con el ranking corto
def fuente_trial(trial, startup_trials):
    if trial.number == 0:
        return "baseline"
    return "azar" if trial.number < startup_trials else "tpe"


def reentrenar_topk(study, top_k, construir_modelo_fn, entrenar_fn,
                    epochs_fase1, epochs_fase2, fine_tune_lr, patience_lr, verbose=1):
    """
    Reentrena los `top_k` mejores trials (por valor de búsqueda) a presupuesto
    completo (fase 1 + fase 2) y devuelve la lista de resultados, cada uno con su
    state_dict en CPU (para no acumular modelos en la VRAM).
    """
    import optuna

    candidatos = sorted(
        [t for t in study.trials
         if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None],
        key=lambda t: t.value, reverse=True
    )[:top_k]

    resultados = []
    for puesto, trial in enumerate(candidatos, start=1):
        if verbose:
            print(f"\n[{puesto}/{len(candidatos)}] Trial {trial.number} "
                  f"— valor de búsqueda: {trial.value:.4f} — {trial.params}")

        semilla()
        limpiar_memoria()

        model_c, optimizer_c, criterion_c = construir_modelo_fn(trial.params)

        h1 = entrenar_fn(model_c, optimizer_c, criterion_c, epochs=epochs_fase1,
                         patience_es=6, patience_lr=patience_lr, min_lr=1e-7, verbose=verbose)

        descongelar_ultimos_bloques(model_c, n_bloques=3)
        optimizer_c2 = construir_optimizador(model_c, trial.params["optimizer"], fine_tune_lr)

        h2 = entrenar_fn(model_c, optimizer_c2, criterion_c, epochs=epochs_fase2,
                         patience_es=5, patience_lr=2, min_lr=1e-7, verbose=verbose)

        valor_completo = max(h1["val_metrica"] + h2["val_metrica"])

        resultados.append({
            "trial": trial.number,
            "params": dict(trial.params),
            "valor_busqueda": float(trial.value),
            "valor_completo": float(valor_completo),
            "history_fase1": h1,
            "history_fase2": h2,
            "state_dict": {k: v.detach().cpu().clone() for k, v in model_c.state_dict().items()},
        })

        del model_c, optimizer_c, optimizer_c2
        limpiar_memoria()

    return resultados


# ── GRÁFICOS ──────────────────────────────────────────────────────────
def graficar_curvas(hist, titulo, archivo, extra_paneles=None):
    """
    Loss y accuracy de train/val por época, con línea vertical en el inicio del
    fine-tuning. `extra_paneles` agrega paneles adicionales de solo-validación, p.ej.
    [(None, "val_ap", "Average precision")] para el modelo binario.
    """
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
    y_bin = label_binarize(y_true, n)

    plt.figure(figsize=(9, 7))
    aucs = {}
    for i, clase in enumerate(class_names):
        fpr, tpr, auc = roc_curve(y_bin[:, i], y_prob[:, i])
        aucs[clase] = auc
        plt.plot(fpr, tpr, label=f"{clase} (AUC={auc:.3f})")

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

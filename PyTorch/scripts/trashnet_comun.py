"""
Módulo común para los scripts de PyTorch de TrashNet.

Aquí vive todo lo que comparten la búsqueda de hiperparámetros (optuna_trashnet.py)
y el reentrenamiento final (entrenar_mejor.py):

  - Rutas del proyecto y carga del dataset (ImageFolder)
  - TrashNetCNN: la misma arquitectura del notebook pero PARAMETRIZABLE,
    de modo que Optuna pueda variar profundidad, filtros, dropout, etc.
  - Bucle de entrenamiento / evaluación de una época
  - Métricas, matriz de confusión y curvas ROC calculadas a mano con NumPy
    (igual que en el notebook: este entorno no tiene scikit-learn instalado)

No se ejecuta por sí solo, se importa.
"""

from __future__ import annotations

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")            # backend sin ventana: los scripts guardan PNG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── RUTAS ─────────────────────────────────────────────────────────────
# Se resuelven desde la ubicación de este archivo, así los scripts funcionan
# sin importar desde qué carpeta los lances.
SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
PYTORCH_DIR  = os.path.dirname(SCRIPTS_DIR)
PROJECT_DIR  = os.path.dirname(PYTORCH_DIR)

DATASET_DIR  = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR    = os.path.join(DATASET_DIR, "train")
VALID_DIR    = os.path.join(DATASET_DIR, "validation")
TEST_DIR     = os.path.join(DATASET_DIR, "test")

MODELOS_DIR      = os.path.join(PYTORCH_DIR, "modelos")
PREDICCIONES_DIR = os.path.join(PYTORCH_DIR, "predicciones")   # salidas del notebook
OPTUNA_DIR       = os.path.join(PYTORCH_DIR, "optuna")         # study.db + mejores_params.json

# Todas las salidas de los scripts viven bajo PyTorch/resultados/, con la misma
# convención de subcarpetas que la carpeta resultados/ del equipo en la raíz.
# Se mantienen separadas a propósito: esto es el trabajo de PyTorch, no el informe
# conjunto del equipo.
RESULTADOS_DIR = os.path.join(PYTORCH_DIR, "resultados")
GRAFICOS_DIR   = os.path.join(RESULTADOS_DIR, "graficos")
CM_DIR         = os.path.join(RESULTADOS_DIR, "matriz_de_confusion")
REPORTE_DIR    = os.path.join(RESULTADOS_DIR, "reporte")


def crear_carpetas_resultados():
    """Crea PyTorch/resultados/{graficos,matriz_de_confusion,reporte} si no existen."""
    for d in (GRAFICOS_DIR, CM_DIR, REPORTE_DIR):
        os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalización estándar de ImageNet (útil si algún día se usa transfer learning;
# para una CNN desde cero simplemente estabiliza la escala de entrada).
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def semilla(seed: int = 42) -> None:
    """Fija las semillas para que los trials sean comparables entre sí."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── DATOS ─────────────────────────────────────────────────────────────
def construir_loaders(img_size=150, batch_size=16, aug="media",
                      num_workers=2, seed=42):
    """
    Devuelve (train_loader, valid_loader, test_loader, class_names).

    `aug` controla la intensidad del aumento de datos, que también es un
    hiperparámetro interesante: con ~1770 imágenes de entrenamiento el
    overfitting aparece rápido, pero demasiada distorsión frena el aprendizaje.
    """
    resize = transforms.Resize((img_size, img_size))
    normal = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    if aug == "ninguna":
        train_extra = []
    elif aug == "leve":
        train_extra = [
            transforms.RandomHorizontalFlip(),
        ]
    elif aug == "media":
        train_extra = [
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
        ]
    elif aug == "fuerte":
        train_extra = [
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        ]
    else:
        raise ValueError(f"aug desconocido: {aug}")

    train_tf = transforms.Compose([resize, *train_extra, normal])
    eval_tf  = transforms.Compose([resize, normal])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    valid_ds = datasets.ImageFolder(VALID_DIR, transform=eval_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

    g = torch.Generator()
    g.manual_seed(seed)

    comun = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                 persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=g, drop_last=False, **comun)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **comun)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **comun)

    return train_loader, valid_loader, test_loader, train_ds.classes


def pesos_de_clase(class_names, loader_dir=TRAIN_DIR):
    """
    Pesos inversamente proporcionales a la frecuencia de cada clase.

    El dataset está desbalanceado (paper ~415 imágenes vs trash ~95), por lo que
    un modelo puede ganar accuracy ignorando 'trash'. Pasar estos pesos a
    CrossEntropyLoss penaliza más los errores en las clases minoritarias.
    """
    conteos = np.array([
        len(os.listdir(os.path.join(loader_dir, c))) for c in class_names
    ], dtype=np.float64)
    pesos = conteos.sum() / (len(conteos) * conteos)
    return torch.tensor(pesos, dtype=torch.float32)


# ── MODELO ────────────────────────────────────────────────────────────
class TrashNetCNN(nn.Module):
    """
    Versión parametrizable de la CNN del notebook.

    Diferencia clave frente a la original: en lugar de un Flatten de tamaño fijo
    (64*17*17, que obliga a img_size=150 y a exactamente 3 bloques) se usa
    AdaptiveAvgPool2d, así el clasificador funciona con cualquier combinación de
    `img_size` y `n_bloques` que proponga Optuna.

    Cada bloque: Conv -> [BatchNorm] -> ReLU -> MaxPool
    Los filtros se duplican por bloque: base, base*2, base*4, ...
    """

    def __init__(self, num_classes=6, n_bloques=3, filtros_base=16,
                 usar_bn=True, dropout=0.3, units_fc=128, activacion="relu"):
        super().__init__()

        act = {"relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activacion]

        capas = []
        in_ch = 3
        for b in range(n_bloques):
            out_ch = filtros_base * (2 ** b)
            capas.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if usar_bn:
                capas.append(nn.BatchNorm2d(out_ch))
            capas.append(act())
            capas.append(nn.MaxPool2d(2))
            in_ch = out_ch

        self.features = nn.Sequential(*capas)
        self.pool     = nn.AdaptiveAvgPool2d((4, 4))     # salida fija: in_ch*4*4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 4 * 4, units_fc),
            act(),
            nn.Dropout(dropout),
            nn.Linear(units_fc, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def construir_modelo(num_classes, params):
    """Crea el modelo a partir de un dict de hiperparámetros (el de Optuna)."""
    return TrashNetCNN(
        num_classes  = num_classes,
        n_bloques    = params["n_bloques"],
        filtros_base = params["filtros_base"],
        usar_bn      = params["usar_bn"],
        dropout      = params["dropout"],
        units_fc     = params["units_fc"],
        activacion   = params["activacion"],
    ).to(DEVICE)


def construir_optimizador(model, params):
    """Optimizador según el hiperparámetro `optimizer` + lr + weight_decay."""
    lr = params["lr"]
    wd = params.get("weight_decay", 0.0)
    nombre = params["optimizer"]
    if nombre == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if nombre == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if nombre == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    if nombre == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"optimizer desconocido: {nombre}")


# ── ENTRENAMIENTO ─────────────────────────────────────────────────────
def correr_epoca(model, loader, criterion, optimizer=None):
    """
    Una pasada completa por `loader`. Si `optimizer` es None -> modo evaluación.
    Devuelve (loss_promedio, accuracy).
    """
    entrenando = optimizer is not None
    model.train() if entrenando else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(entrenando):
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if entrenando:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if entrenando:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def obtener_predicciones(model, loader):
    """Devuelve (y_true, y_pred, y_prob) con las probabilidades softmax."""
    model.eval()
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            probs  = torch.softmax(model(images), dim=1).cpu().numpy()
            y_true_all.append(labels.numpy())
            y_prob_all.append(probs)
    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    return y_true, np.argmax(y_prob, axis=1), y_prob


# ── MÉTRICAS MANUALES (sin sklearn) ───────────────────────────────────
def calcular_cm(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def calcular_metricas_por_clase(cm):
    metricas = {}
    for i in range(len(cm)):
        tp  = cm[i, i]
        fp  = cm[:, i].sum() - tp
        fn  = cm[i, :].sum() - tp
        sup = cm[i, :].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metricas[i] = {"precision": prec, "recall": rec, "f1": f1, "support": int(sup)}
    return metricas


def macro_f1_desde_cm(cm):
    m = calcular_metricas_por_clase(cm)
    return float(np.mean([m[i]["f1"] for i in range(len(cm))]))


def roc_curve_manual(y_true_bin, y_score):
    """
    ROC one-vs-rest recorriendo todos los umbrales presentes en y_score.
    AUC por regla del trapecio.
    """
    thresholds = np.sort(np.unique(y_score))[::-1]
    P = int(y_true_bin.sum())
    N = len(y_true_bin) - P
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    fprs, tprs = [0.0], [0.0]
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tp = int(((pred == 1) & (y_true_bin == 1)).sum())
        fp = int(((pred == 1) & (y_true_bin == 0)).sum())
        fprs.append(fp / N)
        tprs.append(tp / P)
    fprs.append(1.0)
    tprs.append(1.0)

    fpr, tpr = np.array(fprs), np.array(tprs)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy>=2 renombro trapz
    return fpr, tpr, abs(float(_trapz(tpr, fpr)))


def macro_auc(y_true, y_prob, n_clases):
    """Macro-average AUC one-vs-rest. Se usa como métrica objetivo en Optuna."""
    y_bin = np.eye(n_clases)[y_true]
    aucs = [roc_curve_manual(y_bin[:, i], y_prob[:, i])[2] for i in range(n_clases)]
    return float(np.mean(aucs))


# ── GRÁFICOS ──────────────────────────────────────────────────────────
def plot_matriz_confusion(cm, metricas, class_names, acc, output,
                          titulo="TrashNet (PyTorch) — Evaluación en test"):
    """Matriz de confusión + tabla de métricas + barras de F1 (igual al notebook)."""
    n = len(class_names)
    fig = plt.figure(figsize=(16, 7))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], figure=fig)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Etiqueta real", fontsize=12)
    ax.set_title(f"Matriz de Confusión — Test\nAccuracy: {acc:.1%}", fontsize=13, pad=12)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#1E7B45", linewidth=2.5))

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.text(0.05, 0.97, f"Accuracy global: {acc:.2%}", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", color="#1F3864", va="top")

    headers = ["Clase", "Precision", "Recall", "F1", "N"]
    filas = [[class_names[i],
              f"{metricas[i]['precision']:.3f}",
              f"{metricas[i]['recall']:.3f}",
              f"{metricas[i]['f1']:.3f}",
              str(metricas[i]["support"])] for i in range(n)]
    macro_p = np.mean([metricas[i]["precision"] for i in range(n)])
    macro_r = np.mean([metricas[i]["recall"]    for i in range(n)])
    macro_f = np.mean([metricas[i]["f1"]        for i in range(n)])
    filas.append(["macro avg", f"{macro_p:.3f}", f"{macro_r:.3f}", f"{macro_f:.3f}", ""])

    tbl = ax2.table(cellText=filas, colLabels=headers,
                    loc="upper center", bbox=[0.0, 0.42, 1.0, 0.52])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1F3864")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == len(filas):
            cell.set_facecolor("#D6E4F0")
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EEF4FB")
        cell.set_edgecolor("#CCCCCC")

    ax2.text(0.05, 0.38, "F1-Score por clase", transform=ax2.transAxes,
             fontsize=11, fontweight="bold", color="#1F3864")
    colores = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (clase, color) in enumerate(zip(class_names, colores)):
        f1 = metricas[i]["f1"]
        y  = 0.32 - i * 0.05
        ax2.add_patch(plt.Rectangle((0.05, y), f1 * 0.88, 0.04,
                                    transform=ax2.transAxes, facecolor=color, alpha=0.8))
        ax2.text(0.05 + f1 * 0.88 + 0.02, y + 0.013, f"{clase}  {f1:.2f}",
                 transform=ax2.transAxes, fontsize=10, va="center")

    plt.suptitle(titulo, fontsize=14, fontweight="bold", color="#1F3864", y=1.01)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {output}")


def plot_roc(y_true, y_prob, class_names, output,
             titulo="TrashNet (PyTorch) — Curvas ROC (One-vs-Rest)"):
    """
    Un panel ROC por clase (con el umbral óptimo de Youden marcado) más un panel
    final con todas las curvas superpuestas y el macro-AUC.
    Devuelve {clase: auc}.
    """
    n = len(class_names)
    y_bin   = np.eye(n)[y_true]
    colores = plt.cm.tab10(np.linspace(0, 1, n))

    n_paneles = n + 1
    cols = 3
    rows = int(np.ceil(n_paneles / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes_flat = list(np.array(axes).flat)

    auc_por_clase, curvas = {}, {}

    for i, (clase, color) in enumerate(zip(class_names, colores)):
        fpr, tpr, auc = roc_curve_manual(y_bin[:, i], y_prob[:, i])
        auc_por_clase[clase] = auc
        curvas[clase] = (fpr, tpr)

        ax = axes_flat[i]
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatorio")
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"ROC — {clase}", fontsize=12, fontweight="bold", color=color)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)

        # Umbral óptimo por índice de Youden (maximiza TPR - FPR)
        idx_opt = int(np.argmax(tpr - fpr))
        ax.scatter(fpr[idx_opt], tpr[idx_opt], color=color, s=80, zorder=5)
        ax.annotate(f"  Umbral óptimo\n  ({fpr[idx_opt]:.2f}, {tpr[idx_opt]:.2f})",
                    xy=(fpr[idx_opt], tpr[idx_opt]), fontsize=8, color=color)

    ax_all = axes_flat[n]
    for (clase, color) in zip(class_names, colores):
        fpr, tpr = curvas[clase]
        ax_all.plot(fpr, tpr, color=color, lw=2,
                    label=f"{clase}  (AUC={auc_por_clase[clase]:.3f})")
    m_auc = float(np.mean(list(auc_por_clase.values())))
    ax_all.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_all.set_xlim(-0.02, 1.02); ax_all.set_ylim(-0.02, 1.02)
    ax_all.set_xlabel("False Positive Rate", fontsize=11)
    ax_all.set_ylabel("True Positive Rate", fontsize=11)
    ax_all.set_title(f"Todas las clases\nMacro-avg AUC = {m_auc:.3f}",
                     fontsize=12, fontweight="bold")
    ax_all.legend(loc="lower right", fontsize=9)
    ax_all.grid(alpha=0.3)

    for j in range(n_paneles, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle(titulo, fontsize=14, fontweight="bold", color="#1F3864")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {output}")
    return auc_por_clase


def plot_curvas_entrenamiento(history, output):
    """Accuracy y loss de train/val por época."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epocas = range(1, len(history["accuracy"]) + 1)

    ax1.plot(epocas, history["accuracy"],     "-o", ms=4, label="Train")
    ax1.plot(epocas, history["val_accuracy"], "-o", ms=4, label="Validación")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epocas, history["loss"],     "-o", ms=4, label="Train")
    ax2.plot(epocas, history["val_loss"], "-o", ms=4, label="Validación")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Loss")
    ax2.set_title("Loss", fontsize=13, fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Modelo final — curvas de entrenamiento",
                 fontsize=14, fontweight="bold", color="#1F3864")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {output}")

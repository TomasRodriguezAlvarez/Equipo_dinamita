"""
Modelo 3 de la segunda entrega — Clasificador binario reciclable / basura.
Es el modelo DESEQUILIBRADO que pide la pauta (punto 5): solo 5.4% de las imágenes de
train son `basura` (trash), el resto `reciclable`.

Versión script de `PyTorch/binario/TrashNet_PyTorch_Binario_DEFINITIVO.ipynb`.

Lo que lo distingue de los otros dos modelos:
  1. El accuracy no sirve: "siempre reciclable" saca ~94% sin detectar nada. Se
     monitoriza y optimiza la average precision (AP) sobre la clase `basura`.
  2. El umbral se calibra sobre validación (no se usa argmax = umbral 0.5).
  3. Evaluación fuera de distribución (Parte D) contra dataset_gd/, si existe: mide si
     el modelo generaliza más allá de las fotos de estudio de TrashNet.

Uso:
    python entrenar.py
    python entrenar.py --sin-optuna
    python entrenar.py --recall-minimo 0.85

Genera en scriptsPython/pesos/: binario_sin_optuna.pt, binario_optuna.pt
Y en scriptsPython/resultados/: gráficos, matrices de confusión, curvas PR/ROC y
reporte con prefijo `bin_`.
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

# parents[2] = scriptsPython/ (este archivo vive en scriptsPython/modelos/binario/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comun import utils as u

MAPEO_NOMBRE = {
    "cardboard": "reciclable", "glass": "reciclable", "metal": "reciclable",
    "paper": "reciclable", "plastic": "reciclable", "trash": "basura",
}
CLASSES = ["basura", "reciclable"]      # basura = índice 0, clase de interés
N_CLASSES = 2
IDX_BASURA = 0
ORIGINAL_TO_BINARIO = [CLASSES.index(MAPEO_NOMBRE[c]) for c in u.ORIGINAL_CLASSES]

BATCH_SIZE = 32
EPOCHS_FASE1 = 25
EPOCHS_FASE2 = 15
FINE_TUNE_LR = 1e-5
NUM_WORKERS = 0

BASE_DENSE_UNITS = 128
BASE_DROPOUT = 0.35
BASE_LR = 3e-4
BASE_OPTIMIZER = "adam"


def remapear_a_binario(indice_original):
    return ORIGINAL_TO_BINARIO[indice_original]


def cargar_datos(workers):
    train_tf, eval_tf = u.crear_transforms(u.IMG_SIZE)

    def cargar(ruta, transform):
        ds = datasets.ImageFolder(ruta, transform=transform, target_transform=remapear_a_binario)
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
        conteos[remapear_a_binario(u.ORIGINAL_CLASSES.index(c))] += n
    pesos = conteos.sum() / (N_CLASSES * conteos)
    class_weights_tensor = torch.tensor(pesos, dtype=torch.float32, device=u.DEVICE)

    print(f"Clases: {CLASSES} (clase de interés: {CLASSES[IDX_BASURA]})")
    print(f"Desbalance en train: {conteos[1] / conteos[0]:.1f}:1 "
          f"({int(conteos[0])} basura / {int(conteos[1])} reciclable)")
    print(f"Train: {len(train_ds)} | Validation: {len(valid_ds)} | Test: {len(test_ds)}")
    return train_loader, valid_loader, test_loader, class_weights_tensor


def metrica_ap(val_eval_result):
    """Callback para u.entrenar(): la métrica a maximizar es la AP sobre `basura`,
    no la accuracy. Con 5.4% de positivos la accuracy se estanca en ~0.95 desde la
    primera época y no distingue nada."""
    p_basura = val_eval_result["y_prob"][:, IDX_BASURA]
    y_bin = (val_eval_result["y_true"] == IDX_BASURA).astype(int)
    _, _, _, ap = u.precision_recall_curve(y_bin, p_basura)
    return ap


def construir(dense_units, dropout, optimizer_name, lr, nombre, class_weights):
    return u.construir_modelo(dense_units, dropout, optimizer_name, lr,
                              N_CLASSES, nombre, class_weights)


def entrenar_dos_fases(model, optimizer, criterion, train_loader, valid_loader):
    h1 = u.entrenar(model, optimizer, criterion, train_loader, valid_loader,
                    epochs=EPOCHS_FASE1, patience_es=6, patience_lr=3, min_lr=1e-6,
                    metrica_fn=metrica_ap)
    u.descongelar_ultimos_bloques(model, n_bloques=3)
    optimizer2 = u.construir_optimizador(model, BASE_OPTIMIZER, FINE_TUNE_LR)
    h2 = u.entrenar(model, optimizer2, criterion, train_loader, valid_loader,
                    epochs=EPOCHS_FASE2, patience_es=5, patience_lr=2, min_lr=1e-7,
                    metrica_fn=metrica_ap)
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
                             min_lr=1e-7, metrica_fn=metrica_ap, verbose=0)
        mejor_ap = float(max(history["val_metrica"]))
        print(f"Trial {trial.number:02d} | AP={mejor_ap:.4f} | {trial.params}")

        del model, optimizer
        u.limpiar_memoria()
        return mejor_ap

    params_baseline = {
        "dense_units": BASE_DENSE_UNITS, "dropout_rate": BASE_DROPOUT,
        "learning_rate": BASE_LR, "optimizer": BASE_OPTIMIZER,
    }

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=args.startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="TrashNet_Binario_scriptsPython")
    study.enqueue_trial(params_baseline)

    print(f"\nMétrica optimizada: average precision sobre '{CLASSES[IDX_BASURA]}'")
    print(f"{args.trials} trials ({args.startup_trials} de arranque aleatorio + "
          f"{args.trials - args.startup_trials} informados por el TPE)\n")
    study.optimize(objective, n_trials=args.trials)
    return study, params_baseline


def calibrar_umbral(y_val, p_val_basura):
    """Barre el umbral sobre validación y devuelve el que maximiza F1(basura)."""
    umbrales = np.linspace(0.01, 0.99, 200)
    mejores_f1 = []
    for umb in umbrales:
        y_pred = np.where(p_val_basura >= umb, 0, 1)
        cm = u.confusion_matrix(y_val, y_pred, N_CLASSES)
        m = u.metricas_por_clase(cm)[IDX_BASURA]
        mejores_f1.append(m["f1"])
    idx = int(np.argmax(mejores_f1))
    return float(umbrales[idx]), mejores_f1[idx]


def predecir_con_umbral(p_basura, umbral):
    return np.where(np.asarray(p_basura) >= umbral, 0, 1)


def main():
    args = parse_args()
    u.crear_carpetas_resultados()
    u.semilla()

    print(f"Dispositivo: {u.DEVICE}")
    train_loader, valid_loader, test_loader, class_weights = cargar_datos(args.workers)

    print("\n" + "=" * 70)
    print("PARTE A — Modelo binario base (sin Optuna)")
    print("=" * 70)
    model_base, optimizer_base, criterion = construir(
        BASE_DENSE_UNITS, BASE_DROPOUT, BASE_OPTIMIZER, BASE_LR,
        "binario_sin_optuna", class_weights
    )
    u.resumen_modelo(model_base)
    hist_base = entrenar_dos_fases(model_base, optimizer_base, criterion,
                                   train_loader, valid_loader)

    ruta_base = os.path.join(u.PESOS_DIR, "binario_sin_optuna.pt")
    torch.save({
        "state_dict": model_base.state_dict(), "dense_units": BASE_DENSE_UNITS,
        "dropout_rate": BASE_DROPOUT, "classes": CLASSES, "mapeo": MAPEO_NOMBRE,
        "img_size": u.IMG_SIZE,
    }, ruta_base)
    print(f"\n  {ruta_base}")

    model_optuna = model_base
    hist_optuna = hist_base

    if not args.sin_optuna:
        print("\n" + "=" * 70)
        print("PARTE B — Búsqueda de hiperparámetros con Optuna (métrica: AP)")
        print("=" * 70)
        study, params_baseline = buscar_con_optuna(train_loader, valid_loader, class_weights, args)

        def construir_desde_params(params):
            return construir(params["dense_units"], params["dropout_rate"],
                             params["optimizer"], params["learning_rate"],
                             "candidato", class_weights)

        def entrenar_fase(model, optimizer, criterion, epochs, patience_es, patience_lr, min_lr, verbose=1):
            return u.entrenar(model, optimizer, criterion, train_loader, valid_loader,
                              epochs, patience_es, patience_lr, min_lr,
                              metrica_fn=metrica_ap, verbose=verbose)

        resultados_topk = u.reentrenar_topk(
            study, args.top_k, construir_desde_params, entrenar_fase,
            EPOCHS_FASE1, EPOCHS_FASE2, FINE_TUNE_LR, patience_lr=3,
        )

        ranking = sorted(resultados_topk, key=lambda r: r["valor_completo"], reverse=True)
        ganador = ranking[0]

        print("\nTOP-K REENTRENADO A PRESUPUESTO COMPLETO (AP)")
        for r in ranking:
            print(f"  trial {r['trial']:>3}  AP búsqueda={r['valor_busqueda']:.4f}  "
                  f"AP completo={r['valor_completo']:.4f}  {r['params']}")

        model_optuna = u.TrashNetMobileNetV3(
            ganador["params"]["dense_units"], ganador["params"]["dropout_rate"], N_CLASSES
        ).to(u.DEVICE)
        model_optuna.load_state_dict(ganador["state_dict"])
        model_optuna.nombre = "binario_optuna"
        u.descongelar_ultimos_bloques(model_optuna, n_bloques=3)
        hist_optuna = u.unir_historiales(ganador["history_fase1"], ganador["history_fase2"])

        print(f"\nGANADOR: trial {ganador['trial']} — {ganador['params']}")
        if ganador["params"] == params_baseline:
            print("AVISO: el ganador ES la configuración del baseline.")

        ruta_optuna = os.path.join(u.PESOS_DIR, "binario_optuna.pt")
        torch.save({
            "state_dict": model_optuna.state_dict(), **ganador["params"],
            "trial": ganador["trial"], "classes": CLASSES, "mapeo": MAPEO_NOMBRE,
            "img_size": u.IMG_SIZE,
        }, ruta_optuna)
        print(f"  {ruta_optuna}")

    evaluar_y_reportar(model_base, model_optuna, hist_base, hist_optuna,
                       valid_loader, test_loader, args)


def evaluar_y_reportar(model_base, model_optuna, hist_base, hist_optuna,
                       valid_loader, test_loader, args):
    print("\n" + "=" * 70)
    print("PARTE C — Calibración del umbral y evaluación en test")
    print("=" * 70)

    val_optuna = u.evaluar(model_optuna, valid_loader)
    y_val = val_optuna["y_true"]
    p_val_basura = val_optuna["y_prob"][:, IDX_BASURA]

    UMBRAL, f1_calibrado = calibrar_umbral(y_val, p_val_basura)
    m_05 = u.metricas_por_clase(
        u.confusion_matrix(y_val, predecir_con_umbral(p_val_basura, 0.5), N_CLASSES)
    )[IDX_BASURA]

    print(f"Umbral calibrado en validación (máximo F1): {UMBRAL:.3f}")
    print(f"  F1 con umbral calibrado: {f1_calibrado:.4f}  vs  argmax (0.500): {m_05['f1']:.4f}"
          f"  (ganancia {f1_calibrado - m_05['f1']:+.4f})")

    test_base = u.evaluar(model_base, test_loader)
    test_optuna = u.evaluar(model_optuna, test_loader)
    y_test = test_base["y_true"]
    p_test_base = test_base["y_prob"][:, IDX_BASURA]
    p_test_optuna = test_optuna["y_prob"][:, IDX_BASURA]
    b_test = (y_test == IDX_BASURA).astype(int)

    _, _, _, ap_base = u.precision_recall_curve(b_test, p_test_base)
    _, _, _, ap_optuna = u.precision_recall_curve(b_test, p_test_optuna)

    def evaluar_en_umbral(p, umbral):
        y_pred = predecir_con_umbral(p, umbral)
        cm = u.confusion_matrix(y_test, y_pred, N_CLASSES)
        m = u.metricas_por_clase(cm)[IDX_BASURA]
        return {
            "accuracy": u.accuracy_score(y_test, y_pred),
            "balanced_accuracy": u.balanced_accuracy(y_test, y_pred, N_CLASSES),
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "cm": cm,
        }

    res_argmax = evaluar_en_umbral(p_test_optuna, 0.5)
    res_umbral = evaluar_en_umbral(p_test_optuna, UMBRAL)

    baseline_trivial = float((y_test != IDX_BASURA).mean())

    print("\nRESULTADOS EN TEST — clase de interés: basura")
    print(f"{'modelo':<28}{'accuracy':>10}{'bal.acc':>10}{'prec.':>9}{'recall':>9}{'F1':>9}")
    print(f"{'Trivial todo reciclable':<28}{baseline_trivial:>10.4f}{0.5:>10.4f}"
          f"{0.0:>9.4f}{0.0:>9.4f}{0.0:>9.4f}")
    print(f"{'Con Optuna (argmax)':<28}{res_argmax['accuracy']:>10.4f}"
          f"{res_argmax['balanced_accuracy']:>10.4f}{res_argmax['precision']:>9.4f}"
          f"{res_argmax['recall']:>9.4f}{res_argmax['f1']:>9.4f}")
    print(f"{'Con Optuna (umbral)':<28}{res_umbral['accuracy']:>10.4f}"
          f"{res_umbral['balanced_accuracy']:>10.4f}{res_umbral['precision']:>9.4f}"
          f"{res_umbral['recall']:>9.4f}{res_umbral['f1']:>9.4f}")
    print(f"\nAP en test — sin Optuna {ap_base:.4f} | con Optuna {ap_optuna:.4f}")

    # ── Gráficos ──────────────────────────────────────────────────────
    u.graficar_curvas(
        hist_base, "Binario sin Optuna",
        os.path.join(u.GRAFICOS_DIR, "bin_curvas_sin_optuna.png"),
        extra_paneles=[(None, "val_metrica", "Average precision (basura)")],
    )
    u.graficar_curvas(
        hist_optuna, "Binario con Optuna",
        os.path.join(u.GRAFICOS_DIR, "bin_curvas_optuna.png"),
        extra_paneles=[(None, "val_metrica", "Average precision (basura)")],
    )
    u.graficar_matriz_confusion(
        res_argmax["cm"], CLASSES, "Con Optuna — argmax (umbral 0.500)",
        os.path.join(u.CM_DIR, "bin_confusion_argmax.png"),
    )
    u.graficar_matriz_confusion(
        res_umbral["cm"], CLASSES, f"Con Optuna — umbral calibrado ({UMBRAL:.3f})",
        os.path.join(u.CM_DIR, "bin_confusion_umbral.png"),
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 7))
    for nombre, p in [("Sin Optuna", p_test_base), ("Con Optuna", p_test_optuna)]:
        precision, recall, _, ap = u.precision_recall_curve(b_test, p)
        plt.plot(recall, precision, label=f"{nombre} (AP={ap:.3f})")
    plt.axhline(b_test.mean(), linestyle="--", color="gray", label="Azar (prevalencia)")
    plt.xlabel("Recall (basura)"); plt.ylabel("Precision (basura)")
    plt.title("Curva precision-recall — clase basura (test)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    ruta_pr = os.path.join(u.GRAFICOS_DIR, "bin_pr_test.png")
    plt.savefig(ruta_pr, dpi=120); plt.close()
    print(f"  {ruta_pr}")

    # ── Parte D: evaluación fuera de distribución ───────────────────
    hay_gd = os.path.isdir(os.path.join(u.GARBAGE_DIR, "trash"))
    if hay_gd:
        print("\n" + "=" * 70)
        print("PARTE D — Evaluación fuera de distribución (Garbage Dataset)")
        print("=" * 70)

        _, eval_tf = u.crear_transforms(u.IMG_SIZE)
        gd_ds = datasets.ImageFolder(u.GARBAGE_DIR, transform=eval_tf)
        assert gd_ds.classes == ["trash"], f"Se esperaba solo 'trash': {gd_ds.classes}"
        gd_loader = DataLoader(gd_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=args.workers)

        model_optuna.eval()
        p_gd = []
        with torch.no_grad():
            for images, _ in gd_loader:
                probs = torch.softmax(model_optuna(images.to(u.DEVICE)), dim=1)
                p_gd.append(probs[:, IDX_BASURA].cpu().numpy())
        p_gd = np.concatenate(p_gd)

        recall_gd = float((p_gd >= UMBRAL).mean())
        print(f"Imágenes evaluadas: {len(p_gd)}")
        print(f"Recall en test de TrashNet: {res_umbral['recall']:.4f}")
        print(f"Recall en Garbage Dataset:  {recall_gd:.4f}")
        print(f"Caída:                      {recall_gd - res_umbral['recall']:+.4f}")
    else:
        print(f"\n(Parte D omitida: no se encontró {u.GARBAGE_DIR}/trash)")
        recall_gd = None

    # ── Reporte ───────────────────────────────────────────────────────
    reporte = [
        "TrashNet — Binario reciclable/basura (PyTorch, scriptsPython) — Reporte",
        "=" * 70, f"Dispositivo: {u.DEVICE}", f"Mapeo: {MAPEO_NOMBRE}", "",
        f"Umbral calibrado en validación: {UMBRAL:.3f}", "",
        "TEST — clase basura",
        f"  Trivial 'todo reciclable' : accuracy {baseline_trivial:.4f}  F1 0.0000",
        f"  Con Optuna (argmax)       : precision {res_argmax['precision']:.4f}  "
        f"recall {res_argmax['recall']:.4f}  F1 {res_argmax['f1']:.4f}",
        f"  Con Optuna (umbral)       : precision {res_umbral['precision']:.4f}  "
        f"recall {res_umbral['recall']:.4f}  F1 {res_umbral['f1']:.4f}",
        f"  AP sin Optuna / con Optuna: {ap_base:.4f} / {ap_optuna:.4f}", "",
    ]
    if recall_gd is not None:
        reporte += [
            "Fuera de distribución (Garbage Dataset)",
            f"  Recall en TrashNet test: {res_umbral['recall']:.4f}",
            f"  Recall en GD:            {recall_gd:.4f}",
            f"  Caída:                   {recall_gd - res_umbral['recall']:+.4f}", "",
        ]
    reporte.append(u.classification_report(
        y_test, predecir_con_umbral(p_test_optuna, UMBRAL), CLASSES
    ))

    ruta_reporte = os.path.join(u.REPORTE_DIR, "bin_reporte.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write("\n".join(reporte) + "\n")
    print(f"\n  {ruta_reporte}")
    print("\nListo. Resultados en scriptsPython/resultados/ y pesos en scriptsPython/pesos/")


def parse_args():
    p = argparse.ArgumentParser(description="Entrena el modelo binario reciclable/basura (desequilibrado)")
    p.add_argument("--sin-optuna", action="store_true", help="entrenar solo el baseline")
    p.add_argument("--trials", type=int, default=30, help="trials de Optuna (default 30)")
    p.add_argument("--startup-trials", type=int, default=10, help="trials de arranque aleatorio (default 10)")
    p.add_argument("--epocas-busqueda", type=int, default=8, help="épocas por trial (default 8)")
    p.add_argument("--top-k", type=int, default=3, help="mejores trials a reentrenar (default 3)")
    p.add_argument("--workers", type=int, default=NUM_WORKERS, help="num_workers del DataLoader")
    return p.parse_args()


if __name__ == "__main__":
    main()

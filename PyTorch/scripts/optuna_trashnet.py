"""
Optimización de Hiperparámetros con Optuna + PyTorch — TrashNet
Problema  : clasificación de residuos en 6 clases (cardboard, glass, metal, paper, plastic, trash)
Objetivo  : encontrar automáticamente la mejor combinación de hiperparámetros de entrenamiento.

Es la adaptación de scripts/optuna_ejemplo.py (Iris + Keras) al modelo CNN de
PyTorch/TrashNet_PyTorch.ipynb, con los mismos conceptos:

  Trial      : una combinación de hiperparámetros + su resultado
  Study      : colección de trials con un objetivo a maximizar
  Objective  : función que entrena el modelo y retorna la métrica
  Sampler    : TPE, algoritmo bayesiano que aprende qué zonas explorar
  Pruner     : MedianPruner, corta trials malos a mitad de entrenamiento
  suggest_*  : métodos para proponer valores (int, float, categorical)

Diferencia importante frente al ejemplo de Iris: aquí cada trial es CARO (entrenar
una CNN sobre ~1770 imágenes), así que el pruning no es un lujo sino la parte que
hace viable la búsqueda. Se reporta la métrica de validación época por época y
Optuna aborta los trials que van claramente peor que la mediana.

Uso:
    python optuna_trashnet.py                          # 25 trials, 12 épocas, métrica macro_auc
    python optuna_trashnet.py --trials 40 --epocas 15
    python optuna_trashnet.py --metrica val_accuracy
    python optuna_trashnet.py --timeout 3600           # cortar la búsqueda a 1 hora
    python optuna_trashnet.py --trials 20              # relanzar: continúa el mismo study (SQLite)

Genera en PyTorch/optuna/:
    study.db                    — base SQLite con todos los trials (permite reanudar)
    mejores_params.json         — mejor combinación encontrada (la lee entrenar_mejor.py)

Y en PyTorch/resultados/graficos/:
    optuna_historia.png         — métrica por trial + distribución de resultados
    optuna_importancia.png      — importancia relativa de cada hiperparámetro
    optuna_por_parametro.png    — métrica en función de cada hiperparámetro
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import optuna

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trashnet_comun import (
    DEVICE, OPTUNA_DIR, GRAFICOS_DIR, TRAIN_DIR,
    crear_carpetas_resultados,
    construir_loaders, construir_modelo, construir_optimizador,
    correr_epoca, obtener_predicciones,
    calcular_cm, macro_f1_desde_cm, macro_auc,
    pesos_de_clase, semilla,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42


# ── ESPACIO DE BÚSQUEDA ───────────────────────────────────────────────
# Documentación del espacio que explora Optuna. La definición efectiva está en
# `sugerir_hiperparametros`; esta tabla es para leerla de un vistazo / el informe.
ESPACIO = {
    # Arquitectura => lo más importante
    "n_bloques"    : "int   => [2, 4]                bloques conv (Conv-BN-ReLU-Pool)",
    "filtros_base" : "cat   => 16, 32, 48            filtros del primer bloque (se duplican por bloque)",
    "units_fc"     : "int   => [64, 512] step=64     neuronas de la capa densa",
    "activacion"   : "cat   => relu, elu, leaky_relu",

    # Regularización
    "dropout"      : "float => [0.0, 0.6] step=0.05  dropout antes de la capa final",
    "usar_bn"      : "cat   => True, False           BatchNorm en cada bloque conv",
    "weight_decay" : "float => [1e-6, 1e-2] log      penalización L2 sobre los pesos",
    "aug"          : "cat   => leve, media, fuerte   intensidad del aumento de datos",

    # Optimización
    "optimizer"    : "cat   => adam, adamw, sgd, rmsprop",
    "lr"           : "float => [1e-5, 1e-2] log      learning rate",
    "batch_size"   : "cat   => 16, 32, 64",
    "img_size"     : "cat   => 128, 150, 192         resolución de entrada",

    # Desbalance de clases
    "class_weight" : "cat   => True, False           ponderar la loss por frecuencia de clase",
}


def sugerir_hiperparametros(trial: optuna.Trial) -> dict:
    """Propone una combinación de hiperparámetros para este trial."""
    return {
        # Arquitectura
        "n_bloques"    : trial.suggest_int("n_bloques", 2, 4),
        "filtros_base" : trial.suggest_categorical("filtros_base", [16, 32, 48]),
        "units_fc"     : trial.suggest_int("units_fc", 64, 512, step=64),
        "activacion"   : trial.suggest_categorical("activacion", ["relu", "elu", "leaky_relu"]),

        # Regularización
        "dropout"      : trial.suggest_float("dropout", 0.0, 0.6, step=0.05),
        "usar_bn"      : trial.suggest_categorical("usar_bn", [True, False]),
        "weight_decay" : trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "aug"          : trial.suggest_categorical("aug", ["leve", "media", "fuerte"]),

        # Optimización
        "optimizer"    : trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd", "rmsprop"]),
        "lr"           : trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size"   : trial.suggest_categorical("batch_size", [16, 32, 64]),
        "img_size"     : trial.suggest_categorical("img_size", [128, 150, 192]),

        # Desbalance
        "class_weight" : trial.suggest_categorical("class_weight", [True, False]),
    }


# ── FUNCIÓN OBJETIVO ──────────────────────────────────────────────────
def crear_objective(epocas: int, metrica: str, workers: int):
    """
    Devuelve la función objective con la configuración fijada.

    `metrica` es lo que Optuna maximiza, evaluado siempre sobre VALIDACIÓN
    (nunca sobre test: el test se toca una sola vez, en entrenar_mejor.py):
        val_accuracy — la más simple de interpretar
        macro_f1     — trata todas las clases por igual (bueno con desbalance)
        macro_auc    — área bajo la ROC promediada one-vs-rest (por defecto);
                       es la métrica coherente con las gráficas ROC del informe
    """

    def objective(trial: optuna.Trial) -> float:
        semilla(SEED)                       # mismo punto de partida para todos los trials
        params = sugerir_hiperparametros(trial)
        model = None

        try:
            train_loader, valid_loader, _, class_names = construir_loaders(
                img_size   = params["img_size"],
                batch_size = params["batch_size"],
                aug        = params["aug"],
                num_workers= workers,
                seed       = SEED,
            )

            model = construir_modelo(len(class_names), params)

            pesos = pesos_de_clase(class_names).to(DEVICE) if params["class_weight"] else None
            criterion = nn.CrossEntropyLoss(weight=pesos)
            optimizer = construir_optimizador(model, params)

            mejor = 0.0
            for epoca in range(1, epocas + 1):
                correr_epoca(model, train_loader, criterion, optimizer)

                # Evaluar en validación con la métrica elegida
                y_true, y_pred, y_prob = obtener_predicciones(model, valid_loader)
                if metrica == "val_accuracy":
                    valor = float((y_true == y_pred).mean())
                elif metrica == "macro_f1":
                    valor = macro_f1_desde_cm(calcular_cm(y_true, y_pred, len(class_names)))
                elif metrica == "macro_auc":
                    valor = macro_auc(y_true, y_prob, len(class_names))
                else:
                    raise ValueError(f"métrica desconocida: {metrica}")

                mejor = max(mejor, valor)

                # Pruning: le contamos a Optuna cómo va el trial en esta época.
                # Si está por debajo de la mediana de los trials previos, se aborta.
                trial.report(valor, epoca)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Guardamos info extra para el informe (visible en el study)
            trial.set_user_attr("epocas", epocas)
            trial.set_user_attr("n_params", sum(p.numel() for p in model.parameters()))
            return mejor

        except torch.OutOfMemoryError:
            # La combinación no cabe en esta GPU (p.ej. img_size=192 + batch=64 +
            # filtros_base=48 en una tarjeta de 4 GB). No es un error del código:
            # simplemente no es viable en este hardware, así que la descartamos
            # como PRUNED en vez de dejar que la excepción mate todo el study.
            trial.set_user_attr("descartado", "CUDA out of memory")
            # Reportar un valor bajo en el paso 0 (que el bucle nunca usa) para que
            # el sampler TPE aprenda a evitar esta zona del espacio en vez de
            # insistir con combinaciones que no caben.
            trial.report(0.0, 0)
            print(f"{trial.number:>6}  {'—':>8}  {'—':>8}  {'OOM':>8}  "
                  f"descartado: no cabe en la GPU "
                  f"(img={params['img_size']} bs={params['batch_size']} "
                  f"filt={params['filtros_base']} bloq={params['n_bloques']})", flush=True)
            raise optuna.TrialPruned()

        finally:
            # Liberar la VRAM sí o sí, incluso si el trial falló a medias:
            # si no, el siguiente trial arranca con la memoria fragmentada.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


# ── ESTUDIO ───────────────────────────────────────────────────────────
def ejecutar_estudio(args) -> optuna.Study:
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    storage = f"sqlite:///{os.path.join(OPTUNA_DIR, 'study.db')}"

    # Sampler TPE: bayesiano, aprende qué zonas del espacio son prometedoras.
    # Alternativas: RandomSampler, GridSampler, CmaEsSampler.
    sampler = optuna.samplers.TPESampler(seed=SEED)

    # MedianPruner: corta un trial si su métrica está por debajo de la mediana
    # de los trials anteriores en la misma época. n_startup_trials=5 => las
    # primeras 5 combinaciones se entrenan completas para tener referencia.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    # load_if_exists => relanzar el script continúa la búsqueda en vez de empezar de cero
    study = optuna.create_study(
        study_name = args.study_name,
        storage    = storage,
        direction  = "maximize",
        sampler    = sampler,
        pruner     = pruner,
        load_if_exists = True,
    )

    print("─" * 78)
    print("  Optimización de Hiperparámetros con Optuna — TrashNet (PyTorch)")
    print(f"  Device: {DEVICE}   ·   Métrica objetivo: {args.metrica} (validación)")
    print(f"  Trials nuevos: {args.trials}   ·   Épocas máx por trial: {args.epocas}")
    if study.trials:
        print(f"  Continuando study '{args.study_name}' con {len(study.trials)} trials previos")
    print("─" * 78)
    print(f"\n{'Trial':>6}  {'Valor':>8}  {'Mejor':>8}  {'Estado':>8}  Parámetros")
    print("─" * 78)

    t0 = time.time()

    def callback_progreso(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        estado = trial.state.name[:8]
        val    = trial.value if trial.value is not None else float("nan")
        try:
            mejor = study.best_value
        except ValueError:
            mejor = float("nan")
        params = trial.params
        resumen = (
            f"bloq={params.get('n_bloques','?')} "
            f"filt={params.get('filtros_base','?')} "
            f"fc={params.get('units_fc','?')} "
            f"drop={params.get('dropout',0):.2f} "
            f"lr={params.get('lr',0):.1e} "
            f"opt={params.get('optimizer','?')} "
            f"bs={params.get('batch_size','?')} "
            f"img={params.get('img_size','?')} "
            f"aug={params.get('aug','?')}"
        )
        print(f"{trial.number:>6}  {val:>8.4f}  {mejor:>8.4f}  {estado:>8}  {resumen}", flush=True)

    study.optimize(
        crear_objective(args.epocas, args.metrica, args.workers),
        n_trials  = args.trials,
        timeout   = args.timeout,
        callbacks = [callback_progreso],
        gc_after_trial = True,
        show_progress_bar = False,
        # Red de seguridad: sin esto, CUALQUIER excepción en un trial aborta la
        # búsqueda completa y se pierden las horas de cómputo ya invertidas.
        # Con catch, el trial queda como FAIL y la búsqueda sigue con el siguiente.
        catch = (RuntimeError,),
    )

    print(f"\n  Tiempo total de búsqueda: {(time.time() - t0) / 60:.1f} min")
    return study


# ── VISUALIZACIONES ───────────────────────────────────────────────────
def _trials_completos(study):
    """
    Solo los trials que entrenaron las épocas completas.

    Ojo: no basta con filtrar `t.value is not None`. Optuna guarda en `value` de un
    trial PODADO su último valor intermedio, que es un puntaje a medio entrenar y no
    es comparable con uno que completó todas las épocas. Mezclarlos ensucia las
    estadísticas y la importancia. Aparte, los descartados por falta de VRAM llevan
    un 0.0 artificial (reportado para guiar al sampler) que no es un resultado real.
    """
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]


def _trials_podados(study):
    """Podados por el pruner (excluye los descartados por falta de memoria)."""
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
            and t.value is not None
            and not t.user_attrs.get("descartado")]


def plot_historia(study, metrica, output):
    """Métrica por trial (con mejor acumulado) + histograma de resultados."""
    trials  = _trials_completos(study)
    nums    = [t.number for t in trials]
    vals    = [t.value  for t in trials]
    mejores = [max(vals[:i + 1]) for i in range(len(vals))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colores = ["#2E75B6" if v < max(vals) else "#1E7B45" for v in vals]
    ax1.bar(nums, vals, color=colores, alpha=0.75, width=0.7, label="Trial completo")

    # Los podados se muestran aparte: su valor es parcial (menos épocas), así que
    # no es comparable, pero sirve ver dónde y a qué altura los cortó el pruner.
    podados = _trials_podados(study)
    if podados:
        ax1.scatter([t.number for t in podados], [t.value for t in podados],
                    marker="x", color="#999999", s=45, lw=1.5, zorder=4,
                    label="Podado (valor parcial)")

    ax1.plot(nums, mejores, "r-o", ms=4, lw=2, label="Mejor acumulado")
    ax1.axhline(max(vals), color="green", ls="--", lw=1.2, alpha=0.7)
    ax1.set_xlabel("Número de trial", fontsize=12)
    ax1.set_ylabel(metrica, fontsize=12)
    ax1.set_title(f"{metrica} por trial", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9); ax1.grid(axis="y", alpha=0.3)
    # Zoom al rango real de resultados: si se dejara desde 0, todas las barras se
    # verían casi iguales y no se distinguiría nada.
    piso = min(vals + [t.value for t in podados]) if podados else min(vals)
    ax1.set_ylim(piso - 0.05, 1.02)

    mejor_idx = int(np.argmax(vals))
    ax1.annotate(f"Mejor: trial {nums[mejor_idx]}  {vals[mejor_idx]:.4f}",
                 xy=(nums[mejor_idx], vals[mejor_idx]),
                 xytext=(nums[mejor_idx] - 1, min(1.0, vals[mejor_idx] + 0.045)),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 fontsize=9, color="green", fontweight="bold")

    ax2.hist(vals, bins=min(12, max(4, len(vals) // 2)),
             color="#2E75B6", alpha=0.75, edgecolor="white")
    ax2.axvline(max(vals),     color="green",  ls="--", lw=2, label=f"Mejor: {max(vals):.4f}")
    ax2.axvline(np.mean(vals), color="orange", ls="--", lw=2, label=f"Media: {np.mean(vals):.4f}")
    ax2.set_xlabel(metrica, fontsize=12)
    ax2.set_ylabel("Número de trials", fontsize=12)
    ax2.set_title("Distribución de resultados", fontsize=13, fontweight="bold")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    n_oom = sum(1 for t in study.trials if t.user_attrs.get("descartado"))
    resumen = f"{len(trials)} trials completos · {len(podados)} podados"
    if n_oom:
        resumen += f" · {n_oom} descartado(s) por VRAM"
    plt.suptitle(f"Optuna — {resumen}  ·  Sampler: TPE  ·  Dataset: TrashNet\n"
                 f"(las estadísticas usan solo los trials completos)",
                 fontsize=12, fontweight="bold", color="#1F3864")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {output}")


def _codificar_param(trials, param):
    """Convierte los valores de un hiperparámetro a float (categorías -> índice)."""
    cats = sorted({str(t.params[param]) for t in trials if param in t.params})
    vals = []
    for t in trials:
        v = t.params.get(param)
        if v is None:
            vals.append(np.nan)
        elif isinstance(v, bool):
            vals.append(float(v))
        elif isinstance(v, str):
            vals.append(float(cats.index(str(v))))
        else:
            vals.append(float(v))
    return np.array(vals, dtype=float), cats


def plot_importancia(study, metrica, output):
    """
    Importancia de cada hiperparámetro como |correlación de Spearman| con la métrica.
    (Optuna trae get_param_importances con fANOVA, pero requiere scikit-learn, que
    no está instalado en este entorno; esta aproximación no añade dependencias.)
    """
    trials = _trials_completos(study)
    if len(trials) < 5:
        print("  Pocos trials para calcular importancia")
        return

    vals   = np.array([t.value for t in trials])
    r_vals = np.argsort(np.argsort(vals)).astype(float)

    importancias = {}
    for param in trials[0].params:
        pv, _ = _codificar_param(trials, param)
        if np.all(pv == pv[0]) or np.isnan(pv).any():
            importancias[param] = 0.0
            continue
        r_pv = np.argsort(np.argsort(pv)).astype(float)
        corr = np.corrcoef(r_pv, r_vals)[0, 1]
        importancias[param] = 0.0 if np.isnan(corr) else abs(corr)

    importancias = dict(sorted(importancias.items(), key=lambda x: x[1]))
    nombres, valores = list(importancias.keys()), list(importancias.values())

    fig, ax = plt.subplots(figsize=(10, 7))
    colores = plt.colormaps["Blues"](np.linspace(0.35, 0.85, len(nombres)))
    bars = ax.barh(nombres, valores, color=colores, edgecolor="white", height=0.65)
    for bar, val in zip(bars, valores):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    ax.set_xlabel("Importancia (|correlación de Spearman|)", fontsize=12)
    ax.set_title("Importancia de Hiperparámetros\n"
                 f"Mayor valor => más impacto en {metrica}\n"
                 f"calculada sobre {len(trials)} trials completos"
                 f"{' — pocos datos, tómalo como orientativo' if len(trials) < 20 else ''}",
                 fontsize=12, fontweight="bold", color="#1F3864")
    ax.set_xlim(0, max(max(valores) * 1.2, 0.1))
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {output}")


def plot_por_parametro(study, metrica, output):
    """
    Un panel por hiperparámetro: métrica vs valor propuesto. Sirve para ver de un
    golpe qué rangos funcionan (equivalente a los 'slice plots' de Optuna).
    """
    trials = _trials_completos(study)
    if len(trials) < 5:
        return
    params = list(trials[0].params.keys())
    vals   = np.array([t.value for t in trials])

    cols = 4
    rows = int(np.ceil(len(params) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    axes_flat = list(np.array(axes).flat)

    mejor_trial = max(trials, key=lambda t: t.value)

    for ax, param in zip(axes_flat, params):
        pv, cats = _codificar_param(trials, param)
        log = param in ("lr", "weight_decay")

        sc = ax.scatter(pv, vals, c=[t.number for t in trials],
                        cmap="viridis", s=38, alpha=0.85, edgecolors="white", linewidths=0.5)
        # Resaltar el mejor trial
        pv_mejor = pv[[t.number for t in trials].index(mejor_trial.number)]
        ax.scatter([pv_mejor], [mejor_trial.value], marker="*", s=260,
                   color="#1E7B45", edgecolors="black", linewidths=0.6, zorder=5)

        if log:
            ax.set_xscale("log")
        if cats and isinstance(trials[0].params.get(param), (str, bool)):
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, fontsize=8, rotation=20)
        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel(metrica, fontsize=9)
        ax.grid(alpha=0.3)

    for j in range(len(params), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.colorbar(sc, ax=axes_flat[-1] if len(params) < len(axes_flat) else axes_flat,
                 label="Nº de trial", fraction=0.03)
    plt.suptitle(f"{metrica} en función de cada hiperparámetro  (★ = mejor trial)",
                 fontsize=14, fontweight="bold", color="#1F3864")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {output}")


# ── MAIN ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Búsqueda de hiperparámetros con Optuna para TrashNet (PyTorch)")
    p.add_argument("--trials",  type=int, default=25, help="número de combinaciones a probar (default 25)")
    p.add_argument("--epocas",  type=int, default=12, help="épocas por trial (default 12)")
    p.add_argument("--metrica", default="macro_auc",
                   choices=["macro_auc", "val_accuracy", "macro_f1"],
                   help="métrica de validación a maximizar (default macro_auc)")
    p.add_argument("--timeout", type=int, default=None, help="segundos máximos de búsqueda")
    p.add_argument("--workers", type=int, default=2, help="num_workers de los DataLoader")
    p.add_argument("--study-name", default="trashnet_pytorch", help="nombre del study en SQLite")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OPTUNA_DIR, exist_ok=True)

    study = ejecutar_estudio(args)

    trials_ok = _trials_completos(study)
    if not trials_ok:
        print("\nNingún trial completó. Prueba con menos épocas o revisa el dataset.")
        return

    # ── Resumen ───────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("  MEJORES HIPERPARÁMETROS ENCONTRADOS")
    print("═" * 78)
    print(f"  {args.metrica} (validación): {study.best_value:.4f}")
    print("\n  Parámetros:")
    for k, v in study.best_params.items():
        print(f"    {k:<14}: {v}")

    n_prun = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_fail = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    n_oom  = sum(1 for t in study.trials if t.user_attrs.get("descartado"))
    valores = [t.value for t in trials_ok]
    print(f"\n  Trials completados : {len(trials_ok)}")
    print(f"  Trials podados     : {n_prun}" + (f" (de los cuales {n_oom} por falta de VRAM)" if n_oom else ""))
    if n_fail:
        print(f"  Trials con error   : {n_fail}")
    print(f"  Media  {args.metrica:<12}: {np.mean(valores):.4f}")
    print(f"  Std    {args.metrica:<12}: {np.std(valores):.4f}")
    print(f"  Peor resultado     : {min(valores):.4f}")
    print(f"  Mejor resultado    : {max(valores):.4f}")

    # ── Guardar los mejores parámetros para entrenar_mejor.py ─────────
    ruta_params = os.path.join(OPTUNA_DIR, "mejores_params.json")
    with open(ruta_params, "w", encoding="utf-8") as f:
        json.dump({
            "metrica"      : args.metrica,
            "mejor_valor"  : study.best_value,
            "mejor_trial"  : study.best_trial.number,
            "epocas_trial" : args.epocas,
            "n_trials"     : len(study.trials),
            "params"       : study.best_params,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ {ruta_params}")

    # ── Visualizaciones ───────────────────────────────────────────────
    print("\nGenerando visualizaciones...")
    plot_historia(study, args.metrica, os.path.join(GRAFICOS_DIR, "optuna_historia.png"))
    plot_importancia(study, args.metrica, os.path.join(GRAFICOS_DIR, "optuna_importancia.png"))
    plot_por_parametro(study, args.metrica, os.path.join(GRAFICOS_DIR, "optuna_por_parametro.png"))

    print("\n✓ Búsqueda completada.")
    print("  Siguiente paso:  python entrenar_mejor.py")


if __name__ == "__main__":
    main()

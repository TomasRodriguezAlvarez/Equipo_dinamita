"""
Optimización de Hiperparámetros con Optuna + TensorFlow/Keras
Problema  : Clasificación del dataset Iris (4 features, 3 clases)
Objetivo  : Encontrar la mejor combinación de hiperparámetros automáticamente.
Librerías : TensorFlow 2.16 · NumPy 1.26 · matplotlib 3.8 · optuna 3+

Conceptos que ilustra este script:
  - Trial       : una combinación de hiperparámetros + su resultado
  - Study       : colección de trials con un objetivo a optimizar
  - Objective   : función que entrena el modelo y retorna la métrica
  - Sampler     : algoritmo de búsqueda (TPE, Random, Grid, CmaEs)
  - Pruner      : corta trials prometedores para ahorrar tiempo
  - suggest_*   : métodos para proponer valores de hiperparámetros

Genera:
  optuna_historia.png   — valor de la métrica por trial
  optuna_importancia.png — importancia relativa de cada hiperparámetro

"""

import numpy as np
import tensorflow as tf
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Silenciar logs de TF (solo errores)
# Este silenciamiento es muy importante para cuando se usa el código en producción
# Recuerda que el logeo de errores es fundamental, pero el exceso de detalles hace
# que se consuma el espacio de disco demasiado rápido. Y normalmente estos sistemas
# los corremos en C dentro de una plataforma de HW ligera como PC de placa reducida 
# de calibre industrial (NO RASPEBBRY), o microcontroladores como PIC (NO ARDUINO)
# lo particular de estas plataformas es que el espacio es poco y si el registro es 
# distribuido puedes saturar una red fácilmente.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Silenciar logs de Optuna durante los trials (veremos solo el resumen)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 1. CONFIGURACIÓN ──────────────────────────────────────────────────
N_TRIALS  = 30     # número de combinaciones a probar
EPOCHS    = 50     # épocas por trial (bajo para que sea rápido)
SEED      = 42

# ── 2. DATASET: IRIS ──────────────────────────────────────────────────
# 150 muestras · 4 features · 3 clases
# Este ejemplo es uno de los más utilizados en el aprendizaje de ML y DL
# La gracia de esto es que el código ya es estándar y fácil de entender
# Recuerden que lo importante aquí es entender lo que pasa para luego
# aplicarlo al proyecto de Visión Computacional
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cargar_iris():
    iris       = load_iris()
    X, y       = iris.data.astype(np.float32), iris.target
    scaler     = StandardScaler()
    X          = scaler.fit_transform(X).astype(np.float32)
    y_onehot   = tf.keras.utils.to_categorical(y, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ── 3. ESPACIO DE BÚSQUEDA ────────────────────────────────────────────
# Estos son los hiperparámetros que Optuna va a explorar.
# Cada trial recibe valores diferentes dentro de estos rangos.

ESPACIO = {
    # Arquitectura => lo más importante!
    "n_capas"    : "int   => [1, 4]           número de capas ocultas",
    "units"      : "int   => [16, 256] step=16 neuronas por capa",
    "activacion" : "cat   => relu, tanh, elu   funciones de activación",

    # Regularización
    "dropout"    : "float => [0.0, 0.5]        tasa de dropout",     # Acuérdate de la finalidad de un dropout
    "use_bn"     : "cat   => True, False        usar BatchNormalization", 

    # Optimización
    "optimizer"  : "cat   => adam, sgd, rmsprop",
    "lr"         : "float => [1e-4, 1e-1] log  learning rate",
    "batch_size" : "cat   => 8, 16, 32, 64",
}

# ── 4. FUNCIÓN OBJETIVO ───────────────────────────────────────────────
# Esta función es el corazón de Optuna.
# Recibe un trial, propone hiperparámetros, entrena y retorna la métrica.
# Optuna la llama N_TRIALS veces buscando maximizar val_accuracy.

def objective(trial):
    # Limpiar el estado de Keras entre trials
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    # ── Proponer hiperparámetros ──────────────────────────────────────
    # suggest_int    => entero en [low, high]
    # suggest_float  => float en [low, high], log=True para escala log
    # suggest_categorical => elige de una lista

    n_capas    = trial.suggest_int("n_capas",    1, 4)
    units      = trial.suggest_int("units",      16, 256, step=16)
    activacion = trial.suggest_categorical("activacion", ["relu", "tanh", "elu"])
    dropout    = trial.suggest_float("dropout",  0.0, 0.5, step=0.05)
    use_bn     = trial.suggest_categorical("use_bn", [True, False])
    optimizer  = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    lr         = trial.suggest_float("lr",       1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

    # ── Construir modelo con los hiperparámetros del trial ────────────
    model = tf.keras.Sequential(name=f"trial_{trial.number}")
    model.add(tf.keras.layers.Input(shape=(4,)))

    for _ in range(n_capas):
        model.add(tf.keras.layers.Dense(units, activation=activacion))
        if use_bn:
            model.add(tf.keras.layers.BatchNormalization())
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    # ── Compilar con el optimizador propuesto ─────────────────────────
    # El uso de optimizadores es un tema muy común en su uso, pero la realidad es que pocos
    # entienden el efecto de cada uno.
    # SGD lo que hace es actualizar los pesos de la red moviéndose en sentido contrario de la gradiente
    # lo que provoca que el avance de aprendizaje sea lento pero tiende a quedar atrapado en mínimos locales
    # RMSprop este en cambio adapta el LR para cada parámetro de manera individual, esto produce que sea muy útil
    # cuando se tienen gradientes inestables y redes neuronales recurrentes (recursivas)
    # ADAM es el mas utilizado básicamente porque converge muy rápido a un resultado
    opts = {
        "adam"    : tf.keras.optimizers.Adam(lr),
        "sgd"     : tf.keras.optimizers.SGD(lr, momentum=0.9),
        "rmsprop" : tf.keras.optimizers.RMSprop(lr),
    }
    model.compile(
        optimizer = opts[optimizer],
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    # ── Entrenar ──────────────────────────────────────────────────────
    # EarlyStopping para no desperdiciar tiempo en trials malos
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8,
        restore_best_weights=True, verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = EPOCHS,
        batch_size      = batch_size,
        callbacks       = [early_stop],
        verbose         = 0,
    )

    # ── Retornar la métrica a optimizar ───────────────────────────────
    # Optuna maximiza este valor => usamos el mejor val_accuracy del historial
    val_acc = max(history.history["val_accuracy"])
    return val_acc

# ── 5. ESTUDIO OPTUNA ─────────────────────────────────────────────────
def ejecutar_estudio():
    print("─" * 60)
    print("  Optimización de Hiperparámetros con Optuna")
    print(f"  Trials: {N_TRIALS}  ·  Épocas máx por trial: {EPOCHS}")
    print("─" * 60)
    print(f"\n{'Trial':>6}  {'Val Acc':>8}  {'Mejor':>8}  Parámetros")
    print("─" * 60)

    # Sampler: TPE (Tree-structured Parzen Estimator)
    # Algoritmo bayesiano que aprende qué zonas del espacio son prometedoras
    # Alternativas: RandomSampler, GridSampler, CmaEsSampler
    sampler = optuna.samplers.TPESampler(seed=SEED)

    # Pruner: MedianPruner
    # Corta un trial si su métrica a mitad de camino está por debajo
    # de la mediana de los trials anteriores => ahorra tiempo
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction = "maximize",    # queremos maximizar val_accuracy
        sampler   = sampler,
        pruner    = pruner,
        study_name= "iris_mlp",
    )

    # Callback para imprimir progreso en tiempo real
    mejor_hasta_ahora = [0.0]

    def callback_progreso(study, trial):
        val   = trial.value if trial.value is not None else 0.0
        mejor = study.best_value
        marca = "★" if val >= mejor_hasta_ahora[0] else " "
        mejor_hasta_ahora[0] = max(mejor_hasta_ahora[0], val)
        params_str = (
            f"capas={trial.params.get('n_capas','?')} "
            f"units={trial.params.get('units','?')} "
            f"act={trial.params.get('activacion','?')} "
            f"lr={trial.params.get('lr', 0):.1e} "
            f"opt={trial.params.get('optimizer','?')}"
        )
        print(f"{trial.number:>6}  {val:>8.4f}  {mejor:>8.4f}  {marca} {params_str}")

    study.optimize(
        objective,
        n_trials  = N_TRIALS,
        callbacks = [callback_progreso],
        show_progress_bar=False,
    )

    return study

# ── 6. REENTRENAR CON MEJORES PARÁMETROS ─────────────────────────────
def reentrenar_mejor(best_params, train, val, test):
    print("\n" + "─" * 60)
    print("  Reentrenando con los mejores hiperparámetros")
    print("─" * 60)

    X_train, y_train = train
    X_val,   y_val   = val
    X_test,  y_test  = test

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(4,)))
    for _ in range(best_params["n_capas"]):
        model.add(tf.keras.layers.Dense(
            best_params["units"],
            activation=best_params["activacion"]
        ))
        if best_params["use_bn"]:
            model.add(tf.keras.layers.BatchNormalization())
        if best_params["dropout"] > 0:
            model.add(tf.keras.layers.Dropout(best_params["dropout"]))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    lr  = best_params["lr"]
    opt = best_params["optimizer"]
    opts = {
        "adam"    : tf.keras.optimizers.Adam(lr),
        "sgd"     : tf.keras.optimizers.SGD(lr, momentum=0.9),
        "rmsprop" : tf.keras.optimizers.RMSprop(lr),
    }
    model.compile(
        optimizer = opts[opt],
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = 100,
        batch_size      = best_params["batch_size"],
        callbacks       = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=15,
                restore_best_weights=True, verbose=0
            )
        ],
        verbose=0,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Loss    : {loss:.4f}")
    print(f"  Test Accuracy: {acc:.4f}  ({acc:.1%})")
    return model, history

# ── 7. VISUALIZACIONES ────────────────────────────────────────────────
def plot_historia(study, output="optuna_historia.png"):
    trials  = [t for t in study.trials if t.value is not None]
    nums    = [t.number for t in trials]
    vals    = [t.value  for t in trials]
    mejores = [max(vals[:i+1]) for i in range(len(vals))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Val accuracy por trial ─────────────────────────────────────────
    colores = ["#2E75B6" if v < max(vals) else "#1E7B45" for v in vals]
    ax1.bar(nums, vals, color=colores, alpha=0.75, width=0.7)
    ax1.plot(nums, mejores, "r-o", ms=5, lw=2, label="Mejor acumulado")
    ax1.axhline(max(vals), color="green", ls="--", lw=1.2, alpha=0.7)
    ax1.set_xlabel("Número de trial", fontsize=12)
    ax1.set_ylabel("Val Accuracy", fontsize=12)
    ax1.set_title("Val Accuracy por trial", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Marcar el mejor
    mejor_idx = int(np.argmax(vals))
    ax1.annotate(
        f"  Mejor\n  Trial {nums[mejor_idx]}\n  {vals[mejor_idx]:.4f}",
        xy=(nums[mejor_idx], vals[mejor_idx]),
        xytext=(nums[mejor_idx] + 1.5, vals[mejor_idx] - 0.12),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=9, color="green"
    )

    # ── Distribución de val accuracy ──────────────────────────────────
    ax2.hist(vals, bins=10, color="#2E75B6", alpha=0.75, edgecolor="white")
    ax2.axvline(max(vals),  color="green",  ls="--", lw=2, label=f"Mejor: {max(vals):.4f}")
    ax2.axvline(np.mean(vals), color="orange", ls="--", lw=2, label=f"Media: {np.mean(vals):.4f}")
    ax2.set_xlabel("Val Accuracy", fontsize=12)
    ax2.set_ylabel("Número de trials", fontsize=12)
    ax2.set_title("Distribución de resultados", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Optuna — {len(trials)} trials completados  ·  Sampler: TPE  ·  Dataset: Iris",
        fontsize=13, fontweight="bold", color="#1F3864"
    )
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output}")

def plot_importancia(study, output="optuna_importancia.png"):
    # Calcular importancia manual usando correlación de Spearman
    # (Optuna la calcula internamente con FAnova, aquí la aproximamos)
    trials = [t for t in study.trials if t.value is not None]
    if len(trials) < 5:
        print("  Pocos trials para calcular importancia")
        return

    params_nombres = list(trials[0].params.keys())
    importancias   = {}

    vals = np.array([t.value for t in trials])

    for param in params_nombres:
        param_vals = []
        for t in trials:
            v = t.params.get(param)
            # Convertir categorías a números
            if isinstance(v, bool):
                v = float(v)
            elif isinstance(v, str):
                cats = sorted(set(
                    str(tt.params.get(param)) for tt in trials
                    if param in tt.params
                ))
                v = float(cats.index(str(v)))
            param_vals.append(float(v) if v is not None else 0.0)

        # Correlación de Spearman aproximada
        pv    = np.array(param_vals)
        r_pv  = np.argsort(np.argsort(pv)).astype(float)
        r_vals = np.argsort(np.argsort(vals)).astype(float)
        corr   = np.corrcoef(r_pv, r_vals)[0, 1]
        importancias[param] = abs(corr)

    # Ordenar
    importancias = dict(sorted(importancias.items(), key=lambda x: x[1]))
    nombres = list(importancias.keys())
    valores = list(importancias.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colores = plt.colormaps["Blues"](
        np.linspace(0.35, 0.85, len(nombres))
    )
    bars = ax.barh(nombres, valores, color=colores, edgecolor="white", height=0.6)

    for bar, val in zip(bars, valores):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=11)

    ax.set_xlabel("Importancia (|correlación de Spearman|)", fontsize=12)
    ax.set_title(
        "Importancia de Hiperparámetros\n"
        "Mayor valor => más impacto en val_accuracy",
        fontsize=13, fontweight="bold", color="#1F3864"
    )
    ax.set_xlim(0, max(valores) * 1.2)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output}")

# ── 8. MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Datos
    train, val, test = cargar_iris()
    X_train, y_train = train
    X_val,   y_val   = val
    X_test,  y_test  = test
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}\n")

    # Búsqueda de hiperparámetros
    study = ejecutar_estudio()

    # Resumen
    print("\n" + "═" * 60)
    print("  MEJORES HIPERPARÁMETROS ENCONTRADOS")
    print("═" * 60)
    print(f"  Val Accuracy: {study.best_value:.4f}  ({study.best_value:.1%})")
    print("\n  Parámetros:")
    for k, v in study.best_params.items():
        print(f"    {k:<14}: {v}")

    # Estadísticas del estudio
    trials_ok  = [t for t in study.trials if t.value is not None]
    print(f"\n  Trials completados : {len(trials_ok)}/{N_TRIALS}")
    print(f"  Media val_accuracy : {np.mean([t.value for t in trials_ok]):.4f}")
    print(f"  Std  val_accuracy  : {np.std([t.value for t in trials_ok]):.4f}")
    print(f"  Peor resultado     : {min(t.value for t in trials_ok):.4f}")
    print(f"  Mejor resultado    : {max(t.value for t in trials_ok):.4f}")

    # Reentrenar con los mejores parámetros
    modelo_final, hist_final = reentrenar_mejor(
        study.best_params, train, val, test
    )

    # Visualizaciones
    print("\nGenerando visualizaciones...")
    plot_historia(study)
    plot_importancia(study)

    print("\n✓ Completado. Archivos generados:")
    print("  optuna_historia.png")
    print("  optuna_importancia.png")
    print("\n─" * 60)
    print("CONCEPTOS CLAVE DE OPTUNA")
    print("─" * 60)
    conceptos = [
        ("Trial",     "Una combinación de hiperparámetros + su resultado"),
        ("Study",     "Colección de trials con un objetivo a maximizar/minimizar"),
        ("Objective", "Función que entrena el modelo y retorna la métrica"),
        ("TPE",       "Sampler bayesiano: aprende qué zonas explorar primero"),
        ("Pruner",    "Corta trials malos a mitad para ahorrar cómputo"),
        ("suggest_*", "Métodos para proponer valores: int, float, categorical"),
    ]
    for nombre, desc in conceptos:
        print(f"  {nombre:<12}: {desc}")

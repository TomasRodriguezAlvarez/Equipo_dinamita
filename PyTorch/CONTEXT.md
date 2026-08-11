# CONTEXT — rama PyTorch del proyecto TrashNet

Documento de traspaso: en qué estado quedó la parte de PyTorch, qué decisiones se
tomaron y por qué, y qué sigue. Última actualización: **2026-08-10** (cuarto modelo).

---

## 1. Qué es esto

Clasificación de residuos con PyTorch sobre el dataset TrashNet. Es la contraparte de la
implementación en TensorFlow del equipo (`tensorflow/`), sobre el mismo dataset y los
mismos splits.

Responsable de esta parte: Benjamín. El resto del equipo trabaja la versión de
TensorFlow y el informe conjunto en `resultados/` (raíz).

Estado actual: **cuatro modelos terminados y evaluados en test.**

1. **CNN desde cero**, 6 clases de material — scripts + Optuna (§5).
2. **MobileNetV3Small preentrenada**, 6 clases — transfer learning en 2 fases (§6).
3. **Jerárquico por contenedor**, 4 clases — MobileNetV3Small con la búsqueda de
   hiperparámetros ya corregida (§7). **Es el mejor multiclase: 88.80% de accuracy.**
4. **Binario reciclable / basura**, 2 clases — con calibración de umbral y evaluación
   fuera de distribución contra un dataset externo (§8).

Lo más aprovechable para el informe no es el accuracy de ninguno de ellos por separado,
sino las dos lecciones transversales, que salieron de comparar los cuatro:

- **cuándo Optuna aporta y cuándo no** (§9);
- **que ninguno de los cuatro generaliza fuera de TrashNet** — el modelo 4 lo mide: F1
  0.818 en su propio test, recall 0.119 con fotos reales (§8).

---

## 2. Entorno

```bash
conda activate frameworks-ia
```

Es el **único** env con PyTorch (`base` y `~/ek-venv` no lo tienen). En Jupyter hay que
elegir ese kernel: la primera celda de los notebooks imprime `sys.executable` justo para
comprobarlo, y tiene que decir `.../envs/frameworks-ia/bin/python`.

| Paquete | Versión | Nota |
|---|---|---|
| torch | 2.12.1+cu126 | CUDA disponible |
| torchvision | 0.27.1+cu126 | |
| optuna | 4.9.0 | instalado el 2026-08-05 para este trabajo |
| numpy | 2.4.4 | `np.trapz` fue renombrado a `np.trapezoid` |
| matplotlib | 3.11.0 | |
| tensorflow | 2.21.0 | conviven en el mismo env |
| **scikit-learn** | **NO instalado** | ver §10 |

GPU: **NVIDIA GTX 1650 SUPER, 4 GB (3.63 GiB usables)**. Es la restricción más
importante de todo este trabajo — ver §10.

---

## 3. Estructura

```
PyTorch/
├── CONTEXT.md                                    este archivo
├── Lab1_Tensores_Autograd.ipynb                  ejercicio de clase
│
├── TrashNet_PyTorch.ipynb                        MODELO 1 (CNN desde cero, 15 épocas)
├── scripts/                                      MODELO 1: búsqueda y reentrenamiento
│   ├── README.md                                 uso detallado de los scripts
│   ├── trashnet_comun.py                         módulo compartido (datos, modelo, métricas)
│   ├── optuna_trashnet.py                        búsqueda de hiperparámetros
│   └── entrenar_mejor.py                         reentrenamiento final + evaluación
├── optuna/                                       MODELO 1
│   ├── study.db                                  todos los trials (SQLite; permite reanudar)
│   └── mejores_params.json                       mejor combinación encontrada
├── resultados/                                   MODELO 1: salidas de los scripts
│   ├── graficos/  matriz_de_confusion/  reporte/
│
├── TrashNet_PyTorch_MobileNetV3_DEFINITIVO.ipynb MODELO 2 (6 clases, transfer learning)
├── modelos/                                      pesos .pt de los modelos 1 y 2
├── predicciones/                                 figuras de los NOTEBOOKS 1 y 2
│
├── jerarquico/                                   MODELO 3, autocontenido
│   ├── TrashNet_PyTorch_Jerarquico_DEFINITIVO.ipynb
│   ├── modelos/                                  trashnet_jerarquico_{sin_optuna,optuna}.pt
│   └── predicciones/                             jer_*.png
│
└── binario/                                      MODELO 4, autocontenido
    ├── TrashNet_PyTorch_Binario_DEFINITIVO.ipynb
    ├── modelos/                                  trashnet_binario_{sin_optuna,optuna}.pt
    └── predicciones/                             bin_*.png
```

Y en la raíz del repo, fuera de `dataset/` a propósito:

```
dataset_gd/
├── README.md      procedencia y licencia (CC BY 4.0)
└── trash/         453 imágenes del Garbage Dataset, SOLO para evaluación (§8)
```

Los scripts **no escriben** en la carpeta `resultados/` de la raíz del repo: esa es
del informe conjunto del equipo. Si hay que aportar gráficos al informe, se copian a
mano.

Los modelos 3 y 4 viven en su propia carpeta para no mezclar sus pesos y figuras con
los de 6 clases. Por eso sus notebooks usan `../../dataset` en vez de `../dataset`.

`dataset_gd/` está fuera de `dataset/` para que ningún `ImageFolder` del proyecto lo
recoja por accidente: **no es TrashNet y nunca se entrena con él.**

Dataset en `dataset/` (raíz) con splits ya hechos: 1766 train / 377 val / 384 test.

---

## 4. Los cuatro modelos de un vistazo

Todo medido en el mismo conjunto de test (384 imágenes), tocado una sola vez por modelo.

| Modelo | Clases | Accuracy | F1 macro | AUC macro |
|---|---|---|---|---|
| **3. Jerárquico con Optuna** | 4 | **0.8880** | **0.8461** | 0.9768 |
| 3. Jerárquico sin Optuna | 4 | 0.8724 | 0.8281 | 0.9764 |
| 2. MobileNetV3 con Optuna | 6 | 0.8438 | 0.8336 | **0.9802** |
| 2. MobileNetV3 sin Optuna | 6 | 0.8203 | 0.8107 | 0.9761 |
| 1. CNN scratch baseline | 6 | 0.7708 | 0.7570 | 0.9517 |
| 1. CNN scratch con Optuna | 6 | 0.7448 | 0.7355 | 0.9456 |

El modelo 4 va aparte porque su tarea no es comparable con las otras: solo distingue
`basura` de `reciclable`, y su clase de interés es la rara (5.7% del test).

| Modelo 4 (binario) | Accuracy | Bal. accuracy | F1 basura | AP basura |
|---|---|---|---|---|
| Trivial "todo reciclable" | 0.9427 | 0.5000 | 0.0000 | 0.057 |
| Con Optuna, `argmax` | 0.9479 | 0.9083 | 0.6552 | 0.7977 |
| **Con Optuna, umbral calibrado** | **0.9792** | 0.9036 | **0.8182** | 0.7977 |

Dos advertencias al leer estas tablas:

- **Accuracy no es comparable entre 4 y 6 clases**: no son la misma tarea. El AUC macro
  es más comparable, y ahí el mejor sigue siendo el MobileNetV3 de 6 clases (0.9802).
- **1 punto de accuracy son ~4 imágenes.** El piso de ruido medido de estos
  experimentos es de **~1.6 puntos** (ver §9), así que diferencias de ese orden entre dos
  filas no distinguen nada.

El salto grande fue el **transfer learning**: de 0.7708 (mejor CNN desde cero) a 0.8438
con el mismo número de clases, +7 puntos. Todo lo demás son ajustes de un par de puntos.

---

## 5. Modelo 1 — CNN desde cero (6 clases)

CNN entrenada desde cero con los scripts de `scripts/`. Búsqueda con Optuna sobre 12
hiperparámetros (arquitectura + entrenamiento), métrica optimizada **macro-AUC en
validación**.

**Búsqueda (2026-08-05):** 25 trials en 12 min — 12 completos, 12 podados, 1 descartado
por falta de VRAM. Mejor: trial 17, macro_auc 0.9143 (media de los completos 0.8802).

```
n_bloques 3 · filtros_base 16 · units_fc 128 · activacion elu · dropout 0.25
usar_bn False · weight_decay 3.4e-4 · aug media
optimizer adamw · lr 9.8e-4 · batch_size 64 · img_size 150 · class_weight False
```

Señal útil que dejó la búsqueda (`resultados/graficos/optuna_*.png`):

- `aug` es el hiperparámetro más influyente, y `aug=media` gana claro sobre `leve`.
  Coherente con un dataset chico: el aumento de datos ayuda, pero pasarse estorba.
- `batch_size=64` y `n_bloques=3` también se separan bien.
- `img_size` casi no influye → 150 px está bien, no vale la pena pagar 192 (que además
  es lo que provoca los OOM).
- **Cuidado**: esas importancias salen de solo 12 trials completos, son orientativas. El
  pruner cortó tanto que varias categorías quedaron con 1-2 muestras.

**Modelo final:** `entrenar_mejor.py --epocas 50 --patience 10`. Paró en la época 48 por
EarlyStopping; mejor época la 38, con val_accuracy 0.8011. Solo 155.558 parámetros.

**Test: accuracy 0.7448 · macro F1 0.7355 · macro AUC 0.9456**

| Clase | Precision | Recall | F1 | AUC | N |
|---|---|---|---|---|---|
| cardboard | 0.922 | 0.770 | 0.839 | 0.963 | 61 |
| glass | 0.691 | 0.618 | 0.653 | 0.907 | 76 |
| metal | 0.710 | 0.710 | 0.710 | 0.953 | 62 |
| paper | 0.851 | 0.889 | 0.870 | 0.978 | 90 |
| plastic | 0.641 | 0.685 | 0.662 | 0.907 | 73 |
| trash | 0.581 | 0.818 | 0.679 | 0.965 | 22 |

Detalle a explicar en el informe: el macro-AUC (0.946) es mucho más alto que el accuracy
(0.745). No es contradictorio — el modelo ordena bien las clases por probabilidad, pero
el `argmax` se equivoca seguido entre clases parecidas.

La confusión dominante es **glass ↔ plastic ↔ metal**: 12 vidrios predichos como metal y
12 como plástico; 10 plásticos predichos como vidrio. Tiene sentido físico (los tres son
transparentes o brillantes). `paper` y `cardboard` son las clases fáciles. `trash` tiene
recall alto (0.818) pero precision baja (0.581): el modelo la usa como cajón de sastre.

**Aquí Optuna PERDIÓ contra el baseline** (`entrenar_mejor.py --sin-optuna`, los
hiperparámetros originales del notebook: adam, lr 1e-3, batch 16, relu, dropout 0.3):

| Métrica (test) | Baseline | Optuna (trial 17) | Diferencia |
|---|---|---|---|
| Accuracy | **0.7708** | 0.7448 | −2.6 pts |
| Macro F1 | **0.7570** | 0.7355 | −0.022 |
| Macro AUC | **0.9517** | 0.9456 | −0.006 |
| val_accuracy (mejor época) | **0.8249** (ép. 44) | 0.8011 (ép. 38) | −0.024 |

El baseline gana en las cuatro métricas, incluida validación. El análisis de por qué
está en §9; el resumen es **desajuste de horizonte**: se buscó con 12 épocas y se
entrenó con 50.

Reporte completo en `resultados/reporte/`, gráficos en `resultados/graficos/` y
`resultados/matriz_de_confusion/`, pesos en `modelos/trashnet_pytorch_optuna.pt`.

Este modelo es **reproducible al último decimal**: se corrió dos veces y dio las mismas
cifras (misma mejor época, mismo accuracy, mismo AUC por clase), gracias a `semilla(42)`
en `trashnet_comun.py`. Se puede regenerar cualquier resultado perdido sin miedo a que
cambien los números del informe.

---

## 6. Modelo 2 — MobileNetV3Small, 6 clases (2026-08-06)

`TrashNet_PyTorch_MobileNetV3_DEFINITIVO.ipynb`. Réplica del notebook equivalente de
TensorFlow: mismo backbone, mismo protocolo, mismas métricas, para que la comparación
entre frameworks sea justa.

Arquitectura: MobileNetV3Small de ImageNet (`include_top=False`) → GlobalAvgPool →
Dense(units, relu) → Dropout → Dense(6). Entrenamiento en **2 fases**: backbone
congelado a lr 3e-4 (25 épocas), luego fine-tuning de los 3 últimos bloques a lr 1e-5
(15 épocas). `EarlyStopping` y `ReduceLROnPlateau` reimplementados a mano porque PyTorch
no trae callbacks.

**Test:**

| Métrica | Sin Optuna | Con Optuna |
|---|---|---|
| Accuracy | 0.8203 | **0.8438** |
| F1 macro | 0.8107 | **0.8336** |
| AUC macro OVR | 0.9761 | **0.9802** |

Ganador de la búsqueda: 10 trials, mejor trial 1 — `64 units, dropout 0.2, lr 8.71e-4,
adam`.

Sigue el mismo patrón por clase que el modelo 1, pero todo desplazado hacia arriba:
`glass` sube de F1 0.653 a 0.827 y `plastic` de 0.662 a 0.836. `trash` sigue siendo la
peor (F1 0.735, precision 0.667 con recall 0.818): cajón de sastre otra vez.

**Aquí Optuna GANÓ**, +2.34 puntos. La diferencia con el modelo 1 es que aquí la
búsqueda y el entrenamiento final usan el mismo protocolo de épocas.

Ojo: la corrida guardada en el notebook dice `Dispositivo: cpu`, o sea que se ejecutó con
un kernel sin CUDA. Los números son válidos pero si se re-ejecuta en GPU van a cambiar un
poco (el orden de las operaciones en float no es idéntico).

---

## 7. Modelo 3 — Jerárquico por contenedor, 4 clases (2026-08-10)

`jerarquico/TrashNet_PyTorch_Jerarquico_DEFINITIVO.ipynb`. Mismo dataset y mismo modelo
que el §6, pero el target cambia: las 6 clases de material se reasignan a **4 destinos de
contenedor**. La predicción es el contenedor recomendado, no el material.

| Material original | Contenedor |
|---|---|
| metal, plastic | **amarillo** |
| cardboard, paper | **azul** |
| trash | **gris** |
| glass | **verde** |

El remapeo se hace en código con el `target_transform` de `ImageFolder`
(`ORIGINAL_TO_CONTAINER = [1, 3, 0, 1, 0, 2]`), sin duplicar ni mover imágenes. Es el
equivalente del `tf.gather` sobre `tf.data` del notebook de TensorFlow.

**El reagrupamiento empeora el desbalance**: `gris` es solo `trash` (95 train / 22 test)
contra 697 de `azul`, así que su `class_weight` se va a **4.65**. Los pesos se pasan a
`CrossEntropyLoss(weight=...)`, el equivalente del `class_weight` de Keras.

**Test:**

| Métrica | Sin Optuna | Con Optuna |
|---|---|---|
| Accuracy | 0.8724 | **0.8880** |
| Precision macro | 0.8115 | **0.8270** |
| Recall macro | 0.8532 | **0.8762** |
| F1 macro | 0.8281 | **0.8461** |
| AUC macro OVR | 0.9764 | **0.9768** |

**Optuna gana en las cinco métricas** (+1.56 pts de accuracy = 6 imágenes) y también en
validación, que es donde se hizo la selección: **0.8912 contra 0.8700**. Ganador: trial
14, propuesto por el TPE — `256 units, dropout 0.30, lr 3.94e-4, rmsprop`.

Por clase (con Optuna): `azul` 0.963 de F1, `amarillo` 0.868, `verde` 0.837, `gris`
0.717. `gris` mantiene el patrón de siempre: recall 0.864 con precision 0.613, 12 falsos
positivos para 22 imágenes reales.

**El hallazgo importante: el reagrupamiento ayuda menos de lo que parece.** Juntar
`metal` + `plastic` en `amarillo` elimina dos de las tres direcciones de la confusión
glass ↔ plastic ↔ metal, y aun así el accuracy solo sube de 0.8438 (6 clases) a 0.8880.
La razón es que **el error residual es casi todo vidrio ↔ amarillo: 24 de los 43 errores
(56%)**, 12 en cada dirección. Ninguna agrupación por contenedor puede arreglar eso,
porque vidrio y plástico van a contenedores distintos.

Comparación con TensorFlow (mismo modelo y protocolo de entrenamiento):

| | TensorFlow | PyTorch | Dif. |
|---|---|---|---|
| Accuracy sin Optuna | 0.8568 | **0.8724** | +0.0156 |
| Accuracy con Optuna | 0.8594 | **0.8880** | +0.0286 |
| Recall macro sin Optuna | 0.8036 | **0.8532** | +0.0496 |
| Recall macro con Optuna | 0.7870 | **0.8762** | +0.0892 |

La fila "sin Optuna" es comparación directa. La de "con Optuna" **no lo es**: TensorFlow
usa 10 trials y `study.best_params`, PyTorch usa 30 trials con el baseline encolado y
selección por reentrenamiento del top-3 (§9). La ventaja consistente de PyTorch está en
**recall macro** (+5 a +9 puntos): reparte más predicciones hacia las clases minoritarias
en vez de jugar a lo seguro con `azul`. En TensorFlow `verde` se queda en recall 0.566
(el vidrio se le va al amarillo); aquí llega a 0.842.

---

## 8. Modelo 4 — Binario reciclable / basura, 2 clases (2026-08-10)

`binario/TrashNet_PyTorch_Binario_DEFINITIVO.ipynb`. Mismo backbone y mismo protocolo de
2 fases, con el target colapsado a dos clases: `basura` (solo `trash`) contra
`reciclable` (las otras cinco).

```
train        95 basura / 1671 reciclable   (5.4%, ratio 17.6:1)
validation   20 basura /  357 reciclable
test         22 basura /  362 reciclable
```

**El accuracy no sirve como métrica aquí.** Un modelo que siempre diga "reciclable" saca
**94.27%** en test sin detectar ni una basura. Por eso la métrica de decisión es la
**average precision (AP) sobre `basura`**, el EarlyStopping monitoriza AP en vez de
`val_accuracy`, y Optuna optimiza AP. `basura` se define como clase de interés aunque sea
la minoritaria: si se usara `reciclable`, precision y recall saldrían ~0.98 trivialmente.

Ganador: **trial 21** (propuesto por el TPE) — `256 units, dropout 0.25, lr 9.95e-4,
adam`. AP en validación 0.8126.

### El resultado principal: calibrar el umbral vale más que Optuna

| Test, clase basura | argmax (0.500) | **umbral calibrado (0.837)** |
|---|---|---|
| TP / FP / FN / TN | 19 / 17 / 3 / 345 | **18 / 4 / 4 / 358** |
| Precision | 0.5278 | **0.8182** |
| Recall | 0.8636 | 0.8182 |
| **F1** | 0.6552 | **0.8182** |
| Accuracy | 0.9479 | 0.9792 |
| Balanced accuracy | 0.9083 | 0.9036 |

AP en test: **0.7977** con Optuna, 0.7593 sin Optuna.

Calibrar el umbral sobre validación subió el F1 **+0.163**, más de cuatro veces la mejora
que aportó Optuna (+0.038 de AP). El mecanismo se ve en la matriz: `argmax` producía **17
falsos positivos** para 19 aciertos; subir el umbral a 0.837 los deja en **4**, a costa de
un solo verdadero positivo. Se cambian 13 falsos positivos por 1 falso negativo.

Es la lección más transferible del notebook: **con clases desbalanceadas el umbral es un
hiperparámetro más, y suele ser el que más rinde.** Los modelos 1, 2 y 3 usan `argmax`
implícitamente y ninguno lo justifica.

### Supera al jerárquico

El modelo 3 ya resolvía esta tarea de forma implícita con su clase `gris`, así que era el
baseline real a batir:

| | Precision | Recall | F1 |
|---|---|---|---|
| Trivial "todo reciclable" | — | 0.000 | 0.000 |
| Jerárquico, clase `gris` | 0.6129 | 0.8636 | 0.7170 |
| **Binario con umbral** | **0.8182** | 0.8182 | **0.8182** |

**+0.101 de F1.** Toda la ventaja está en precision (0.61 → 0.82) a cambio de algo de
recall. Es decir: el binario deja de usar la clase basura como cajón de sastre, que era el
problema crónico de `trash` y `gris` en los tres modelos anteriores.

### Evaluación fuera de distribución: el modelo NO generaliza

Las 453 imágenes de `dataset_gd/trash` (Garbage Dataset, CC BY 4.0) no se usaron para
entrenar ni para calibrar el umbral. Son fotos reales con fondo y contexto, tomadas con
móviles, frente a las fotos de estudio sobre fondo blanco de TrashNet.

| | Recall sobre basura |
|---|---|
| TrashNet test (22 imágenes) | **0.8182** |
| **Garbage Dataset (453 imágenes)** | **0.1192** (54 de 453) |

**Caída de 70 puntos.** Con `argmax` tampoco mejora (0.1457). Y no es duda del modelo: la
**mediana de `P(basura)` en GD es 0.0037** — afirma con total confianza que las fotos
reales de basura son reciclables.

El desglose por resolución descarta la explicación alternativa:

| Grupo | n | Recall | P(basura) media |
|---|---|---|---|
| lado menor < 224 px | 136 | 0.0074 | 0.0481 |
| **lado menor >= 224 px** | **317** | **0.1672** | 0.2166 |

Si el problema fuera la nitidez, el grupo de buena resolución funcionaría. Falla igual.
**Es cambio de dominio: el modelo aprendió el estudio fotográfico, no el objeto.**

Para el informe esto no es un fracaso, es el hallazgo mejor medido del proyecto:

> Un modelo con F1 0.818 y accuracy 97.9% en su propio test cae a recall 0.119 con fotos
> reales del mismo tipo de residuo. Los cuatro modelos están evaluados solo dentro de
> TrashNet, y este experimento cuantifica hasta dónde se pueden extrapolar sus números:
> prácticamente nada fuera del dominio de estudio.

Es una comprobación que la mayoría de trabajos sobre TrashNet no hace, y aquí está
respaldada con 453 imágenes y un control que descarta la calidad de imagen.

### Por qué se usó la carpeta `original` de GD y no las `standardized_*`

El dataset se descarga con tres variantes. Las dos `standardized_*` hacen las imágenes
cuadradas con **padding letterbox** gris `(114,114,114)`, y en algunas blanco. TrashNet
nunca tiene esas barras, así que evaluar con ellas metería un artefacto visual presente
**solo** en la clase de basura y sería imposible saber si el modelo falla por el objeto o
por el relleno. Con `original`, ambos conjuntos reciben el mismo trato: aspect ratio
nativo → `Resize((224,224))`. Está documentado en `dataset_gd/README.md`.

### Límites de este modelo

- **95 imágenes de basura en train.** Puede estar memorizándolas en vez de aprender la
  categoría.
- **20 en validación.** El umbral 0.837 se calibró sobre 20 ejemplos: un acierto o fallo
  mueve el recall 5 puntos, así que el umbral es ruidoso. Una alternativa más sólida sería
  validación cruzada de 5 folds solo para elegirlo.
- **22 en test.** Todas las métricas de la clase basura arrastran esa incertidumbre. Por
  eso la evaluación sobre GD, con 453 imágenes, es la evidencia más fuerte del notebook
  aunque solo mida recall.

### Nota sobre el notebook

La corrida del 2026-08-10 (1h42m) **se cerró sin guardar**, así que el `.ipynb` no tiene
outputs. Los pesos y las 10 figuras de `binario/predicciones/` sí quedaron, y todas las
cifras de esta sección se reconstruyeron cargando `trashnet_binario_optuna.pt` y
re-infiriendo sobre validación, test y GD. Lo único no recuperable es la tabla de trials
de Optuna y las curvas de entrenamiento. Al re-ejecutar, **guardar el notebook**.

---

## 9. Lo que aprendimos sobre Optuna (la parte más útil para el informe)

Optuna ganó en dos de los tres modelos y perdió en uno. El patrón es consistente y
explicable, y vale más contarlo así que presentar un número de accuracy.

### Las tres cosas que hacen que "Optuna no mejore"

**1. La búsqueda puede no ser bayesiana en absoluto.** `TPESampler` tiene
`n_startup_trials=10` por defecto: los primeros 10 trials se muestrean **al azar de la
distribución previa** y solo a partir del 11 el sampler construye su modelo. Con
exactamente 10 trials, los 10 son aleatorios y el TPE nunca se usa. Comprobado: con
`n_trials=10`, los parámetros de `TPESampler(seed=42)` coinciden **byte por byte** con
los de `RandomSampler(seed=42)`.

Esto afectaba al modelo 3 y **sigue afectando al notebook jerárquico de TensorFlow**, que
usa 10 trials. No invalida sus resultados, pero lo que corrió ahí fue random search de 10
muestras, no optimización bayesiana.

Con 30 trials el TPE sí aporta, y se ve: los 10 aleatorios dan media 0.8215 y bajan hasta
0.7427; los 20 informados dan media 0.8483 y su **peor** trial (0.8329) es mejor que 6 de
los 10 aleatorios. Lo que aprendió es concreto: todos los trials malos tienen learning
rate bajo, y el TPE abandonó por completo la región `lr < 1e-4`. El gráfico está en
`jerarquico/predicciones/jer_optuna_historia.png`.

**2. El baseline nunca se evalúa.** Optuna reporta el mejor de *lo que probó*; no se
compara contra nada. En el modelo 3 la configuración del baseline cabía perfectamente en
el espacio de búsqueda y en 10 tiros sobre 4 dimensiones no salió. La solución es
**encolarla como trial 0** con `study.enqueue_trial(...)`: así la búsqueda arranca
sabiendo qué hay que batir. Dato interesante del modelo 3: el baseline quedó **20º de 30**
en la métrica de búsqueda, o sea que 19 configuraciones le ganaban.

**3. Desajuste de horizonte.** El ranking de los trials se mide con pocas épocas, pero el
modelo final se entrena con muchas más. "Mejor a 8 épocas" no es la misma pregunta que
"mejor a 25+15 en dos fases", y las configuraciones con learning rate alto arrancan
rápido y luego se desestabilizan. Esta fue **la causa principal de la derrota en el
modelo 1** (12 épocas de búsqueda contra 50 de entrenamiento) y también de la primera
versión del modelo 3, donde la ganadora reventó en la época 6 (val_loss de 0.505 a 0.796)
y disparó el EarlyStopping en la 8, mientras el baseline usaba las 25 épocas completas.

La solución es **reentrenar el top-k a presupuesto completo y elegir con eso**, en vez de
confiar en el ranking corto. Es la práctica estándar y es barata (3 configs × ~5 min).

### El piso de ruido: ~1.6 puntos

Antes de afirmar que un modelo es mejor que otro hay que comparar la diferencia contra el
ruido del experimento. En la primera versión del modelo 3, **la misma configuración dio
val_accuracy 0.8727 durante la búsqueda y 0.8568 al reentrenarla**: 1.6 puntos de
diferencia por puro estado del RNG, porque los trials previos habían consumido
aleatoriedad y el orden del shuffle y del augmentation cambió.

De ahí salió el arreglo de llamar a **`semilla()` al inicio de cada trial**, que reinicia
`random`, `numpy`, `torch`, CUDA y el `generator` del DataLoader. Reduce el ruido pero no
lo elimina: sigue habiendo una sola semilla por configuración.

Con 384 imágenes de test, 1 punto de accuracy son 4 imágenes. Cualquier diferencia por
debajo de ~1.6 puntos hay que reportarla como **"no se puede distinguir"**, no como una
mejora. Vale para el +0.26 de TensorFlow y también para el +1.56 del modelo 3 — en ese
caso el argumento fuerte no es el accuracy solo, es que gana en las cinco métricas y
también en validación con el mismo criterio con que se eligió.

### Y aun así, el umbral rinde más que la búsqueda

En el modelo 4, con la búsqueda ya corregida, Optuna aportó +0.038 de AP y **calibrar el
umbral aportó +0.163 de F1**. Cuatro veces más, y sin entrenar nada.

Conviene tenerlo presente antes de invertir una hora en 30 trials: en problemas
desbalanceados, el orden de rentabilidad suele ser **(1) elegir bien la métrica, (2)
calibrar el umbral, (3) buscar hiperparámetros**. Los modelos 1, 2 y 3 se saltaron los dos
primeros pasos.

### Cuál de los tres arreglos sirvió

En el modelo 3, **el que movió la aguja fue subir de 10 a 30 trials**. El reentrenamiento
del top-3 salió no-op: el ranking se mantuvo `[14, 4, 12]` y los tres mejoraron al pasar
a presupuesto completo. Igual conviene mantenerlo, porque es lo que permite *afirmar* que
el ranking aguanta en vez de suponerlo.

---

## 10. Decisiones y trampas (leer antes de tocar el código)

**Sin scikit-learn.** El env no lo tiene, y el notebook original ya calculaba las
métricas a mano con NumPy por eso. Todo lo demás mantiene ese enfoque: matriz de
confusión, precision/recall/F1, `classification_report` y ROC/AUC (regla del trapecio)
están implementados en `scripts/trashnet_comun.py` y replicados en los notebooks 2 y 3.
Son genéricos en el número de clases, así que el mismo código sirve para 6 y para 4.
Consecuencia colateral: la importancia de hiperparámetros usa |correlación de Spearman|
en vez de fANOVA, porque `optuna.importance` requiere sklearn.

**Límite de VRAM (4 GB).** En el modelo 1, `img_size=192` + `batch_size=64` +
`filtros_base=48` no cabe y lanza `torch.OutOfMemoryError`. Está manejado: el trial se
marca `PRUNED` en vez de reventar. Dos cosas que costaron caro descubrir:

1. Optuna **re-lanza** por defecto cualquier excepción del objective, así que un solo OOM
   mataba la búsqueda completa y se perdía todo el cómputo previo. Se resolvió con
   `catch=(RuntimeError,)` en `study.optimize` + `try/except` sobre `torch.OutOfMemoryError`.
2. Si hay un kernel de Jupyter con **TensorFlow** vivo, TF reserva casi toda la VRAM al
   arrancar y PyTorch se queda sin nada. **Cerrar los notebooks de TF antes de entrenar.**

Los modelos 2 y 3 no tienen problema de VRAM: MobileNetV3Small a 224px con batch 32 son
~1M de parámetros y entra de sobra.

**Jupyter sobrescribe los cambios hechos fuera.** Si el notebook está abierto en Jupyter y
alguien edita el `.ipynb` desde fuera, al guardar Jupyter escribe su copia en memoria y
revierte esos cambios. Pasó con el notebook 3. **Cerrar y recargar el notebook antes de
editarlo por fuera.**

**BatchNorm en el fine-tuning.** En Keras, `layer.trainable = False` sobre una
BatchNorm congela también sus estadísticas. En PyTorch hacen falta **dos cosas**:
`requires_grad = False` en sus parámetros *y* dejar el módulo en modo `eval()`. Por eso
`correr_epoca()` llama siempre a `model.features.eval()`, en las dos fases. Si se omite,
las estadísticas de ImageNet se pisan con las del dataset chico y el fine-tuning se
degrada sin dar ningún error.

**`AdaptiveAvgPool2d` en lugar del `Flatten` fijo.** El notebook 1 tiene
`Linear(64*17*17, 128)`, que ata el modelo a `img_size=150` y exactamente 3 bloques conv.
`TrashNetCNN` usa pooling adaptativo para que Optuna pueda variar profundidad y
resolución. Por eso el modelo de los scripts **no es cargable** en el notebook original y
viceversa.

**Valor de los trials podados.** En Optuna 4, un trial `PRUNED` guarda en `value` su
último valor intermedio: un puntaje a medio entrenar. Mezclarlos con los completos
falsea las estadísticas (pasó: la media aparecía como 0.7854 en vez de 0.8802). Filtrar
siempre por `state == COMPLETE`. `study.best_value` ya considera solo los completos, así
que ese sí es confiable.

**Test se toca una sola vez.** Durante la búsqueda y la selección solo se usa validación.
En los tres modelos la evaluación en test ocurre una única vez, al final; si no, la
métrica final quedaría contaminada por la selección de hiperparámetros.

**Un study por presupuesto.** `optuna/study.db` tiene los trials del modelo 1 evaluados a
12 épocas. Mezclar ahí trials con otro número de épocas hace que los valores no sean
comparables entre sí. Los notebooks 2 y 3 crean su study **en memoria** justamente para
no contaminarlo.

**Accuracy bajo desbalance fuerte.** En el modelo 4 el 94.3% del test es `reciclable`:
el modelo trivial que siempre dice esa clase saca 0.9427 de accuracy y no detecta nada.
Cualquier accuracy que se reporte en ese notebook hay que compararla contra ese número, no
contra 0.5. La **balanced accuracy** (media de los recalls por clase) sí vale: el trivial
saca 0.5.

**AUC contra AP.** Con pocos positivos la curva ROC es engañosamente optimista, porque la
tasa de falsos positivos apenas se mueve aunque haya muchos falsos positivos en términos
absolutos. La curva **precision-recall** y su AP sí lo reflejan. En el modelo 4 la AUC sale
muy por encima de la AP y no es una contradicción: es el desbalance. Reportar solo la AUC
en un problema así es engañoso.

**El umbral no es 0.5 por defecto.** Usar `argmax` equivale a fijar el umbral en 0.5 sin
justificarlo. Con clases desbalanceadas el óptimo casi nunca está ahí. Se calibra **sobre
validación** y solo entonces se evalúa test. Ver §9.

**Cambio de dominio al mezclar datasets.** Las imágenes de TrashNet son objetos aislados
sobre fondo blanco de estudio; casi cualquier dataset externo de residuos son fotos reales
con fondo y contexto. Si se mezclan como clases distintas, el modelo aprende
"fondo blanco = clase A, fondo con contexto = clase B" y da métricas excelentes sin mirar
el objeto. Por eso `dataset_gd/` **solo se usa para evaluar**, nunca para entrenar. Si
algún día se entrena con datos externos, hay que hacer la ablación de fondo antes de
creerse el resultado.

**Desbalance del dataset.** `paper` tiene 415 imágenes de train y `trash` solo 95, así
que accuracy premia ignorar `trash`. Por eso en el modelo 1 la métrica por defecto es
macro-AUC, y en los modelos 2 y 3 se usan `class_weight` en la loss. En las 4 clases del
modelo 3 el desbalance es peor todavía (`gris` = 95 contra `azul` = 697).

---

## 11. Qué sigue

En orden de utilidad:

1. **Cerrar la brecha de dominio, que ahora está medida (§8).** Es lo más importante que
   queda: hoy sabemos que el mejor modelo cae de recall 0.818 a 0.119 con fotos reales.
   Tres caminos, de menos a más esfuerzo: (a) augmentation que simule el mundo real
   —fondos aleatorios, recortes, desenfoque, cambios fuertes de iluminación—; (b)
   entrenar dentro de GD, que tiene 12.259 imágenes en 10 clases y dominio consistente;
   (c) fotografiar 100-200 objetos de rechazo en las mismas condiciones que TrashNet, que
   es la única forma de conseguir negativos en el dominio original y daría datos propios
   para el informe.
2. **Atacar la confusión vidrio ↔ amarillo (o glass ↔ plastic ↔ metal).** Es donde se
   pierde más de la mitad del accuracy restante en los modelos multiclase, y ninguna
   reagrupación de clases lo resuelve (§7). Ideas: aumento de datos con más variación de
   color/brillo/reflejos, un backbone más grande (MobileNetV3**Large** o ResNet18), o
   entrenar un clasificador binario dedicado solo a distinguir vidrio de plástico/metal
   sobre las imágenes que el modelo principal manda a esas dos clases.
3. **Arreglar la búsqueda del modelo 1 con lo aprendido en §9.** Hoy es el único de los
   tres donde Optuna pierde, y ya sabemos por qué. Los tres cambios son barajos:

   ```bash
   conda activate frameworks-ia
   cd PyTorch/scripts
   python optuna_trashnet.py --trials 30 --epocas 30 --study-name trashnet_30ep
   python entrenar_mejor.py --epocas 50 --patience 10
   ```

   Falta añadir a los scripts el `enqueue_trial` del baseline y el reentrenamiento del
   top-k, que en el notebook 3 ya están implementados y se pueden copiar de ahí.
4. **Corregir el notebook jerárquico de TensorFlow**, que sigue con 10 trials y por lo
   tanto con random search (§9, punto 1). Es de Tomás; conviene avisarle. Con eso la
   comparación entre frameworks volvería a ser directa en las dos filas.
5. **Re-ejecutar el modelo 2 en GPU**, que quedó corrido en CPU, y **re-ejecutar y
   guardar el modelo 4**, cuyo notebook quedó sin outputs.
6. **Varias semillas por configuración** si hay tiempo. Es la única forma de bajar el piso
   de ruido de 1.6 puntos y poder afirmar diferencias chicas con algo de rigor. 3 semillas
   × el modelo final son ~15 min.

---

## 12. Comandos de referencia

```bash
conda activate frameworks-ia
nvidia-smi                        # comprobar que la GPU está libre antes de entrenar

# Modelo 1 — búsqueda (relanzar continúa el study; --trials cuenta trials NUEVOS)
cd PyTorch/scripts
python optuna_trashnet.py --trials 25 --epocas 12
python optuna_trashnet.py --metrica val_accuracy     # o macro_f1
python optuna_trashnet.py --timeout 900              # cortar por tiempo

# Modelo 1 — reentrenamiento y evaluación
python entrenar_mejor.py --epocas 50 --patience 10
python entrenar_mejor.py --sin-optuna
```

Modelos 2, 3 y 4: son notebooks, se ejecutan de arriba a abajo con el kernel de
`frameworks-ia`. Los modelos 3 y 4 tardan **~1 hora** cada uno en la GTX 1650 SUPER
(Parte A ~6 min, 30 trials ~35 min, top-3 ~15 min). Si hay que recortar, `TOP_K = 2` ahorra ~5 min y
`OPTUNA_TRIALS = 20` unos 12 min más, aunque con 20 trials el TPE se queda con solo 10
informados.

Si una corrida de los scripts se cae, no se pierde nada: el study vive en
`optuna/study.db` y relanzar el mismo comando continúa donde quedó. Los notebooks, en
cambio, tienen el study en memoria: si se interrumpe la búsqueda hay que repetirla.

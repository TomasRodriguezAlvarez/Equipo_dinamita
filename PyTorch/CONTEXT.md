# CONTEXT — rama PyTorch del proyecto TrashNet

Documento de traspaso: en qué estado quedó la parte de PyTorch, qué decisiones se
tomaron y por qué, y qué sigue. Última actualización: **2026-08-05**.

---

## 1. Qué es esto

Clasificación de residuos en 6 clases (`cardboard, glass, metal, paper, plastic, trash`)
con PyTorch. Es la contraparte de la implementación en TensorFlow del equipo
(`tensorflow/`), sobre el mismo dataset.

Responsable de esta parte: Benjamín. El resto del equipo trabaja la versión de
TensorFlow y el informe conjunto en `resultados/` (raíz).

Estado actual: **búsqueda con Optuna y reentrenamiento final hechos.** Resultado en
test: **accuracy 74.5%, macro-AUC 0.9456**. Falta el baseline para comparar (ver §6).

---

## 2. Entorno

```bash
conda activate frameworks-ia
```

Es el **único** env con PyTorch (`base` y `~/ek-venv` no lo tienen).

| Paquete | Versión | Nota |
|---|---|---|
| torch | 2.12.1+cu126 | CUDA disponible |
| torchvision | 0.27.1+cu126 | |
| optuna | 4.9.0 | instalado el 2026-08-05 para este trabajo |
| numpy | 2.4.4 | `np.trapz` fue renombrado a `np.trapezoid` |
| matplotlib | 3.11.0 | |
| tensorflow | 2.21.0 | conviven en el mismo env |
| **scikit-learn** | **NO instalado** | ver §5 |

GPU: **NVIDIA GTX 1650 SUPER, 4 GB (3.63 GiB usables)**. Es la restricción más
importante de todo este trabajo — ver §5.

---

## 3. Estructura

```
PyTorch/
├── CONTEXT.md                      este archivo
├── TrashNet_PyTorch.ipynb          notebook original (CNN desde cero, 15 épocas)
├── Lab1_Tensores_Autograd.ipynb    ejercicio de clase
├── scripts/
│   ├── README.md                   uso detallado de los scripts
│   ├── trashnet_comun.py           módulo compartido (datos, modelo, métricas, gráficos)
│   ├── optuna_trashnet.py          búsqueda de hiperparámetros
│   └── entrenar_mejor.py           reentrenamiento final + evaluación en test
├── optuna/
│   ├── study.db                    todos los trials (SQLite; permite reanudar)
│   └── mejores_params.json         mejor combinación encontrada
├── resultados/                     ← salidas de los scripts
│   ├── graficos/
│   ├── matriz_de_confusion/
│   └── reporte/
├── modelos/                        pesos .pt
└── predicciones/                   salidas del NOTEBOOK (no de los scripts)
```

Los scripts **no escriben** en la carpeta `resultados/` de la raíz del repo: esa es
del informe conjunto del equipo. Si hay que aportar gráficos al informe, se copian a
mano desde `PyTorch/resultados/`.

Dataset en `../dataset/` con splits ya hechos: 1766 train / 377 val / 384 test.

---

## 4. Resultados de la búsqueda (2026-08-05)

25 trials útiles en **12 min**: 12 completos, 12 podados por el pruner, 1 descartado
por falta de VRAM. Métrica optimizada: **macro-AUC en validación**.

**Mejor: trial 17 — macro_auc = 0.9143** (media de los completos 0.8802, peor 0.7911).

```
n_bloques 3 · filtros_base 16 · units_fc 128 · activacion elu · dropout 0.25
usar_bn False · weight_decay 3.4e-4 · aug media
optimizer adamw · lr 9.8e-4 · batch_size 64 · img_size 150 · class_weight False
```

Lo que se observó en los gráficos (`resultados/graficos/optuna_*.png`):

- `aug` es el hiperparámetro más influyente, y `aug=media` gana claro sobre `leve`.
  Coherente con un dataset chico: el aumento de datos ayuda, pero pasarse estorba.
- `batch_size=64` y `n_bloques=3` también se separan bien.
- `img_size` casi no influye → 150 px está bien, no vale la pena pagar 192 (que además
  es lo que provoca los OOM).
- **Cuidado**: esas importancias salen de solo 12 trials completos, son orientativas.
  El pruner cortó tanto que varias categorías quedaron con 1-2 muestras (`relu`,
  `rmsprop` y `aug=fuerte` casi no aparecen).

---

## 4b. Modelo final (2026-08-05, 22:05)

`entrenar_mejor.py --epocas 50 --patience 10` con los hiperparámetros del trial 17.
Paró en la época 48 por EarlyStopping; mejor época la 38, con val_accuracy 0.8011.
Modelo pequeño: 155.558 parámetros.

**Test: accuracy 0.7448 (74.5%) · macro F1 0.7355 · macro AUC 0.9456**

| Clase | Precision | Recall | F1 | AUC | N |
|---|---|---|---|---|---|
| cardboard | 0.922 | 0.770 | 0.839 | 0.963 | 61 |
| glass | 0.691 | 0.618 | 0.653 | 0.907 | 76 |
| metal | 0.710 | 0.710 | 0.710 | 0.953 | 62 |
| paper | 0.851 | 0.889 | 0.870 | 0.978 | 90 |
| plastic | 0.641 | 0.685 | 0.662 | 0.907 | 73 |
| trash | 0.581 | 0.818 | 0.679 | 0.965 | 22 |

Detalle a explicar en el informe: el macro-AUC (0.946) es mucho más alto que el
accuracy (0.745). No es contradictorio — el modelo ordena bien las clases por
probabilidad, pero el `argmax` se equivoca seguido entre clases parecidas.

La confusión dominante es **glass ↔ plastic ↔ metal**: 12 vidrios predichos como metal
y 12 como plástico; 10 plásticos predichos como vidrio. Tiene sentido físico (los tres
son transparentes o brillantes). `paper` y `cardboard` son las clases fáciles.
`trash` tiene recall alto (0.818) pero precision baja (0.581): el modelo la usa como
cajón de sastre.

Reporte completo en `resultados/reporte/reporte_optuna.txt`, gráficos en
`resultados/graficos/` y `resultados/matriz_de_confusion/`, pesos en
`modelos/trashnet_pytorch_optuna.pt`.

El entrenamiento es **reproducible**: se corrió dos veces y dio las mismas cifras hasta
el último decimal (misma mejor época, mismo accuracy, mismo AUC por clase), gracias a
`semilla(42)` en `trashnet_comun.py`. Útil saberlo: se puede regenerar cualquier
resultado perdido sin miedo a que cambien los números del informe.

---

## 4c. Comparación con el baseline — Optuna NO ganó

`entrenar_mejor.py --sin-optuna` entrena la misma CNN con los hiperparámetros originales
del notebook (`PARAMS_BASELINE`: adam, lr 1e-3, batch 16, relu, dropout 0.3, sin
weight_decay). Mismo presupuesto: hasta 50 épocas con EarlyStopping.

| Métrica (test) | Baseline (notebook) | Optuna (trial 17) | Diferencia |
|---|---|---|---|
| Accuracy | **0.7708** | 0.7448 | −2.6 pts para Optuna |
| Macro F1 | **0.7570** | 0.7355 | −0.022 |
| Macro AUC | **0.9517** | 0.9456 | −0.006 |
| val_accuracy (mejor época) | **0.8249** (ép. 44) | 0.8011 (ép. 38) | −0.024 |

**El baseline gana en las cuatro métricas, incluida validación.** Por clase, la ventaja
del baseline está justo en las difíciles: `plastic` F1 0.745 vs 0.662 y `glass` 0.699 vs
0.653. Optuna solo gana marginalmente en `paper`, `metal` y `trash`.

### Por qué pasó esto (importante para el informe)

1. **Desajuste de horizonte, la causa principal.** La búsqueda evaluó cada trial con
   **12 épocas**, pero el entrenamiento final corre hasta **50**. Optuna optimizó
   "mejor configuración a 12 épocas", que no es la misma pregunta que "mejor
   configuración a 50 épocas". La config ganadora usa `batch_size=64`, que avanza rápido
   al principio; el baseline con `batch_size=16` hace ~4× más actualizaciones por época
   y sigue mejorando cuando el otro ya se estancó.
2. **Poca evidencia.** Solo 12 trials completos, y el pruner descartó configuraciones
   que quizá eran buenas a largo plazo justamente por ir lento al principio.
3. **Las diferencias son chicas y de una sola semilla.** El test tiene 384 imágenes:
   2.6 puntos de accuracy son ~10 imágenes. No es una diferencia concluyente en ningún
   sentido — tampoco permitiría afirmar que el baseline es "mejor" de forma robusta.

Esto **no invalida el trabajo con Optuna**: la búsqueda sí encontró señal útil y
reproducible (`aug=media` claramente mejor que `leve`, `img_size=192` innecesario,
`n_bloques=3` mejor que 2). Lo que muestra es una trampa metodológica real y muy común,
y reportarla honestamente vale más que esconderla.

### Cómo se arreglaría

- **Subir las épocas por trial** para que el horizonte de búsqueda se parezca al final
  (`--epocas 30`), a costa de ~2.5× más tiempo. Es la solución directa.
- **Reentrenar el top-k a presupuesto completo** y elegir con eso, en vez de confiar en
  el ranking a 12 épocas. Es la práctica estándar y es barata: 3-5 configs × 3 min.
- Suavizar el pruner (`n_warmup_steps` más alto) para no matar configs de arranque lento.

---

## 5. Decisiones y trampas (leer antes de tocar el código)

**Sin scikit-learn.** El env no lo tiene, y el notebook original ya calculaba las
métricas a mano con NumPy por eso. Los scripts mantienen ese enfoque: matriz de
confusión, precision/recall/F1 y ROC/AUC (regla del trapecio) están implementadas en
`trashnet_comun.py`. Consecuencia colateral: la importancia de hiperparámetros usa
|correlación de Spearman| en vez de fANOVA, porque `optuna.importance` requiere sklearn.
Si algún día se instala sklearn, conviene cambiar a `optuna.importance.get_param_importances`.

**Límite de VRAM (4 GB).** La combinación `img_size=192` + `batch_size=64` +
`filtros_base=48` no cabe y lanza `torch.OutOfMemoryError`. Está manejado: el trial se
marca `PRUNED` en vez de reventar. Dos cosas que costaron caro descubrir:

1. Optuna **re-lanza** por defecto cualquier excepción del objective, así que un solo
   OOM mataba la búsqueda completa y se perdía todo el cómputo previo. Se resolvió con
   `catch=(RuntimeError,)` en `study.optimize` + `try/except` sobre `torch.OutOfMemoryError`.
2. Si hay un kernel de Jupyter con **TensorFlow** vivo, TF reserva casi toda la VRAM al
   arrancar y PyTorch se queda sin nada. **Cerrar los notebooks de TF antes de entrenar.**

**`AdaptiveAvgPool2d` en lugar del `Flatten` fijo.** El notebook tiene
`Linear(64*17*17, 128)`, que ata el modelo a `img_size=150` y exactamente 3 bloques
conv. `TrashNetCNN` usa pooling adaptativo para que Optuna pueda variar profundidad y
resolución. Por eso el modelo de los scripts **no es cargable** en el notebook original
y viceversa.

**Valor de los trials podados.** En Optuna 4, un trial `PRUNED` guarda en `value` su
último valor intermedio: un puntaje a medio entrenar. Mezclarlos con los completos
falsea las estadísticas (pasó: la media aparecía como 0.7854 en vez de 0.8802). Filtrar
siempre por `state == COMPLETE` — eso hace `_trials_completos()`. `study.best_value` de
Optuna ya considera solo los completos, así que ese sí es confiable.

**Test se toca una sola vez.** Durante la búsqueda solo se usa validación. La
evaluación en test ocurre únicamente en `entrenar_mejor.py`; si no, la métrica final
quedaría contaminada por la selección de hiperparámetros.

**Desbalance del dataset.** `paper` tiene 415 imágenes de train y `trash` solo 95, así
que accuracy premia ignorar `trash`. Por eso la métrica por defecto es macro-AUC y
existe el hiperparámetro `class_weight` (pondera la `CrossEntropyLoss`). Curiosamente
Optuna eligió `class_weight=False`, vale la pena revisar si eso se sostiene con más trials.

---

## 6. Qué sigue

En orden de utilidad:

1. **Cerrar el problema del §4c: buscar con un horizonte parecido al final.** Es lo más
   importante que falta, porque hoy el baseline le gana a Optuna.

   ```bash
   conda activate frameworks-ia
   cd PyTorch/scripts
   python optuna_trashnet.py --trials 15 --epocas 30    # ~25-30 min
   python entrenar_mejor.py --epocas 50 --patience 10
   ```

   Ojo: el study actual (`optuna/study.db`) tiene trials evaluados a 12 épocas. Mezclar
   ahí trials de 30 épocas hace que los valores no sean comparables entre sí. Conviene
   usar `--study-name trashnet_30ep` para arrancar un study limpio y conservar el viejo
   como evidencia de la comparación.
2. **Más trials si las importancias van al informe.** `python optuna_trashnet.py --trials 25`
   continúa el mismo study (no empieza de cero) y llevaría los completos de 12 a ~24,
   con lo que el gráfico de importancia deja de ser solo orientativo.
3. **Atacar la confusión glass ↔ plastic ↔ metal**, que es donde se pierde casi todo el
   accuracy (ver §4b). Ideas: `class_weight=True` forzado, aumento de datos con más
   variación de color/brillo, o directamente un backbone preentrenado (punto 4).
4. **Transfer learning.** La CNN es desde cero y el techo se siente cerca de 0.91 de
   macro-AUC. El lado de TensorFlow ya usa MobileNetV3Small; hacer lo equivalente con
   `torchvision.models` (MobileNetV3 o ResNet18 preentrenados, congelando el backbone)
   probablemente dé el salto más grande, y haría la comparación entre frameworks más
   justa. Ojo con la VRAM: congelar el backbone ayuda bastante.

---

## 7. Comandos de referencia

```bash
# Búsqueda (relanzar continúa el study; --trials cuenta trials NUEVOS)
python optuna_trashnet.py --trials 25 --epocas 12
python optuna_trashnet.py --metrica val_accuracy     # o macro_f1
python optuna_trashnet.py --timeout 900              # cortar por tiempo

# Reentrenamiento y evaluación
python entrenar_mejor.py --epocas 50 --patience 10
python entrenar_mejor.py --sin-optuna

# Comprobar que la GPU está libre antes de entrenar
nvidia-smi
```

Si una corrida se cae, no se pierde nada: el study vive en `optuna/study.db` y
relanzar el mismo comando continúa donde quedó.

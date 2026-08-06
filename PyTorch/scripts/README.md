# TrashNet · PyTorch + Optuna

Optimización de hiperparámetros y evaluación del clasificador de residuos (6 clases).
Adaptación de `scripts/optuna_ejemplo.py` (Iris + Keras) al modelo de
`PyTorch/TrashNet_PyTorch.ipynb`.

## Archivos

| Archivo | Qué hace |
|---|---|
| `trashnet_comun.py` | Módulo compartido: dataloaders, `TrashNetCNN` parametrizable, bucle de época, métricas / matriz de confusión / ROC calculadas a mano con NumPy y los gráficos. No se ejecuta solo. |
| `optuna_trashnet.py` | Búsqueda de hiperparámetros con Optuna (TPE + MedianPruner). Guarda el study, los mejores parámetros y los gráficos del estudio. |
| `entrenar_mejor.py` | Reentrena con los mejores hiperparámetros, evalúa en test y genera matriz de confusión, curvas ROC y reporte. |

Las métricas se calculan con NumPy (sin scikit-learn), igual que en el notebook,
porque el entorno `frameworks-ia` no tiene sklearn instalado.

## Entorno

```bash
conda activate frameworks-ia     # torch 2.12 + CUDA, optuna 4.9
cd PyTorch/scripts
```

## Uso

```bash
# 1) Buscar hiperparámetros (relanzarlo continúa el mismo study en SQLite)
python optuna_trashnet.py --trials 25 --epocas 12

# 2) Reentrenar con los mejores y evaluar en test
python entrenar_mejor.py --epocas 50 --patience 10

# 3) Baseline para comparar (hiperparámetros del notebook original)
python entrenar_mejor.py --sin-optuna
```

Opciones útiles de `optuna_trashnet.py`:

- `--metrica {macro_auc, val_accuracy, macro_f1}` — qué maximizar en validación.
  Por defecto `macro_auc`, coherente con las gráficas ROC del informe y más
  robusto al desbalance del dataset (paper ~415 imágenes vs trash ~95).
- `--timeout 3600` — cortar la búsqueda por tiempo (segundos).
- `--workers N` — `num_workers` de los DataLoader.

## Hiperparámetros que se exploran

Arquitectura: `n_bloques`, `filtros_base`, `units_fc`, `activacion`
Regularización: `dropout`, `usar_bn`, `weight_decay`, `aug` (intensidad del aumento de datos)
Optimización: `optimizer`, `lr`, `batch_size`, `img_size`
Desbalance: `class_weight` (ponderar la loss por frecuencia de clase)

La CNN usa `AdaptiveAvgPool2d` en vez del `Flatten` de tamaño fijo del notebook,
por eso `img_size` y `n_bloques` pueden variar libremente.

## Salidas

Todo queda bajo `PyTorch/`. Los scripts no escriben en la carpeta `resultados/` de la
raíz, que es del informe conjunto del equipo.

```
PyTorch/optuna/
  study.db                          base SQLite con todos los trials (permite reanudar)
  mejores_params.json               mejor combinación (la lee entrenar_mejor.py)

PyTorch/resultados/
  graficos/
    optuna_historia.png             métrica por trial + distribución
    optuna_importancia.png          importancia de cada hiperparámetro
    optuna_por_parametro.png        métrica vs cada hiperparámetro
    roc_curves_optuna.png           ROC one-vs-rest con AUC por clase
    curvas_entrenamiento_optuna.png accuracy y loss por época
  matriz_de_confusion/
    confusion_matrix_optuna.png     matriz + tabla de precision/recall/F1
  reporte/
    reporte_optuna.txt              resumen completo en texto

PyTorch/modelos/trashnet_pytorch_optuna.pt    pesos + class_names + hiperparámetros
```

Con `--sin-optuna` el sufijo pasa a `_baseline`, así los dos quedan lado a lado para
comparar. `PyTorch/predicciones/` se deja intacta: es la salida del notebook.

## Notas de metodología

- Durante la búsqueda solo se usa **validación**. El conjunto de **test** se toca
  una única vez, en `entrenar_mejor.py`; de lo contrario la métrica final estaría
  contaminada por la selección de hiperparámetros.
- El pruning (`MedianPruner`) es clave aquí: cada trial entrena una CNN completa,
  así que abortar temprano los trials malos es lo que hace viable la búsqueda.
  Las primeras 5 combinaciones se entrenan completas para tener referencia.
- Todos los trials arrancan con la misma semilla (42) para que sean comparables.

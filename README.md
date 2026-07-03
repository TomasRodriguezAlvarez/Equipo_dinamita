# Equipo Dinamita — TrashNet (PyTorch + TensorFlow)

Integrantes:
- Tomás Rodríguez
- Benjamín Sanchez

Link de descarga del dataset:
https://www.kaggle.com/datasets/feyzazkefe/trashnet

Clasificador de imágenes de residuos en **6 categorías**:

- cardboard
- glass
- metal
- paper
- plastic
- trash

---

# Archivos entregados

La entrega del proyecto considera los siguientes archivos:

- `TrashNet_PyTorch.ipynb`
- `TrashNet_TensorFlow_MobileNetV3.ipynb`
- `README.md`
- `separar_imagenes.py`
- Carpeta `imagenes_a_test/`
- Modelo final entrenado de PyTorch (`.pt`)
- Modelo final entrenado de TensorFlow (`.keras`)
- Link de descarga del dataset original TrashNet

Durante el desarrollo también se utilizaron archivos auxiliares (checkpoints, modelos intermedios, predicciones y resultados), los cuales no forman parte de la entrega final.

---

# Preparación del proyecto

## 1. Descargar el dataset

Descargar el dataset desde el enlace entregado junto con el proyecto.

Una vez descargado, ubicar el contenido dentro de:

```
dataset/dataset-resized/
```

La estructura inicial debe quedar:

```
dataset/
└── dataset-resized/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

---

## 2. Generar automáticamente los conjuntos de entrenamiento

Ejecutar:

```bash
python separar_imagenes.py
```

Este script divide automáticamente el dataset utilizando:

- 70% Entrenamiento
- 15% Validación
- 15% Test

manteniendo la estructura por clases.

Al finalizar se crearán automáticamente:

```
dataset/
├── train/
├── validation/
└── test/
```

No es necesario realizar esta separación manualmente.

> Si estas carpetas ya existen, no es necesario volver a ejecutar el script.

---

# Requisitos

- Python 3.11
- TensorFlow
- PyTorch
- NumPy
- Matplotlib
- Pillow
- Jupyter Notebook

Instalación:

```bash
pip install tensorflow torch torchvision numpy matplotlib pillow jupyter ipykernel
```

---

# Ejecución

## PyTorch

Abrir:

```
TrashNet_PyTorch.ipynb
```

El notebook realiza:

- carga del dataset
- entrenamiento
- evaluación
- generación de métricas
- matriz de confusión
- predicción sobre imágenes nuevas
- guardado del modelo

---

## TensorFlow

Abrir:

```
TrashNet_TensorFlow_MobileNetV3.ipynb
```

El notebook realiza:

- carga del dataset
- cálculo de class weights
- data augmentation
- Transfer Learning con MobileNetV3Small
- entrenamiento
- evaluación
- métricas
- curvas ROC
- matriz de confusión
- guardado del modelo

---

# Modelos entregados

## PyTorch

Modelo entrenado:

```
trashnet_pytorch.pt
```

## TensorFlow

Modelo entrenado:

```
trashnet_mobilenetv3_small_fase1.keras
```

---

# Prueba de imágenes

La carpeta

```
imagenes_a_test/
```

contiene imágenes independientes para probar ambos modelos sin necesidad de utilizar el conjunto de test.

Cada notebook incluye una sección para cargar una imagen y obtener:

- clase predicha
- probabilidad/confianza
- visualización de la imagen

---

# Notas

- Ambos modelos fueron entrenados utilizando el mismo dataset.
- El dataset original no se incluye directamente en la entrega debido a su tamaño, por lo que se proporciona un enlace para su descarga.
- El script `separar_imagenes.py` permite reconstruir automáticamente las carpetas `train`, `validation` y `test`, garantizando la estructura requerida por ambos notebooks.
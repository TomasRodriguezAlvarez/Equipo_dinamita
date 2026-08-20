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

Además del clasificador de 6 clases, el proyecto creció con **dos variantes más
del target** y una **app móvil** que corre los tres modelos en el teléfono:

| Modelo | Clases | Dónde está |
|---|---|---|
| MobileNetV3 (material) | 6: cardboard, glass, metal, paper, plastic, trash | `PyTorch/`, `tensorflow/` |
| Jerárquico (contenedor) | 4: amarillo, azul, gris, verde | `PyTorch/jerarquico/` |
| Binario | 2: basura, reciclable (con umbral calibrado) | `PyTorch/binario/` |

El detalle de entrenamiento, métricas y decisiones de la rama PyTorch está en
[`PyTorch/CONTEXT.md`](PyTorch/CONTEXT.md). La app móvil está en
[`trashnet-tester/`](trashnet-tester/README.md).

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

# Exportación a ONNX

Los tres modelos PyTorch (las variantes `_optuna`, que son las de mejor
desempeño) se exportan a ONNX para poder correrlos fuera de Python:

```bash
conda activate frameworks-ia
pip install onnx onnxruntime onnxscript      # solo la primera vez
python scripts/export_onnx.py                # genera onnx/
cd scripts && python verify_onnx.py          # contrasta ONNX vs PyTorch
```

`export_onnx.py` deja en `onnx/` seis archivos: un `.onnx` y un `.labels.json`
por modelo. El `.labels.json` lleva las clases, el tamaño de entrada, la
normalización y —en el binario— el umbral calibrado. **La app lee todo de ahí, no
hardcodea ninguno de esos valores**, así que si se reentrena un modelo alcanza con
regenerar los archivos.

`verify_onnx.py` compara los logits del ONNX contra los del PyTorch original sobre
`imagenes_a_test/`. Tiene que dar `[OK]` en todas las líneas.

---

# App móvil (Ionic + OpenCV.js + ONNX Runtime Web)

`trashnet-tester/` es una app Ionic/Angular que corre los **tres modelos
on-device**, sin llamadas de red: se saca o se elige una foto y se obtiene la
clase predicha con su confianza.

```bash
cd trashnet-tester
npm install
npm start                # navegador, http://localhost:8100

source android-env.sh    # JDK y SDK están en $HOME, no en el sistema
npm run android:apk      # APK de debug (~34 MB)
```

Puntos a tener en cuenta:

- **La regla de decisión no es la misma para los tres modelos.** Los dos
  multiclase usan `argmax`; el binario usa el **umbral calibrado 0.837**, que es
  lo que sube su F1 de 0.655 a 0.818 (ver `PyTorch/CONTEXT.md` §8).
- **El preprocesamiento replica exactamente el `eval_tf` del entrenamiento**, y
  lo hace con **OpenCV.js**. El `cv.resize` directo no sirve: OpenCV no tiene un
  bilineal con antialias como el de torchvision, y con `INTER_AREA` el modelo
  jerárquico cambia su predicción sobre `test 1.jpeg`. Como el resample de
  Pillow es separable y lineal, sus matrices de coeficientes se multiplican con
  `cv.gemm`: el resize lo ejecuta OpenCV sin perder la equivalencia. La app trae
  **tres pipelines seleccionables** para poder comparar.
- **Hay un autotest** que corre `imagenes_a_test/` y compara contra las
  predicciones de los modelos PyTorch originales (`npm run autotest`). Da **6/6**
  con los dos pipelines fieles al entrenamiento y **5/6** con `cv.resize`,
  verificado en navegador y en teléfono Android.

Todo el detalle —incluidos los desvíos respecto del plan original y los errores
que solo aparecen probando en un navegador real— está en
[`trashnet-tester/README.md`](trashnet-tester/README.md). El plan que se siguió
para construirla está en [`PLAN_APP_IONIC.md`](PLAN_APP_IONIC.md).

> **Ojo con las predicciones de la app.** Está medido que estos modelos no
> generalizan fuera del estudio fotográfico de TrashNet: el recall del binario
> cae de 0.818 a 0.119 con fotos reales de celular. Una predicción mala con una
> foto propia es el comportamiento esperado, no necesariamente un bug. El
> autotest sirve justo para distinguir los dos casos: si da 6/6, el pipeline
> está bien y lo que falla es el modelo.

---

# Notas

- Ambos modelos fueron entrenados utilizando el mismo dataset.
- El dataset original no se incluye directamente en la entrega debido a su tamaño, por lo que se proporciona un enlace para su descarga.
- El script `separar_imagenes.py` permite reconstruir automáticamente las carpetas `train`, `validation` y `test`, garantizando la estructura requerida por ambos notebooks.
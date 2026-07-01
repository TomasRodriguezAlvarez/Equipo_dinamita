# Equipo Dinamita — TrashNet (PyTorch)

Clasificador de imágenes de residuos en **6 categorías**: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

---

## Estructura del repositorio

```
Equipo_dinamita/
├── dataset/
│   ├── train/         # imágenes de entrenamiento (1766 imgs, 6 clases)
│   ├── validation/    # imágenes de validación (377 imgs)
│   ├── test/          # imágenes de prueba (384 imgs)
│   └── dataset-resized/
├── PyTorch/
│   ├── TrashNet_PyTorch.ipynb        # notebook principal (entrenamiento + evaluación)
│   ├── Lab1_Tensores_Autograd.ipynb  # laboratorio introductorio
│   └── modelos/
│       └── trashnet_pytorch.pt       # modelo entrenado
├── imagenes_a_test/   # imágenes sueltas para probar el modelo
└── README.md
```

Cada subcarpeta de `train/`, `validation/` y `test/` debe tener una carpeta por clase con sus imágenes dentro:

```
dataset/train/cardboard/*.jpg
dataset/train/glass/*.jpg
...
```

---

## Requisitos previos

- **Python 3.11**
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o Anaconda (recomendado), o `venv`.
- No es obligatorio tener GPU: el proyecto funciona en **CPU** (más lento) o en **GPU NVIDIA con CUDA** (más rápido).

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/TomasRodriguezAlvarez/Equipo_dinamita.git
cd Equipo_dinamita
```

### 2. Crear el entorno

Con **conda** (recomendado):

```bash
conda create -n frameworks-ia python=3.11 -y
conda activate frameworks-ia
```

O con **venv** (alternativa sin conda):

```bash
python3.11 -m venv frameworks-ia
source frameworks-ia/bin/activate      # Windows: frameworks-ia\Scripts\activate
```

### 3. Instalar PyTorch

Elige **una** de las dos opciones según tu hardware:

**Opción A — Solo CPU** (funciona en cualquier computador, sin tarjeta gráfica NVIDIA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Opción B — GPU NVIDIA con CUDA 12.x** (entrenamiento mucho más rápido):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> Para otras versiones consulta el selector oficial: https://pytorch.org/get-started/locally/

### 4. Instalar el resto de dependencias

```bash
pip install numpy matplotlib jupyter ipykernel pillow
```

### 5. Registrar el kernel de Jupyter

```bash
python -m ipykernel install --user --name frameworks-ia --display-name "Python (frameworks-ia)"
```

---

## Verificar la instalación

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU disponible:', torch.cuda.is_available())"
```

- Si tienes GPU y CUDA correctamente instalados → `GPU disponible: True`.
- Si usas CPU → `GPU disponible: False` (es normal, el modelo igual funciona).

El código detecta el dispositivo automáticamente:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Entrenar y evaluar el modelo

1. Inicia Jupyter:

   ```bash
   cd PyTorch
   jupyter notebook TrashNet_PyTorch.ipynb
   ```

2. Selecciona el kernel **"Python (frameworks-ia)"**.
3. Ejecuta todas las celdas (`Run All`).

El notebook realiza:

- Carga del dataset con `ImageFolder` y *data augmentation*.
- Definición de la CNN (`Conv 16 → 32 → 64` + `Dense 128` + `Dropout 0.3` → 6 clases).
- Entrenamiento (5 épocas por defecto) y gráficos de *accuracy* / *loss*.
- Evaluación sobre el conjunto de test.
- Guardado del modelo en `modelos/trashnet_pytorch.pt`.

> **Nota sobre el rendimiento:** en GPU, 5 épocas tardan ~2 minutos. En CPU pueden tardar bastante más (10–30 min según el equipo). Si vas a entrenar en CPU y quieres que sea más rápido, reduce `EPOCHS` o `BATCH_SIZE` en el notebook.

### Configuración principal (editable en el notebook)

| Parámetro    | Valor por defecto |
|--------------|-------------------|
| `IMG_SIZE`   | `(150, 150)`      |
| `BATCH_SIZE` | `16`              |
| `EPOCHS`     | `5`               |
| Optimizador  | `Adam`            |
| Pérdida      | `CrossEntropyLoss`|

---

## Predecir sobre una imagen nueva

La última celda del notebook incluye la función `predecir()`. Ejemplo mínimo para usar el modelo guardado:

```python
import torch
from PIL import Image
from torchvision import transforms

ckpt = torch.load('modelos/trashnet_pytorch.pt', map_location='cpu')
# ... reconstruir el modelo TrashNet y cargar ckpt['model_state_dict'] ...

eval_tf = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])
img = Image.open('../imagenes_a_test/mi_imagen.jpg').convert('RGB')
# x = eval_tf(img).unsqueeze(0)  ->  model(x)  ->  softmax  ->  clase
```
---


# TensorFlow (Transfer Learning con MobileNetV3Small)

## Modelo implementado

Además del modelo desarrollado en PyTorch, se implementó una segunda
solución utilizando **TensorFlow/Keras** mediante **Transfer Learning**
con **MobileNetV3Small** preentrenado en ImageNet.

El desarrollo se realizó en:

`tensorflow/TrashNet_MobileNetV3.ipynb`

------------------------------------------------------------------------

## Instalación

``` bash
conda activate frameworks-ia
```

Si el entorno no existe:

``` bash
conda create -n frameworks-ia python=3.11 -y
conda activate frameworks-ia
pip install tensorflow numpy matplotlib pillow jupyter ipykernel
python -m ipykernel install --user --name frameworks-ia --display-name "Python (frameworks-ia)"
```

------------------------------------------------------------------------

## Ejecución

``` bash
cd tensorflow
jupyter notebook
```

Abrir:

`TrashNet_MobileNetV3.ipynb`

------------------------------------------------------------------------

## Flujo del notebook

1.  Imports.
2.  Configuración.
3.  Carga del dataset.
4.  Revisión de imágenes por clase.
5.  Cálculo de Class Weights.
6.  Data Augmentation.
7.  MobileNetV3Small.
8.  Entrenamiento (Fase 1).
9.  Evaluación.
10. Gráficos de Accuracy y Loss.
11. Guardado del modelo.

------------------------------------------------------------------------

## Configuración

  Parámetro                                Valor
  ------------------- --------------------------
  IMG_SIZE                               160x160
  BATCH_SIZE                                   4
  Optimizer                                 Adam
  Learning Rate                            0.001
  Loss                  categorical_crossentropy
  Data Augmentation                           Sí
  Class Weights                               Sí

------------------------------------------------------------------------

## Resultados

  Modelo               Test Accuracy
  ------------------ ---------------
  CNN TensorFlow              57.8 %
  MobileNetV3Small            81.0 %

La utilización de Transfer Learning permitió mejorar el rendimiento en
aproximadamente 23 puntos porcentuales.

------------------------------------------------------------------------

## Modelo generado

`tensorflow/modelo_guardado/trashnet_mobilenetv3_small_fase1.keras`

Carga del modelo:

``` python
import tensorflow as tf

modelo = tf.keras.models.load_model(
    "modelo_guardado/trashnet_mobilenetv3_small_fase1.keras"
)
```

------------------------------------------------------------------------

## Limitaciones

El entrenamiento se realizó en una instancia AWS Academy con memoria
limitada (\~1 GB RAM).

Para garantizar estabilidad fue necesario:

-   IMG_SIZE = 160x160.
-   BATCH_SIZE = 4.
-   No utilizar prefetch(AUTOTUNE).
-   No realizar Fine-Tuning completo por restricciones de memoria.

------------------------------------------------------------------------
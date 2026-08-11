# dataset_gd — clase `trash` del Garbage Dataset (GD)

**Esta carpeta NO es parte de TrashNet y no se mezcla con `dataset/`.** Son 453 imágenes
de un dataset externo, usadas **solo para evaluación**, nunca para entrenar.

## Qué contiene

```
dataset_gd/
└── trash/     453 imágenes .jpg/.jpeg (24 MB)
```

Es la clase `trash` (residuo de rechazo, no reciclable) del *Garbage Dataset*, que en
total tiene 12.259 imágenes en 10 clases. Aquí solo se copió esa clase.

## Para qué está

Evaluación **fuera de distribución** del modelo binario
(`PyTorch/binario/`). El modelo se entrena con las 95 imágenes de `trash` de TrashNet y
se evalúa además contra estas 453, que son 20× las 22 del test de TrashNet.

La pregunta que responde: *un modelo entrenado con fotos de estudio sobre fondo blanco,
¿sirve con fotos reales?*

## Por qué la carpeta `original` y no las `standardized_*`

El dataset se descarga con tres variantes: `original`, `standardized_256` y
`standardized_384`. Se copió **`original`** a propósito.

Las dos `standardized_*` hacen las imágenes cuadradas con **padding letterbox** de color
gris `(114,114,114)` — y en algunas, blanco `(255,255,255)`. Las imágenes de TrashNet
nunca tienen esas barras, así que evaluar con ellas metería un artefacto visual presente
**solo** en la clase de basura: sería imposible saber si el modelo falla por el objeto o
por el relleno.

Con `original`, ambos conjuntos reciben el mismo trato: aspect ratio nativo →
`transforms.Resize((224,224))`.

## Advertencia sobre la resolución

Las imágenes originales son heterogéneas: ancho 51–4032 px (mediana 301), alto 132–3024
(mediana 269). **136 de 453 (30%) tienen algún lado por debajo de 224 px** y se escalan
hacia arriba, así que se ven borrosas.

Por eso el notebook reporta los resultados **desglosados por resolución** (≥224 vs <224):
sirve para separar "el modelo no generaliza" de "la imagen no se ve".

## Procedencia y licencia

- Fuente: <https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2>
- Paper: *The Garbage Dataset (GD): A Multi-Class Image Benchmark for Automated Waste
  Segregation* — <https://arxiv.org/abs/2602.10500>
- Autor: Suman Kunwar (`sumn2u`)
- **Licencia: CC BY 4.0** — se puede usar y redistribuir citando la fuente.
  <https://creativecommons.org/licenses/by/4.0/>
- Copiado el 2026-08-10 desde la carpeta `original/trash` del zip de Kaggle, sin
  modificar ningún archivo (verificado por hash MD5).

Si estas imágenes se usan en el informe, hay que citar el dataset.

## Verificación hecha al copiar

- 453 de 453 archivos, hash MD5 idéntico al origen
- 0 imágenes ilegibles, 0 duplicados exactos
- las 453 en modo RGB

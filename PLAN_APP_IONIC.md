# Instrucciones para Claude: app Ionic para probar los modelos TrashNet

Este documento es autocontenido: asume que quien lo ejecuta (otra sesión de Claude,
en Ubuntu o en Windows) no tiene contexto previo de esta conversación. Todas las
decisiones de diseño ya están tomadas — no se necesita preguntarle al usuario nada de
lo que aparece acá, salvo lo listado en "Cuándo detenerse y preguntar" al final.

## Contexto del repo

Repo: proyecto de clasificación de residuos (dataset TrashNet, 6 clases:
cardboard/glass/metal/paper/plastic/trash), con modelos entrenados en 2 frameworks
(PyTorch y TensorFlow). El detalle completo de entrenamiento, arquitecturas y
métricas está en [`PyTorch/CONTEXT.md`](PyTorch/CONTEXT.md) — leerlo si hace falta
más contexto de por qué existen 3 variantes de modelo.

**Objetivo final:** una app Ionic que tome/seleccione una foto y corra inferencia
on-device con los 3 modelos siguientes, exportados a ONNX:

| Modelo | Clases | Checkpoint PyTorch origen |
|---|---|---|
| `trashnet_mobilenetv3` | cardboard, glass, metal, paper, plastic, trash (6) | `PyTorch/modelos/trashnet_mobilenetv3_optuna.pt` |
| `trashnet_jerarquico` | amarillo, azul, gris, verde (4, agrupación por contenedor) | `PyTorch/jerarquico/modelos/trashnet_jerarquico_optuna.pt` |
| `trashnet_binario` | basura, reciclable (2, **con umbral calibrado 0.837, no argmax**) | `PyTorch/binario/modelos/trashnet_binario_optuna.pt` |

Todos son MobileNetV3Small, entrada 224×224, normalización ImageNet
(`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`). Se usan las variantes
`_optuna` porque son las de mejor desempeño en los 3 casos.

**Advertencia a tener presente y comunicar al usuario si corresponde:** está medido
que estos modelos generalizan mal a fotos reales fuera del estudio fotográfico de
TrashNet (recall cae de 0.818 a 0.119 en el dataset externo `dataset_gd/`, ver
`PyTorch/CONTEXT.md` §8). Si al probar la app con fotos de celular las predicciones
son malas, no asumir que es un bug de la app antes de descartar esto.

**Estado al momento de escribir esto:** no existe nada de ONNX, Ionic ni Capacitor en
el repo. Ya existen y están listos para usar:

- [`scripts/export_onnx.py`](scripts/export_onnx.py) — exporta los 3 checkpoints a ONNX.
- [`scripts/verify_onnx.py`](scripts/verify_onnx.py) — verifica cada export contra el
  PyTorch original.

Revisar el estado real del repo (`git status`, `ls onnx/` si existe, `ls` en la raíz
buscando una carpeta de proyecto Ionic) antes de asumir en qué fase se está — este
documento puede ejecutarse en más de una sesión.

---

## Fase 1 — Exportar los 3 modelos a ONNX

**Dónde:** entorno Ubuntu, env conda `frameworks-ia` (documentado en
`PyTorch/CONTEXT.md` §2). Ese env ya tiene `torch 2.12.1+cu126` y
`torchvision 0.27.1+cu126`, las versiones exactas usadas para entrenar. No se
necesita GPU para exportar: se hace en CPU (`map_location="cpu"`).

Si esta fase se ejecuta en una máquina que no es la Ubuntu de entrenamiento (sin el
env `frameworks-ia`), detenerse y preguntar al usuario cómo proceder — no hay que
reinstalar todo un env nuevo sin confirmar.

1. Actualizar el checkout:
   ```bash
   git pull
   ```
2. Activar el env y confirmar que es el correcto:
   ```bash
   conda activate frameworks-ia
   python -c "import sys; print(sys.executable)"
   ```
   Debe imprimir una ruta terminada en `.../envs/frameworks-ia/bin/python`. Si no,
   detenerse — no continuar en el env equivocado.
3. Instalar las dos dependencias que faltan (no chocan con lo ya instalado):
   ```bash
   pip install onnx onnxruntime
   ```
4. Exportar, desde la raíz del repo:
   ```bash
   python scripts/export_onnx.py
   ```
   Resultado esperado: carpeta nueva `onnx/` con 6 archivos (3 `.onnx` + 3
   `.labels.json`, uno por modelo). Cada `.labels.json` trae clases, input size,
   mean/std, y en el binario el umbral 0.837 y el índice de la clase `basura`.
5. Verificar:
   ```bash
   cd scripts
   python verify_onnx.py
   ```
   Criterio de éxito: **todas** las líneas impresas dicen `[OK]`, con `diff_max` del
   orden de `1e-05` o menor. Si aparece algún `[MISMATCH]`: no avanzar a la Fase 2.
   Investigar primero (versión de opset en `export_onnx.py`, versión de
   torch/torchvision) y volver a exportar.
6. Publicar el resultado para que quede disponible en el otro entorno (Windows u
   otro checkout):
   ```bash
   git add onnx/
   git commit -m "Agrega modelos exportados a ONNX"
   git push
   ```
   Si el usuario no quiere versionar binarios en git, preguntar antes de hacer commit
   y ofrecer copiar la carpeta `onnx/` por otro medio en su lugar.

**Criterio de fase completa:** `onnx/` existe con 6 archivos y `verify_onnx.py` dio
`[OK]` en todo.

---

## Fase 2 — Setup del proyecto Ionic

**Dónde:** cualquier máquina con Node disponible (no requiere el env de Python).

1. Confirmar que Node/npm están disponibles (`node -v`, `npm -v`). Si no, detenerse y
   pedir al usuario que los instale — no instalar Node sin permiso explícito.
2. Crear el proyecto:
   ```bash
   npm install -g @ionic/cli
   ionic start trashnet-tester blank --type=angular --capacitor
   cd trashnet-tester
   npm install onnxruntime-web
   npm install @capacitor/camera
   ```
   Decisión ya tomada: Angular (`--type=angular`), porque es el default de Ionic y el
   mejor soportado. No preguntar por esto salvo que el usuario ya haya indicado otra
   preferencia (React/Vue) en la conversación donde se invoque este plan.
3. Copiar los 6 archivos de `onnx/` (generados en la Fase 1) a:
   ```
   trashnet-tester/src/assets/models/
   ```

**Criterio de fase completa:** el proyecto Ionic compila (`ionic serve` levanta sin
error) y los 6 archivos de modelos están en `src/assets/models/`.

---

## Fase 3 — Preprocesamiento de imagen (OpenCV.js)

Instalar `opencv.js` (build WASM) como asset local — sin llamadas a red en runtime.

El pipeline **tiene que replicar exactamente** el `eval_tf` usado en entrenamiento
(`transforms.Resize((224,224))` + `ToTensor()` + `Normalize(mean, std)`):

1. Capturar/seleccionar imagen con `@capacitor/camera`.
2. Cargar en `cv.Mat`, convertir a RGB si hace falta.
3. `cv.resize()` **directo a 224×224, sin crop, sin mantener aspect ratio** — así se
   entrenó, y desviarse de esto introduce un mismatch de dominio adicional al ya
   conocido (ver advertencia arriba). Punto no negociable: no agregar center-crop
   "para verse mejor".
4. Normalizar por canal: `(pixel/255 - mean) / std`, usando los valores de
   `mean`/`std` del `labels.json` correspondiente (no hardcodear, leerlos del
   archivo).
5. Reordenar a layout **CHW** (no HWC) y armar el `Float32Array` que espera
   `onnxruntime-web`.

**Criterio de fase completa:** función de preprocesamiento que, dada una imagen y un
`labels.json`, devuelve un tensor listo para `onnxruntime-web`.

---

## Fase 4 — Inferencia con onnxruntime-web

1. Cargar la sesión ONNX del modelo seleccionado:
   `ort.InferenceSession.create('assets/models/trashnet_X.onnx')`.
2. Correr `session.run({ input: tensor })` → logits.
3. Aplicar `softmax` a los logits para obtener probabilidades.
4. Decidir la clase — **la lógica difiere por modelo, no usar la misma regla para los
   3**:
   - `trashnet_mobilenetv3` y `trashnet_jerarquico`: `argmax` sobre las
     probabilidades.
   - `trashnet_binario`: **no usar argmax.** Es `basura` si
     `prob[idx_basura] >= umbral_basura` (0.837, leer del `labels.json`, no
     hardcodear), `reciclable` en caso contrario.
5. UI: selector de los 3 modelos, mostrar clase predicha + confianza.

**Criterio de fase completa:** la app predice sobre una imagen de prueba y muestra
resultado en pantalla, con la lógica de decisión correcta por modelo.

---

## Fase 5 — Validación

1. Cargar en la app las imágenes de `imagenes_a_test/` y confirmar que la predicción
   coincide con lo que reportó `verify_onnx.py` en la Fase 1 para esas mismas
   imágenes (si están entre las probadas) o con lo que reportan los notebooks.
2. Probar con fotos reales tomadas con la cámara del celular. Documentar el
   comportamiento sin asumir que una mala predicción es un bug — comparar contra la
   advertencia de generalización del modelo binario (recall 0.818 → 0.119 medido en
   `dataset_gd/`).
3. Probar en dispositivo real (Android/iOS), no solo en navegador de escritorio:
   confirmar que `onnxruntime-web` (WASM) y `opencv.js` cargan bien dentro del
   WebView de Capacitor. Si falla ahí y no en desktop, revisar CSP / tipo MIME de los
   `.wasm` en `capacitor.config.ts` antes de reportar como bloqueante.

**Criterio de fase completa:** los 3 modelos predicen correctamente sobre
`imagenes_a_test/` en un dispositivo real (no solo en el navegador de desarrollo).

---

## Riesgos conocidos (no bloquean el plan, pero conviene anticiparlos)

- WASM en WebView de Capacitor en dispositivo real puede necesitar configuración
  extra de CSP en `capacitor.config.ts` que no aparece al probar solo en desktop.
- El tamaño combinado de los 3 `.onnx` (~15-20 MB) es razonable para bundlear en la
  app. Si se necesita más liviano, cuantizar a INT8 con
  `onnxruntime.quantization` en la Fase 1 — opcional, no hacerlo salvo que el tamaño
  sea un problema real.

## Cuándo detenerse y preguntar al usuario

- Si la Fase 1 se va a correr en una máquina sin el env `frameworks-ia` ya armado.
- Si `verify_onnx.py` da `[MISMATCH]` y la causa no es evidente tras revisar
  versiones de opset/torchvision.
- Antes de hacer `git push` de binarios grandes (`onnx/`) si no quedó claro que el
  usuario quiere versionarlos.
- Si se quiere usar un framework distinto de Angular para el proyecto Ionic (React,
  Vue) y no hay indicación previa de cuál prefiere el equipo.
- Antes de instalar herramientas nuevas a nivel de sistema (Node, conda, Xcode/Android
  SDK para build nativo) que no estén ya presentes.

## Checklist

- [x] Fase 1: `onnx/` generado y verificado (`[OK]` en todo `verify_onnx.py`,
      `diff_max` ≤ 2.3e-05).
- [x] Fase 1: `onnx/` disponible en el entorno donde se desarrolla la app.
      **Sin commitear** — el usuario hace los commits.
- [x] Fase 2: proyecto Ionic creado en `trashnet-tester/`, dependencias
      instaladas, modelos copiados a `src/assets/models/`.
- [x] Fase 3: pipeline de preprocesamiento (resize 224×224 sin crop + normalize
      con mean/std leídos del `labels.json`), **con OpenCV.js** — pero el resize
      no es `cv.resize`, ver más abajo.
- [x] Fase 4: inferencia con `onnxruntime-web` + lógica de decisión correcta por
      modelo (argmax vs. umbral calibrado del binario).
- [x] Fase 5: validado contra `imagenes_a_test/`, **6/6 coincide con PyTorch**,
      en Firefox headless sobre la app corriendo (`npm run autotest` da lo mismo
      sin navegador, con un decodificador JPEG algo distinto).
- [x] Fase 5: proyecto Android agregado y APK de debug compilando
      (`npm run android:apk`). JDK 21 y Android SDK instalados en `$HOME`.
- [x] Fase 5: APK instalado en un teléfono real y probado con fotos de la
      cámara (2026-08-20). El autotest **dentro de la app** llegó a correr ahí,
      pero después sus botones se retiraron de la pantalla junto con otros
      textos técnicos; en escritorio queda `npm run autotest`. Ver "La pantalla"
      en `trashnet-tester/README.md`.

## Desvíos respecto de lo que decía este plan

Documentados en detalle en `trashnet-tester/README.md`:

1. **El resize no es un `cv.resize`.** OpenCV no tiene bilineal con antialias,
   que es lo que hace `transforms.Resize((224,224))` sobre una imagen PIL. Con
   `cv.INTER_AREA` (la mejor opción directa de OpenCV) el modelo jerárquico
   predice `amarillo` en vez de `verde` sobre `test 1.jpeg`. El resample de
   Pillow es separable y lineal, así que se le pasan sus matrices de
   coeficientes a `cv.gemm`: el resize lo ejecuta OpenCV y el resultado coincide
   con torchvision (máx 1/255 sobre el 0.08% de los píxeles, 6/6). La app trae
   los tres pipelines seleccionables para poder comparar en vivo.
2. **Opset 18 en vez de 17** en `scripts/export_onnx.py`: pedir 17 hace fallar al
   version converter de `onnx`.
3. **`scripts/export_onnx.py` estaba roto**: asumía que los `.pt` eran un
   `state_dict` pelado, pero son un dict con `state_dict` + metadata.
4. Se agregaron `@ionic/pwa-elements` (cámara en navegador de escritorio) y
   `scripts/generar_autotest.py` (referencia del autotest).
5. Dos configuraciones de `onnxruntime-web` que solo se manifiestan en el
   navegador (`wasmPaths` tiene que ser URL absoluta; el bundle por defecto pide
   el `.wasm` de JSEP): detalle en `trashnet-tester/README.md`. Anticipan el
   riesgo de WASM en WebView que este plan ya listaba.
6. **OpenCV.js se carga como asset, no como import** — son 13 MB y no tienen por
   qué descargarse si el usuario se queda en el pipeline sin OpenCV. El APK pasó
   de ~29 MB a ~34 MB.
7. **La UI quedó más chica que la que pedía la Fase 4/5**: se retiraron el
   autotest, la advertencia de fotos reales y la línea con la regla de decisión.
   La lógica no cambió — el binario sigue usando el umbral 0.837 —, solo dejó de
   mostrarse en pantalla.

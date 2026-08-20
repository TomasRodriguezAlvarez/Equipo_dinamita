# trashnet-tester

App Ionic/Angular que corre los 3 modelos TrashNet on-device con ONNX Runtime
Web y preprocesa las imágenes con OpenCV.js. Implementa el `PLAN_APP_IONIC.md`
de la raíz del repo.

**Estado:** funcionando. El autotest da 6/6 contra las predicciones de los
modelos PyTorch originales con los dos pipelines fieles al entrenamiento,
verificado en navegador de escritorio y en un teléfono Android real
(2026-08-20, con el APK instalado por `adb`).

## Correr

```bash
npm install
npm start            # http://localhost:8100
```

Los modelos se cargan de `src/assets/models/` (ya están commiteados ahí). Si se
reentrena alguno, regenerarlos desde la raíz del repo con el env de Python:

```bash
conda activate frameworks-ia
python scripts/export_onnx.py
cp onnx/* trashnet-tester/src/assets/models/
```

## La pantalla

De arriba abajo: **Modelo** (cuál de los tres), **Preprocesamiento** (cuál de
los tres pipelines), los botones **Cámara / Galería**, la vista previa y las
tarjetas de resultado. Los botones de predecir viven en un **pie fijo** que
aparece con la foto cargada: antes iban en el flujo del contenido y, con la
vista previa ocupando 40% del alto, el segundo quedaba fuera de la pantalla del
teléfono.

Tres cosas se sacaron a propósito, para que la app se pueda mostrar sin explicar
el proyecto entero:

| Qué se sacó | Dónde estaba | Cómo volver |
|---|---|---|
| Tarjeta de autotest con sus dos botones | al final del contenido | reponer la tarjeta en `home.page.html`; `autotest.service.ts` sigue ahí |
| Advertencia sobre fotos reales | tarjeta amarilla bajo los resultados | reponer la tarjeta; el aviso sigue al final de este README |
| Regla de decisión (`umbral calibrado: …` / `argmax …`) | bajo el porcentaje de cada resultado | reponer `<p class="regla">`; el campo `regla` se sigue calculando |

**Sacar el texto de la regla no cambió la lógica**: el binario sigue decidiendo
por el umbral 0.837 y no por `argmax`. Solo dejó de contarlo en pantalla.

Los nombres de los pipelines en la UI son deliberadamente poco técnicos —
"OpenCV — exacto", no "cv.gemm con coeficientes de Pillow"—. La equivalencia con
los identificadores del código está en la tabla de la sección siguiente.

## Preprocesamiento: los tres pipelines

La app trae **tres implementaciones del preprocesamiento**, seleccionables desde
la pantalla principal. Las tres hacen lo mismo (resize a 224×224 sin crop → RGB
en [0,1] → normalize con mean/std de ImageNet → layout CHW); lo que cambia es
quién hace el resize y con qué filtro.

| En pantalla | Modo | Qué usa | Diferencia vs PIL | Autotest | Tiempo* |
|---|---|---|---|---|---|
| Sin OpenCV — exacto | `pillow` | `resample.ts`, JavaScript puro | referencia | **6/6** | 21-33 ms |
| OpenCV — exacto | `opencv` | OpenCV.js: `cvtColor` + `gemm` + `convertTo` | máx **1/255** sobre el 0.08% de los píxeles | **6/6** | 614-745 ms |
| OpenCV — aproximado | `opencv-area` | OpenCV.js: `cv.resize(INTER_AREA)` | MAE 1.0, máx **27/255** sobre el 51% | **5/6** | 12-16 ms |

Los nombres de la primera columna son los del desplegable; los de la segunda,
los identificadores en el código y en la salida de `npm run autotest`.

\* medido con `npm run preproceso` sobre `imagenes_a_test/` (900×1600 y
1200×1600) en el escritorio. En el teléfono son bastante más.

### Por qué el resize con OpenCV no es un `cv.resize`

Los modelos se entrenaron con `transforms.Resize((224,224))` de torchvision, que
sobre una imagen PIL aplica un bilineal **con antialias**: el soporte del filtro
se ensancha con el factor de reducción, así que al achicar 5× promedia los ~11
píxeles que caen dentro, no los 2×2 vecinos.

OpenCV no tiene esa variante. `INTER_LINEAR` no filtra e `INTER_AREA` es un
filtro de caja, no triangular. La consecuencia no es cosmética: con `INTER_AREA`
el modelo jerárquico predice `amarillo` en vez de `verde` sobre `test 1.jpeg`, y
el autotest baja a 5/6.

La salida es que el resample de Pillow **es separable y lineal**: cada píxel de
salida es una combinación lineal de los de entrada, primero por filas y después
por columnas. Eso se escribe como dos productos de matrices,

```
horizontal:  (H × Went) · (Went × Wsal)  →  H    × Wsal
vertical:    (Hsal × H) · (H    × Wsal)  →  Hsal × Wsal
```

donde las matrices son los pesos triangulares que calcula
`precalcularCoeficientes()` en `resample.ts`. Los dos productos los hace
`cv.gemm`, con un redondeo a uint8 en medio (Pillow también lo hace, y omitirlo
mueve el resultado). Así el resize lo ejecuta OpenCV y aun así el tensor coincide
con el de torchvision.

**El precio es el tiempo:** ~30× más lento que el bucle en JavaScript. La matriz
de coeficientes es densa (1200×224) pero el filtro es ralo — solo ~11 pesos por
columna son distintos de cero —, así que `gemm` hace unas 100 veces más
multiplicaciones de las necesarias. Es un intercambio consciente: el modo
`pillow` está ahí para quien prefiera la velocidad. Los tiempos se muestran en
pantalla junto a cada predicción.

`opencv-area` no está para usarse, está para **poder mostrar la diferencia**:
cambiando de pipeline y prediciendo sobre `test 1.jpeg` se ve el cambio de
`verde` a `amarillo`. La comparación completa contra PyTorch está en
`npm run autotest`.

## Autotest

Compara las predicciones de la app contra las de los modelos PyTorch originales
sobre las imágenes de `imagenes_a_test/`. Hay dos formas de correrlo:

- **En Node**, sin navegador, corriendo los tres pipelines:

  ```bash
  npm run autotest
  ```

  El exit code ignora a propósito el fallo de `opencv-area`: ese modo está para
  mostrar la diferencia, no para pasar.

`autotest.service.ts` corre lo mismo **dentro de la app**, que era la forma de
validar en el dispositivo real. Los botones que lo lanzaban se sacaron de la
pantalla para la presentación, así que hoy el servicio existe pero no lo llama
nadie: para volver a montarlo alcanza con reponer la tarjeta en
`home.page.html`.

Y para comparar los pipelines a nivel de píxel (sin ejecutar los modelos):

```bash
npm run preproceso
```

La referencia vive en `src/assets/pruebas/esperado.json` y se regenera con:

```bash
conda activate frameworks-ia
python scripts/generar_autotest.py
```

## Android

En la máquina donde se armó esto, el JDK 21 y el Android SDK quedaron instalados
**en `$HOME`** y no en el sistema, porque no había `sudo` sin contraseña:

- `~/opt/jdk-21.0.12+8`
- `~/Android/Sdk` (platform-tools, platforms 35 y 36, build-tools 35 y 36)

`android-env.sh` exporta esas rutas. **Son las de esa instalación**: si clonás el
repo en otra máquina, o instalás Android Studio (que trae su propio JDK y SDK),
editá ese archivo con tus rutas o exportá `JAVA_HOME` y `ANDROID_HOME` a mano. No
hace falta `android/local.properties` — Gradle toma el SDK de `ANDROID_HOME`.

Antes de cualquier comando de Android:

```bash
source android-env.sh
```

Compilar el APK de debug, desde `trashnet-tester/`:

```bash
npm run android:apk
```

Deja el APK en `android/app/build/outputs/apk/debug/app-debug.apk` (~34 MB).
Ese script hace las tres cosas que hay que hacer siempre en ese orden: build de
Angular a `www/`, `cap sync` para copiar `www/` dentro del proyecto nativo, y
`gradlew assembleDebug`. **Si editás código web y solo corrés `gradlew`, el APK
sale con la versión vieja**: falta el `cap sync`.

Instalar en un teléfono conectado por USB con depuración activada:

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

Para ver los errores de la WebView mientras corre (es donde aparecen los
problemas de WASM que no se ven en desktop):

```bash
adb logcat | grep -i -E "chromium|capacitor|onnx"
```

Al probar en el teléfono conviene empezar por **Galería** con una imagen
conocida y recién después ir a **Cámara** con fotos reales. Para comprobar que
el pipeline entero anda dentro de la WebView hace falta el autotest en la app,
que hoy no tiene botón (ver la sección anterior).

## Qué NO está versionado

Todo lo que falta en un clon limpio se regenera; no hay nada que haya que pasar
por otro medio:

| Qué | Se regenera con |
|---|---|
| `node_modules/` | `npm install` |
| `www/` | `ng build` (o `npm run android:apk`) |
| `.angular/` (caché) | solo |
| `android/build/`, `android/app/build/` | `gradlew` |
| `android/app/src/main/assets/public/` | `npx cap sync` |
| `android/local.properties` | opcional, ver arriba |
| `pruebas/autotest-node.cjs` | `npm run autotest` |

El **APK tampoco se versiona** (`*.apk` está en el `.gitignore` de Android, y
además vive dentro de `build/`). Si hace falta repartirlo sin que el otro tenga
que compilar, lo razonable es adjuntarlo a un Release de GitHub y no meter un
binario de 34 MB en el historial del repo.

## Decisiones que se desvían del plan original

**1. El resize con OpenCV no es un `cv.resize`.** El plan pedía
`cv.resize()` a secas, pero OpenCV no tiene un bilineal con antialias, que es lo
que aplica `transforms.Resize((224,224))` de torchvision sobre una imagen PIL.
Medido sobre `imagenes_a_test/`:

| Resize | Diferencia vs PIL | Predicciones que coinciden |
|---|---|---|
| `cv.INTER_LINEAR` | MAE 3.0–3.6 | — |
| `cv.INTER_AREA` (modo `opencv-area`) | MAE 1.0, máx 27/255 | 5/6 (el jerárquico da `amarillo` en vez de `verde`) |
| `resample.ts` (modo `pillow`) | MAE 0.0008, máx 1/255 | **6/6** |
| `cv.gemm` con los pesos de PIL (modo `opencv`) | MAE 0.0008, máx 1/255 | **6/6** |

La solución fue aprovechar que el resample de Pillow es separable y lineal, y
darle los productos de matrices a `cv.gemm`: el resize lo hace OpenCV sin perder
la equivalencia con torchvision. El detalle está más arriba, en
"Preprocesamiento: los tres pipelines".

**2. Opset 18 en vez de 17.** Pedir 17 dispara el version converter de `onnx`,
que falla en el Reshape del pooling. `onnxruntime-web` soporta 18 sin problema.

**3. Se agregó `@ionic/pwa-elements`**, que `@capacitor/camera` necesita para
abrir la cámara en el navegador de escritorio.

**4. La UI quedó más chica que la que describe el plan.** El plan pedía un
autotest accesible desde la app (Fase 5) y la app lo tuvo, pero los botones se
retiraron de la pantalla junto con la advertencia de fotos reales y la línea de
la regla de decisión, para poder mostrar la app sin explicar el proyecto. Lo que
se pierde es poder validar el pipeline *dentro del teléfono*; en escritorio
sigue estando `npm run autotest`. Ver "La pantalla".

**5. OpenCV.js se carga a mano, no como import.** `opencv.js` son 13 MB: dentro
del bundle de Angular se descargarían siempre, incluso si el usuario se queda en
el pipeline sin OpenCV. Se copia como asset (ver `angular.json`) y
`opencv.service.ts` lo inyecta con un `<script>` la primera vez que hace falta.
Los tipos sí vienen del paquete npm, con un `import type`, que TypeScript borra
al compilar. El APK pasó de ~29 MB a 33.8 MB.

## Dos cosas que solo se ven probando en un navegador

Ninguna de las dos aparece al compilar ni en el autotest de Node; las dos rompían
la app entera con `no available backend found`:

1. **`ort.env.wasm.wasmPaths` tiene que ser una URL absoluta.** ORT carga su
   loader con un `import()` dinámico, y `'assets/ort/'` cuenta como *bare
   specifier*, que el navegador rechaza. Se resuelve con
   `new URL('assets/ort/', document.baseURI).href`, que además sirve dentro del
   WebView de Capacitor, donde el origen no es el mismo.
2. **El bundle por defecto de `onnxruntime-web` carga el artefacto JSEP**
   (`ort-wasm-simd-threaded.jsep.wasm`) aunque solo se pida el execution provider
   `wasm`, así que ese es el par de archivos que hay que copiar en `angular.json`.
   El subpath `onnxruntime-web/wasm` evitaría ese archivo de 27 MB, pero sus tipos
   no resuelven con el `moduleResolution: "node"` que trae el proyecto — queda
   como mejora pendiente si el tamaño molesta.

## Estructura

```
trashnet-tester/
├── android-env.sh                        JAVA_HOME / ANDROID_HOME (source antes de compilar)
├── android/                              proyecto nativo, generado por `npx cap add android`
├── pruebas/
│   ├── autotest-node.ts                  autotest sin navegador (npm run autotest)
│   └── comparar-preproceso.ts            diff píxel a píxel de los 3 pipelines (npm run preproceso)
├── src/assets/
│   ├── models/                           los 6 archivos de onnx/ (3 .onnx + 3 .labels.json)
│   ├── opencv/                           (en build) opencv.js, servido local
│   ├── ort/                              (en build) el .wasm de ONNX Runtime, servido local
│   └── pruebas/                          imágenes de test + esperado.json de referencia
└── src/app/
    ├── modelos/modelo.ts                 registro de los 3 modelos + tipos
    ├── servicios/
    │   ├── resample.ts                   resize bilineal con antialias, port de Pillow
    │   ├── opencv-pipeline.ts            el mismo resample con cv.gemm, + INTER_AREA
    │   ├── opencv.service.ts             carga opencv.js bajo demanda
    │   ├── preprocesamiento.service.ts   elige pipeline: canvas → resize → CHW normalizado
    │   ├── inferencia.service.ts         sesiones ORT + regla de decisión por modelo
    │   └── autotest.service.ts           comparación contra PyTorch (sin UI, ver "Autotest")
    └── home/                             UI
```

Scripts de npm:

| Comando | Qué hace |
|---|---|
| `npm start` | dev server en http://localhost:8100 |
| `npm run build` | build de producción a `www/` |
| `npm run autotest` | autotest en Node de los 3 pipelines, sin navegador |
| `npm run preproceso` | compara los 3 pipelines píxel a píxel, sin correr los modelos |
| `npm run android:apk` | build + `cap sync` + APK de debug |

Y en la raíz del repo, del lado de Python:

| Script | Qué hace |
|---|---|
| `scripts/export_onnx.py` | exporta los 3 `.pt` a `onnx/` |
| `scripts/verify_onnx.py` | contrasta cada ONNX contra su PyTorch |
| `scripts/generar_autotest.py` | copia las imágenes de prueba y genera `esperado.json` |

## Ojo con las predicciones

Está medido que estos modelos **no generalizan** fuera del estudio fotográfico de
TrashNet: el recall del binario cae de 0.818 a 0.119 sobre fotos reales de
celular (453 imágenes de `dataset_gd/`, ver `PyTorch/CONTEXT.md` §8). Antes de
reportar una predicción mala como bug de la app, correr el autotest: si da 6/6,
el pipeline está bien y lo que falla es el modelo.

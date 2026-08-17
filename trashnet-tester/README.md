# trashnet-tester

App Ionic/Angular que corre los 3 modelos TrashNet on-device con ONNX Runtime
Web. Implementa el `PLAN_APP_IONIC.md` de la raíz del repo.

**Estado:** funcionando. El autotest da 6/6 contra las predicciones de los
modelos PyTorch originales, verificado en navegador de escritorio y en un
teléfono Android real.

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

## Autotest

Compara las predicciones de la app contra las de los modelos PyTorch originales
sobre las imágenes de `imagenes_a_test/`. Hay dos formas de correrlo:

- **En la app** (botón "Correr autotest" en la pantalla principal). Es la que
  vale para validar en dispositivo real.
- **En Node**, sin navegador:

  ```bash
  npm run autotest
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

Deja el APK en `android/app/build/outputs/apk/debug/app-debug.apk` (~29 MB).
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

Al probar en el teléfono, conviene este orden: primero **"Correr autotest"** (si
da 6/6, el pipeline entero anda dentro de la WebView), después **Galería**, y
recién ahí **Cámara** con fotos reales.

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
binario de 29 MB en el historial del repo.

## Decisiones que se desvían del plan original

**1. El resize NO usa OpenCV.js.** El plan pedía `cv.resize()`, pero OpenCV no
tiene un bilineal con antialias, que es lo que aplica
`transforms.Resize((224,224))` de torchvision sobre una imagen PIL. Medido sobre
`imagenes_a_test/`:

| Resize | Diferencia vs PIL | Predicciones que coinciden |
|---|---|---|
| `cv.INTER_LINEAR` | MAE 3.0–3.6 | — |
| `cv.INTER_AREA` | MAE 1.0, máx 27/255 | 5/6 (el jerárquico daba `amarillo` en vez de `verde`) |
| `src/app/servicios/resample.ts` | MAE 0.0008, máx 1/255 | **6/6** |

`resample.ts` reimplementa el `Resample.c` de Pillow. Como el resize era lo único
para lo que hacía falta OpenCV (el `cvtColor` RGBA→RGB es un bucle trivial), se
sacó la dependencia entera: 11 MB menos de assets.

**2. Opset 18 en vez de 17.** Pedir 17 dispara el version converter de `onnx`,
que falla en el Reshape del pooling. `onnxruntime-web` soporta 18 sin problema.

**3. Se agregó `@ionic/pwa-elements`**, que `@capacitor/camera` necesita para
abrir la cámara en el navegador de escritorio.

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
├── pruebas/autotest-node.ts              autotest sin navegador (npm run autotest)
├── src/assets/
│   ├── models/                           los 6 archivos de onnx/ (3 .onnx + 3 .labels.json)
│   ├── ort/                              (en build) el .wasm de ONNX Runtime, servido local
│   └── pruebas/                          imágenes de test + esperado.json de referencia
└── src/app/
    ├── modelos/modelo.ts                 registro de los 3 modelos + tipos
    ├── servicios/
    │   ├── resample.ts                   resize bilineal con antialias, port de Pillow
    │   ├── preprocesamiento.service.ts   canvas → resize → CHW normalizado
    │   ├── inferencia.service.ts         sesiones ORT + regla de decisión por modelo
    │   └── autotest.service.ts           comparación contra las predicciones de PyTorch
    └── home/                             UI
```

Scripts de npm:

| Comando | Qué hace |
|---|---|
| `npm start` | dev server en http://localhost:8100 |
| `npm run build` | build de producción a `www/` |
| `npm run autotest` | autotest en Node, sin navegador |
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

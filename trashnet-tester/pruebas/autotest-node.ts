/**
 * Corre los mismos pipelines que la app (los tres modos de preprocesamiento +
 * onnxruntime-web + la regla de decisión por modelo) fuera del navegador, y los
 * compara contra `src/assets/pruebas/esperado.json`, que trae la predicción de
 * los modelos PyTorch originales.
 *
 * Sirve para validar el preprocesamiento sin depender de un navegador. Lo único
 * que no cubre es el decodificado JPEG del navegador (acá lo hace jpeg-js, allá
 * el canvas), así que el autotest de la app sigue siendo el que vale para la
 * Fase 5 en dispositivo.
 *
 * Se espera 6/6 en `pillow` y `opencv`, y 5/6 en `opencv-area`: ese último es
 * `cv.resize(INTER_AREA)`, que no reproduce el resize con antialias de
 * torchvision. Ver `src/app/servicios/opencv-pipeline.ts`.
 *
 * Uso:  npm run autotest
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as jpeg from 'jpeg-js';
import * as ort from 'onnxruntime-web';
import cvReady from '@techstark/opencv-js';
import {
  aTensorConOpenCV,
  ModoPreproceso,
  MODOS,
  OpenCV,
} from '../src/app/servicios/opencv-pipeline';
import { redimensionarComoPIL } from '../src/app/servicios/resample';

interface Etiquetas {
  clases: string[];
  input_size: number;
  mean: number[];
  std: number[];
  umbral_basura?: number;
  idx_basura?: number;
}

interface Caso {
  modelo: string;
  imagen: string;
  clase: string;
  confianza: number;
}

const RAIZ = join(__dirname, '..');
const ASSETS = join(RAIZ, 'src/assets');

function leerJson<T>(ruta: string): T {
  return JSON.parse(readFileSync(ruta, 'utf-8')) as T;
}

/** Equivalente a canvas.getImageData: RGBA en un buffer plano. */
function decodificar(ruta: string): ImageData {
  const { data, width, height } = jpeg.decode(readFileSync(ruta), {
    useTArray: true,
  });
  return { data, width, height } as ImageData;
}

function aTensor(
  cv: OpenCV,
  imagen: ImageData,
  etiquetas: Etiquetas,
  modo: ModoPreproceso,
): Float32Array {
  if (modo !== 'pillow') {
    return aTensorConOpenCV(cv, imagen, etiquetas, modo);
  }
  const lado = etiquetas.input_size;
  const px = redimensionarComoPIL(imagen, lado, lado);
  const plano = lado * lado;
  const tensor = new Float32Array(3 * plano);
  for (let i = 0; i < plano; i++) {
    for (let c = 0; c < 3; c++) {
      tensor[c * plano + i] =
        (px[i * 3 + c] / 255 - etiquetas.mean[c]) / etiquetas.std[c];
    }
  }
  return tensor;
}

function softmax(logits: number[]): number[] {
  const maximo = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maximo));
  const suma = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / suma);
}

function decidir(etiquetas: Etiquetas, probs: number[]) {
  const { umbral_basura, idx_basura } = etiquetas;
  if (umbral_basura !== undefined && idx_basura !== undefined) {
    const idx = probs[idx_basura] >= umbral_basura ? idx_basura : 1 - idx_basura;
    return { clase: etiquetas.clases[idx], confianza: probs[idx] };
  }
  let mejor = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[mejor]) mejor = i;
  }
  return { clase: etiquetas.clases[mejor], confianza: probs[mejor] };
}

async function main() {
  const { casos } = leerJson<{ casos: Caso[] }>(
    join(ASSETS, 'pruebas/esperado.json'),
  );

  const cv = (await (cvReady as unknown as Promise<OpenCV>)) as OpenCV;
  const sesiones = new Map<string, ort.InferenceSession>();
  const etiquetasPorModelo = new Map<string, Etiquetas>();
  let modosConFallo = 0;

  for (const { id: modo, titulo } of MODOS) {
    let fallos = 0;
    console.log(`\n=== ${titulo}  [${modo}] ===`);

    for (const caso of casos) {
      if (!sesiones.has(caso.modelo)) {
        const bytes = readFileSync(join(ASSETS, `models/${caso.modelo}.onnx`));
        sesiones.set(
          caso.modelo,
          await ort.InferenceSession.create(new Uint8Array(bytes)),
        );
        etiquetasPorModelo.set(
          caso.modelo,
          leerJson<Etiquetas>(join(ASSETS, `models/${caso.modelo}.labels.json`)),
        );
      }
      const sesion = sesiones.get(caso.modelo)!;
      const etiquetas = etiquetasPorModelo.get(caso.modelo)!;

      const imagen = decodificar(join(ASSETS, caso.imagen.replace('assets/', '')));
      const datos = aTensor(cv, imagen, etiquetas, modo);
      const lado = etiquetas.input_size;
      const salida = await sesion.run({
        input: new ort.Tensor('float32', datos, [1, 3, lado, lado]),
      });
      const probs = softmax(Array.from(salida['logits'].data as Float32Array));
      const { clase, confianza } = decidir(etiquetas, probs);

      const ok = clase === caso.clase;
      if (!ok) fallos++;
      const dConf = Math.abs(confianza - caso.confianza);
      console.log(
        `[${ok ? 'OK ' : 'DIF'}] ${caso.modelo.padEnd(22)} ${caso.imagen
          .split('/')
          .pop()!
          .padEnd(13)} pytorch=${caso.clase} (${caso.confianza.toFixed(4)})  ` +
          `js=${clase} (${confianza.toFixed(4)})  dconf=${dConf.toFixed(5)}`,
      );
    }

    console.log(`${casos.length - fallos}/${casos.length} coinciden`);
    // `opencv-area` falla a propósito: está para mostrar la diferencia, no para
    // pasar. Solo los dos pipelines fieles a torchvision cuentan para el exit code.
    if (fallos !== 0 && modo !== 'opencv-area') {
      modosConFallo++;
    }
  }

  process.exit(modosConFallo === 0 ? 0 : 1);
}

main();

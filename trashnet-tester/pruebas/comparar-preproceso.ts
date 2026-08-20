import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as jpeg from 'jpeg-js';
import cvReady from '@techstark/opencv-js';
import { redimensionarComoPIL } from '../src/app/servicios/resample';
import { aTensorConOpenCV, OpenCV } from '../src/app/servicios/opencv-pipeline';

const ASSETS = join(__dirname, '../src/assets');
const ET = { input_size: 224, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };

function decodificar(ruta: string): ImageData {
  const { data, width, height } = jpeg.decode(readFileSync(ruta), { useTArray: true });
  return { data, width, height } as ImageData;
}

/** El tensor CHW de referencia, con el resample de Pillow en JS puro. */
function tensorPillow(imagen: ImageData): Float32Array {
  const lado = ET.input_size;
  const px = redimensionarComoPIL(imagen, lado, lado);
  const plano = lado * lado;
  const t = new Float32Array(3 * plano);
  for (let i = 0; i < plano; i++)
    for (let c = 0; c < 3; c++)
      t[c * plano + i] = (px[i * 3 + c] / 255 - ET.mean[c]) / ET.std[c];
  return t;
}

async function main() {
  const cv = (await (cvReady as any)) as OpenCV;
  console.log('OpenCV listo:', (cv as any).getBuildInformation ? 'sí' : 'sí');

  for (const nombre of ['test_1.jpeg', 'test_2.jpeg']) {
    const imagen = decodificar(join(ASSETS, 'pruebas', nombre));
    const tp = Date.now();
    const ref = tensorPillow(imagen);
    console.log(`${nombre} ${'pillow'.padEnd(12)} ${imagen.width}x${imagen.height}  referencia                                                  ${Date.now() - tp}ms`);

    for (const modo of ['opencv', 'opencv-area'] as const) {
      const t0 = Date.now();
      const got = aTensorConOpenCV(cv, imagen, ET, modo);
      const ms = Date.now() - t0;

      let maxDif = 0, distintos = 0, suma = 0;
      for (let i = 0; i < ref.length; i++) {
        const d = Math.abs(ref[i] - got[i]);
        suma += d;
        if (d > 1e-6) distintos++;
        if (d > maxDif) maxDif = d;
      }
      // en unidades de 0-255 (deshaciendo la normalizacion, std medio ~0.226)
      const escala = 255 * 0.226;
      console.log(
        `${nombre} ${modo.padEnd(12)} ${imagen.width}x${imagen.height}  ` +
        `maxDif=${(maxDif * escala).toFixed(3)}/255  ` +
        `MAE=${((suma / ref.length) * escala).toFixed(4)}/255  ` +
        `pixeles distintos=${((distintos / ref.length) * 100).toFixed(3)}%  ${ms}ms`,
      );
    }
  }
}
main();

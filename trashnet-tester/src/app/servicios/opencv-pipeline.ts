/**
 * Pipeline de preprocesamiento implementado con OpenCV.js.
 *
 * Es el equivalente de `preprocesamiento.service.ts` (que usa `resample.ts`, un
 * port a mano del resample de Pillow), pero delegando el trabajo en OpenCV:
 * conversión de color, redimensión y normalización.
 *
 * ## Por qué el resize no es un `cv.resize` a secas
 *
 * Los modelos se entrenaron con `transforms.Resize((224,224))` de torchvision,
 * que sobre una imagen PIL aplica un bilineal **con antialias**: el soporte del
 * filtro se ensancha con el factor de reducción, así que al achicar promedia
 * todos los píxeles que caen dentro, no solo los 2×2 vecinos.
 *
 * OpenCV no tiene esa variante. `INTER_LINEAR` no filtra (MAE 3.0-3.6 contra
 * PIL) e `INTER_AREA` es un filtro de caja, no triangular (MAE 1.0, hasta
 * 27/255 en un píxel), suficiente para que el modelo jerárquico cambie su
 * predicción sobre `test 1.jpeg` de `verde` a `amarillo`.
 *
 * La salida es que el resample de Pillow **es separable y lineal**: cada píxel
 * de salida es una combinación lineal de los de entrada, primero por filas y
 * después por columnas. Eso se escribe como dos productos de matrices
 *
 *     horizontal:  (H × Went) · (Went × Wsal)  →  H × Wsal
 *     vertical:    (Hsal × H) · (H     × Wsal) →  Hsal × Wsal
 *
 * donde las matrices son los pesos triangulares que ya calcula
 * `precalcularCoeficientes()`. Los dos productos los hace `cv.gemm`, o sea que
 * el resize lo ejecuta OpenCV y aun así el resultado es idéntico al de PIL.
 *
 * El modo `'opencv-area'` existe para poder mostrar la diferencia en vivo desde
 * la app: es el `cv.resize(INTER_AREA)` directo, el camino "obvio", y falla el
 * autotest en el caso del jerárquico.
 */

import { Coeficientes, precalcularCoeficientes } from './resample';

/** El namespace de OpenCV.js. Se pasa como parámetro para que este módulo sirva
 *  igual en el navegador (script tag) que en Node (require del paquete). */
export type OpenCV = typeof import('@techstark/opencv-js');

export type ModoPreproceso = 'pillow' | 'opencv' | 'opencv-area';

export const MODOS: {
  id: ModoPreproceso;
  /** Nombre en el desplegable: quién hace el resize y si coincide con el entrenamiento. */
  titulo: string;
  /** Nombre corto, para la línea de tiempos debajo de cada predicción. */
  corto: string;
  /** Una línea de explicación bajo el desplegable, en lenguaje llano: la ve
   *  cualquiera que abra la app. El detalle técnico está en el README. */
  detalle: string;
}[] = [
  {
    id: 'pillow',
    titulo: 'Sin OpenCV — exacto',
    corto: 'sin OpenCV',
    detalle:
      'No usa OpenCV. Da el mismo resultado que el modelo original y es el más rápido.',
  },
  {
    id: 'opencv',
    titulo: 'OpenCV — exacto',
    corto: 'OpenCV',
    detalle:
      'OpenCV prepara la foto igual que cuando se entrenó el modelo. Es el más fiel, pero tarda un poco más.',
  },
  {
    id: 'opencv-area',
    titulo: 'OpenCV — aproximado',
    corto: 'OpenCV aprox.',
    detalle:
      'OpenCV con su forma más simple de achicar la foto. Es rápido, pero puede cambiar la respuesta del modelo.',
  },
];

/** Matriz de pesos como Mat CV_32F, en la orientación que pide cada pasada. */
function matrizDeCoeficientes(
  cv: OpenCV,
  coefs: Coeficientes[],
  tamEntrada: number,
  transpuesta: boolean,
): any {
  const tamSalida = coefs.length;
  // Sin transponer: (salida × entrada), la que multiplica por la izquierda.
  const filas = transpuesta ? tamEntrada : tamSalida;
  const columnas = transpuesta ? tamSalida : tamEntrada;
  const datos = new Float32Array(filas * columnas);

  for (let i = 0; i < tamSalida; i++) {
    const { inicio, pesos } = coefs[i];
    for (let k = 0; k < pesos.length; k++) {
      const j = inicio + k;
      datos[transpuesta ? j * tamSalida + i : i * tamEntrada + j] = pesos[k];
    }
  }
  return cv.matFromArray(filas, columnas, cv.CV_32F, datos as unknown as number[]);
}

/**
 * Lleva una imagen a `Float32Array` en layout CHW usando OpenCV.
 *
 * Todos los `cv.Mat` se liberan a mano: OpenCV.js corre sobre memoria WASM, que
 * el recolector de basura de JavaScript no toca. Sin los `delete()` la app se
 * queda sin heap después de unas cuantas predicciones.
 */
export function aTensorConOpenCV(
  cv: OpenCV,
  imagen: ImageData,
  etiquetas: { input_size: number; mean: number[]; std: number[] },
  modo: 'opencv' | 'opencv-area',
): Float32Array {
  const lado = etiquetas.input_size;
  const plano = lado * lado;
  const tensor = new Float32Array(3 * plano);

  // Todo lo que se cree se registra acá y se libera en el finally.
  const basura: { delete(): void }[] = [];
  const nueva = <T extends { delete(): void }>(m: T): T => {
    basura.push(m);
    return m;
  };

  try {
    const src = nueva(cv.matFromImageData(imagen));

    // El canvas entrega RGBA; el modelo espera RGB. El alfa se descarta.
    const rgb = nueva(new cv.Mat());
    cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);

    let redimensionada: any;

    if (modo === 'opencv-area') {
      redimensionada = nueva(new cv.Mat());
      cv.resize(rgb, redimensionada, new cv.Size(lado, lado), 0, 0, cv.INTER_AREA);
    } else {
      redimensionada = nueva(redimensionarConGemm(cv, rgb, lado, lado, nueva));
    }

    // CHW: cada canal por separado, que es justo lo que devuelve cv.split.
    const canales = nueva(new cv.MatVector());
    cv.split(redimensionada, canales);

    for (let c = 0; c < 3; c++) {
      const canal = nueva(canales.get(c));
      // convertTo hace escala y desplazamiento en una sola pasada:
      //     px * alpha + beta  =  px / (255·std) - mean/std  =  (px/255 - mean)/std
      // que es exactamente ToTensor() + Normalize(mean, std).
      const normalizado = nueva(new cv.Mat());
      canal.convertTo(
        normalizado,
        cv.CV_32F,
        1 / (255 * etiquetas.std[c]),
        -etiquetas.mean[c] / etiquetas.std[c],
      );
      tensor.set(normalizado.data32F, c * plano);
    }

    return tensor;
  } finally {
    for (const m of basura) {
      m.delete();
    }
  }
}

/**
 * Resample bilineal con antialias de Pillow, ejecutado con `cv.gemm`.
 *
 * Se hace canal por canal porque `gemm` solo opera sobre matrices de un canal.
 * El redondeo a uint8 entre las dos pasadas no es un detalle de implementación:
 * Pillow lo hace, y omitirlo mueve el resultado.
 */
function redimensionarConGemm(
  cv: OpenCV,
  rgb: any,
  anchoSalida: number,
  altoSalida: number,
  nueva: <T extends { delete(): void }>(m: T) => T,
): any {
  const anchoEntrada = rgb.cols;
  const altoEntrada = rgb.rows;

  // Horizontal: multiplica por la derecha, así que va transpuesta (Went × Wsal).
  const mh = nueva(
    matrizDeCoeficientes(
      cv,
      precalcularCoeficientes(anchoEntrada, anchoSalida),
      anchoEntrada,
      true,
    ),
  );
  // Vertical: multiplica por la izquierda (Hsal × Hent).
  const mv = nueva(
    matrizDeCoeficientes(
      cv,
      precalcularCoeficientes(altoEntrada, altoSalida),
      altoEntrada,
      false,
    ),
  );
  const vacio = nueva(new cv.Mat());

  const entrada = nueva(new cv.MatVector());
  cv.split(rgb, entrada);
  const salida = nueva(new cv.MatVector());

  for (let c = 0; c < 3; c++) {
    const canal = nueva(entrada.get(c));
    const canal32 = nueva(new cv.Mat());
    canal.convertTo(canal32, cv.CV_32F);

    // (H × Went) · (Went × Wsal) → H × Wsal
    const horizontal = nueva(new cv.Mat());
    cv.gemm(canal32, mh, 1, vacio, 0, horizontal, 0);

    // Pillow satura y redondea a uint8 acá, antes de la pasada vertical.
    const horizontal8 = nueva(new cv.Mat());
    horizontal.convertTo(horizontal8, cv.CV_8U);
    const horizontal32 = nueva(new cv.Mat());
    horizontal8.convertTo(horizontal32, cv.CV_32F);

    // (Hsal × H) · (H × Wsal) → Hsal × Wsal
    const vertical = nueva(new cv.Mat());
    cv.gemm(mv, horizontal32, 1, vacio, 0, vertical, 0);

    const vertical8 = nueva(new cv.Mat());
    vertical.convertTo(vertical8, cv.CV_8U);
    salida.push_back(vertical8);
  }

  const destino = new cv.Mat();
  cv.merge(salida, destino);
  return destino;
}

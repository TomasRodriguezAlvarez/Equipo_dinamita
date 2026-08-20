/**
 * Reimplementación del resample bilineal con antialias de Pillow
 * (`Resample.c`, `precompute_coeffs` + `ImagingResampleHorizontal/Vertical`).
 *
 * Por qué existe: los modelos se entrenaron con `transforms.Resize((224,224))`
 * de torchvision, que sobre una imagen PIL llama a `Image.resize(..., BILINEAR)`.
 * Ese bilineal escala el soporte del filtro con el factor de reducción (o sea,
 * hace antialias). OpenCV no trae esa variante: `INTER_LINEAR` no filtra, e
 * `INTER_AREA` es un filtro de caja. Medido sobre `imagenes_a_test/`,
 * `INTER_AREA` daba hasta 27/255 de diferencia por píxel y llegaba a cambiar la
 * clase predicha del modelo jerárquico (verde → amarillo); esto se queda en
 * 1/255 sobre el 0.08% de los píxeles y no cambia ninguna.
 *
 * `precalcularCoeficientes()` se exporta porque `opencv-pipeline.ts` arma con
 * esos mismos pesos las matrices que multiplica con `cv.gemm`: ahí el resample
 * lo ejecuta OpenCV, pero el filtro sigue siendo este.
 *
 * Las dos pasadas son separables y van en el mismo orden que Pillow: primero
 * horizontal, después vertical, redondeando a uint8 en medio (ese redondeo
 * intermedio también es parte del algoritmo original).
 */

export interface Coeficientes {
  /** Primera columna/fila de entrada que aporta a este píxel de salida. */
  inicio: number;
  /** Pesos ya normalizados para que sumen 1. */
  pesos: Float64Array;
}

export function precalcularCoeficientes(
  tamEntrada: number,
  tamSalida: number,
): Coeficientes[] {
  const escala = tamEntrada / tamSalida;
  // Al achicar, el filtro se ensancha con el factor de reducción: eso es lo que
  // produce el antialias. Al agrandar se queda en 1 y es un bilineal común.
  const escalaFiltro = Math.max(1.0, escala);
  const soporte = escalaFiltro; // el filtro triangular tiene soporte 1
  const inverso = 1.0 / escalaFiltro;

  const salida: Coeficientes[] = [];
  for (let i = 0; i < tamSalida; i++) {
    const centro = (i + 0.5) * escala;
    const inicio = Math.max(0, Math.trunc(centro - soporte + 0.5));
    const fin = Math.min(tamEntrada, Math.trunc(centro + soporte + 0.5));

    const pesos = new Float64Array(fin - inicio);
    let suma = 0;
    for (let k = 0; k < pesos.length; k++) {
      const t = Math.abs((inicio + k - centro + 0.5) * inverso);
      const w = t < 1.0 ? 1.0 - t : 0.0;
      pesos[k] = w;
      suma += w;
    }
    if (suma !== 0) {
      for (let k = 0; k < pesos.length; k++) {
        pesos[k] /= suma;
      }
    }
    salida.push({ inicio, pesos });
  }
  return salida;
}

/** Redondea al entero más cercano y satura en [0, 255], como el clip8 de Pillow. */
function clip8(v: number): number {
  const r = Math.floor(v + 0.5);
  return r < 0 ? 0 : r > 255 ? 255 : r;
}

/** RGBA de un ImageData → RGB con el ancho ya reducido. Salida HWC. */
function pasadaHorizontal(
  src: Uint8ClampedArray,
  anchoEntrada: number,
  alto: number,
  coefs: Coeficientes[],
): Uint8Array {
  const anchoSalida = coefs.length;
  const salida = new Uint8Array(anchoSalida * alto * 3);

  for (let y = 0; y < alto; y++) {
    const baseEntrada = y * anchoEntrada * 4;
    const baseSalida = y * anchoSalida * 3;
    for (let x = 0; x < anchoSalida; x++) {
      const { inicio, pesos } = coefs[x];
      let r = 0;
      let g = 0;
      let b = 0;
      for (let k = 0; k < pesos.length; k++) {
        const p = baseEntrada + (inicio + k) * 4; // se ignora el canal alfa
        const w = pesos[k];
        r += src[p] * w;
        g += src[p + 1] * w;
        b += src[p + 2] * w;
      }
      const d = baseSalida + x * 3;
      salida[d] = clip8(r);
      salida[d + 1] = clip8(g);
      salida[d + 2] = clip8(b);
    }
  }
  return salida;
}

/** RGB HWC → RGB HWC con el alto ya reducido. */
function pasadaVertical(
  src: Uint8Array,
  ancho: number,
  coefs: Coeficientes[],
): Uint8Array {
  const altoSalida = coefs.length;
  const salida = new Uint8Array(ancho * altoSalida * 3);

  for (let y = 0; y < altoSalida; y++) {
    const { inicio, pesos } = coefs[y];
    for (let x = 0; x < ancho; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let k = 0; k < pesos.length; k++) {
        const p = ((inicio + k) * ancho + x) * 3;
        const w = pesos[k];
        r += src[p] * w;
        g += src[p + 1] * w;
        b += src[p + 2] * w;
      }
      const d = (y * ancho + x) * 3;
      salida[d] = clip8(r);
      salida[d + 1] = clip8(g);
      salida[d + 2] = clip8(b);
    }
  }
  return salida;
}

/**
 * Equivalente de `PIL.Image.resize((ancho, alto), Image.BILINEAR)`.
 *
 * Devuelve los píxeles RGB en layout HWC (fila, columna, canal), sin alfa.
 * No preserva el aspect ratio a propósito: el entrenamiento usó
 * `Resize((224,224))`, que deforma la imagen, no `Resize(224) + CenterCrop`.
 */
export function redimensionarComoPIL(
  imagen: ImageData,
  anchoSalida: number,
  altoSalida: number,
): Uint8Array {
  const horizontal = pasadaHorizontal(
    imagen.data,
    imagen.width,
    imagen.height,
    precalcularCoeficientes(imagen.width, anchoSalida),
  );
  return pasadaVertical(
    horizontal,
    anchoSalida,
    precalcularCoeficientes(imagen.height, altoSalida),
  );
}

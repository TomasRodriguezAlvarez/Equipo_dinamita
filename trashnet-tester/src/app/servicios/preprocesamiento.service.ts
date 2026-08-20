import { Injectable } from '@angular/core';
import { EtiquetasModelo } from '../modelos/modelo';
import {
  aTensorConOpenCV,
  ModoPreproceso,
} from './opencv-pipeline';
import { OpencvService } from './opencv.service';
import { redimensionarComoPIL } from './resample';

/**
 * Replica el pipeline `eval_tf` de los notebooks de entrenamiento:
 *
 *     transforms.Resize((224, 224))   →  redimensión directa, SIN crop
 *     transforms.ToTensor()           →  RGB, [0, 1], layout CHW
 *     transforms.Normalize(mean, std) →  (px - mean) / std por canal
 *
 * Desviarse de esto (por ejemplo agregando un center-crop "para que se vea
 * mejor") mete un desajuste de dominio encima del que ya está medido en
 * PyTorch/CONTEXT.md §8, y arruina la comparación con los números del informe.
 *
 * Hay tres implementaciones seleccionables desde la UI; la comparación entre
 * ellas está en `opencv-pipeline.ts` y en el README.
 */
@Injectable({ providedIn: 'root' })
export class PreprocesamientoService {
  /** Implementación en uso. La cambia el selector de la pantalla principal. */
  modo: ModoPreproceso = 'opencv';

  // Los 3 modelos comparten input_size, mean y std, así que al comparar los 3
  // sobre la misma foto el preprocesamiento da lo mismo y basta con hacerlo una
  // vez. Importa: el resize con gemm cuesta cientos de ms.
  private cache = new WeakMap<HTMLImageElement, { clave: string; tensor: Float32Array }>();

  constructor(private opencv: OpencvService) {}

  /** Carga un dataURL/objectURL como <img> ya decodificada. */
  cargarImagen(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('No se pudo decodificar la imagen'));
      img.src = url;
    });
  }

  /** Deja OpenCV cargado si el modo actual lo necesita. */
  async precargar(): Promise<void> {
    if (this.modo !== 'pillow') {
      await this.opencv.cargar();
    }
  }

  /**
   * Devuelve el Float32Array en layout CHW listo para onnxruntime-web,
   * de largo 3 * input_size * input_size.
   */
  async aTensor(
    img: HTMLImageElement,
    etiquetas: EtiquetasModelo,
  ): Promise<Float32Array> {
    const clave = [
      this.modo,
      etiquetas.input_size,
      etiquetas.mean.join(','),
      etiquetas.std.join(','),
    ].join('|');

    const guardado = this.cache.get(img);
    if (guardado && guardado.clave === clave) {
      return guardado.tensor;
    }

    const datos = this.aImageData(img);
    const tensor =
      this.modo === 'pillow'
        ? this.conResample(datos, etiquetas)
        : aTensorConOpenCV(await this.opencv.cargar(), datos, etiquetas, this.modo);

    this.cache.set(img, { clave, tensor });
    return tensor;
  }

  /**
   * El canvas es la única forma de leer los píxeles de un <img>. Se dibuja al
   * tamaño natural y se redimensiona aparte: si se dejara escalar al canvas, el
   * navegador aplicaría su propio filtro, que no es el de PIL ni el de OpenCV.
   */
  private aImageData(img: HTMLImageElement): ImageData {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      throw new Error('No se pudo crear el contexto 2D del canvas');
    }
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }

  /** Pipeline en JavaScript puro, sin OpenCV: el port del resample de Pillow. */
  private conResample(
    datos: ImageData,
    etiquetas: EtiquetasModelo,
  ): Float32Array {
    const lado = etiquetas.input_size;
    const px = redimensionarComoPIL(datos, lado, lado); // RGB en HWC

    // ToTensor + Normalize: HWC → CHW, [0,255] → [0,1] → estandarizado.
    const plano = lado * lado;
    const tensor = new Float32Array(3 * plano);
    const { mean, std } = etiquetas;
    for (let i = 0; i < plano; i++) {
      for (let c = 0; c < 3; c++) {
        tensor[c * plano + i] = (px[i * 3 + c] / 255 - mean[c]) / std[c];
      }
    }
    return tensor;
  }
}

import { Injectable } from '@angular/core';
import * as ort from 'onnxruntime-web';
import {
  DefinicionModelo,
  EtiquetasModelo,
  ProbabilidadClase,
  Resultado,
} from '../modelos/modelo';
import { PreprocesamientoService } from './preprocesamiento.service';

// Los .wasm salen de assets/ort/ (los copia angular.json desde node_modules) y
// no del CDN por defecto de onnxruntime-web: dentro del WebView de Capacitor no
// hay red garantizada.
//
// Tiene que ser una URL absoluta: ORT carga el loader con un import() dinámico, y
// una ruta relativa como 'assets/ort/' cuenta como bare specifier y lo rechaza el
// navegador. document.baseURI la resuelve bien tanto en `ionic serve` como en el
// WebView de Capacitor, donde el origen no es el mismo.
ort.env.wasm.wasmPaths = new URL('assets/ort/', document.baseURI).href;
// El backend multihilo necesita cross-origin isolation (COOP/COEP), que ni
// `ionic serve` ni el WebView traen puestos. Con 1 hilo funciona en todos lados.
ort.env.wasm.numThreads = 1;

@Injectable({ providedIn: 'root' })
export class InferenciaService {
  private sesiones = new Map<string, Promise<ort.InferenceSession>>();
  private etiquetas = new Map<string, Promise<EtiquetasModelo>>();

  constructor(private preproceso: PreprocesamientoService) {}

  /** Etiquetas del modelo (mean/std/clases/umbral), cacheadas por id. */
  cargarEtiquetas(modelo: DefinicionModelo): Promise<EtiquetasModelo> {
    let pendiente = this.etiquetas.get(modelo.id);
    if (!pendiente) {
      pendiente = fetch(modelo.rutaEtiquetas).then((r) => {
        if (!r.ok) {
          throw new Error(`No se pudo leer ${modelo.rutaEtiquetas}`);
        }
        return r.json() as Promise<EtiquetasModelo>;
      });
      this.etiquetas.set(modelo.id, pendiente);
    }
    return pendiente;
  }

  /** Sesión ONNX del modelo, cacheada por id (crearla cuesta ~1 s). */
  cargarSesion(modelo: DefinicionModelo): Promise<ort.InferenceSession> {
    let pendiente = this.sesiones.get(modelo.id);
    if (!pendiente) {
      pendiente = ort.InferenceSession.create(modelo.rutaOnnx, {
        executionProviders: ['wasm'],
      });
      this.sesiones.set(modelo.id, pendiente);
    }
    return pendiente;
  }

  /** Deja el modelo y el preprocesamiento listos antes de la primera foto. */
  async precargar(modelo: DefinicionModelo): Promise<void> {
    await Promise.all([
      this.cargarSesion(modelo),
      this.cargarEtiquetas(modelo),
      this.preproceso.precargar(),
    ]);
  }

  async predecir(
    modelo: DefinicionModelo,
    img: HTMLImageElement,
  ): Promise<Resultado> {
    const [sesion, etiquetas] = await Promise.all([
      this.cargarSesion(modelo),
      this.cargarEtiquetas(modelo),
    ]);

    const t0 = performance.now();
    const datos = await this.preproceso.aTensor(img, etiquetas);
    const t1 = performance.now();

    const lado = etiquetas.input_size;
    const entrada = new ort.Tensor('float32', datos, [1, 3, lado, lado]);
    const salida = await sesion.run({ input: entrada });
    const t2 = performance.now();

    const logits = Array.from(salida['logits'].data as Float32Array);
    const probs = softmax(logits);

    const probabilidades: ProbabilidadClase[] = etiquetas.clases.map(
      (clase, i) => ({ clase, prob: probs[i] }),
    );

    const decision = this.decidir(etiquetas, probs);

    return {
      ...decision,
      modo: this.preproceso.modo,
      probabilidades,
      msPreproceso: t1 - t0,
      msInferencia: t2 - t1,
    };
  }

  /**
   * La regla de decisión NO es la misma para los 3 modelos.
   *
   * El binario se calibró sobre validación con umbral 0.837 en vez de argmax
   * (que equivale a un umbral de 0.5 sin justificar). Con ese cambio el F1 de
   * `basura` sube de 0.655 a 0.818 — más de cuatro veces lo que aportó Optuna.
   * Ver PyTorch/CONTEXT.md §8. El umbral se lee del labels.json.
   */
  private decidir(
    etiquetas: EtiquetasModelo,
    probs: number[],
  ): { clase: string; confianza: number; regla: string } {
    const { umbral_basura, idx_basura } = etiquetas;

    if (umbral_basura !== undefined && idx_basura !== undefined) {
      const pBasura = probs[idx_basura];
      const esBasura = pBasura >= umbral_basura;
      const idx = esBasura ? idx_basura : 1 - idx_basura;
      return {
        clase: etiquetas.clases[idx],
        confianza: probs[idx],
        regla:
          `umbral calibrado: P(${etiquetas.clases[idx_basura]}) = ` +
          `${pBasura.toFixed(4)} ${esBasura ? '≥' : '<'} ${umbral_basura}`,
      };
    }

    let mejor = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[mejor]) {
        mejor = i;
      }
    }
    return {
      clase: etiquetas.clases[mejor],
      confianza: probs[mejor],
      regla: 'argmax sobre las probabilidades',
    };
  }
}

/** Softmax estable: se resta el máximo para que exp() no desborde. */
function softmax(logits: number[]): number[] {
  const maximo = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maximo));
  const suma = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / suma);
}

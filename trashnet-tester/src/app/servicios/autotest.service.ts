import { Injectable } from '@angular/core';
import { MODELOS } from '../modelos/modelo';
import { InferenciaService } from './inferencia.service';
import { PreprocesamientoService } from './preprocesamiento.service';

/** Un caso de `assets/pruebas/esperado.json`, generado por scripts/generar_autotest.py. */
interface CasoEsperado {
  modelo: string;
  imagen: string;
  clase: string;
  confianza: number;
}

export interface ResultadoAutotest {
  modelo: string;
  imagen: string;
  claseEsperada: string;
  claseObtenida: string;
  confianzaEsperada: number;
  confianzaObtenida: number;
  ok: boolean;
}

/**
 * Corre las imágenes de `imagenes_a_test/` con onnxruntime-web y compara contra
 * la predicción de los modelos PyTorch originales (Fase 5 del plan).
 *
 * Vive dentro de la app y no en un test de Node a propósito: así se puede
 * ejecutar también en el dispositivo real, que es donde el plan pide validar.
 */
@Injectable({ providedIn: 'root' })
export class AutotestService {
  constructor(
    private inferencia: InferenciaService,
    private preproceso: PreprocesamientoService,
  ) {}

  async correr(): Promise<ResultadoAutotest[]> {
    const respuesta = await fetch('assets/pruebas/esperado.json');
    if (!respuesta.ok) {
      throw new Error('Falta assets/pruebas/esperado.json (correr scripts/generar_autotest.py)');
    }
    const { casos } = (await respuesta.json()) as { casos: CasoEsperado[] };

    // Las imágenes se decodifican una sola vez aunque se usen en varios casos.
    const cache = new Map<string, HTMLImageElement>();
    const salida: ResultadoAutotest[] = [];

    for (const caso of casos) {
      const modelo = MODELOS.find((m) => m.id === caso.modelo);
      if (!modelo) {
        throw new Error(`El esperado.json referencia un modelo desconocido: ${caso.modelo}`);
      }

      let img = cache.get(caso.imagen);
      if (!img) {
        img = await this.preproceso.cargarImagen(caso.imagen);
        cache.set(caso.imagen, img);
      }

      const resultado = await this.inferencia.predecir(modelo, img);
      salida.push({
        modelo: caso.modelo,
        imagen: caso.imagen.split('/').pop() ?? caso.imagen,
        claseEsperada: caso.clase,
        claseObtenida: resultado.clase,
        confianzaEsperada: caso.confianza,
        confianzaObtenida: resultado.confianza,
        ok: caso.clase === resultado.clase,
      });
    }

    return salida;
  }
}

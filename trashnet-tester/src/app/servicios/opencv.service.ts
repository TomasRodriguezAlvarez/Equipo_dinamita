import { Injectable } from '@angular/core';
import { OpenCV } from './opencv-pipeline';

/**
 * Carga OpenCV.js bajo demanda.
 *
 * No se importa como módulo a propósito: `opencv.js` son 13 MB que quedarían
 * dentro del bundle de Angular y se descargarían siempre, incluso si el usuario
 * nunca sale del pipeline en JavaScript puro. Se copia como asset (ver
 * `angular.json`) y se inyecta con un `<script>` la primera vez que hace falta.
 *
 * El archivo es un UMD: al no haber `define` ni `module` en el navegador, deja
 * el módulo en `window.cv`. En las versiones nuevas de Emscripten eso es una
 * promesa que resuelve cuando el WASM terminó de inicializar; en las viejas es
 * el módulo directo y hay que esperar `onRuntimeInitialized`. Se contemplan los
 * dos casos porque cuál de los dos toca depende de con qué se compiló el build.
 */
@Injectable({ providedIn: 'root' })
export class OpencvService {
  private pendiente: Promise<OpenCV> | null = null;

  /** `true` cuando OpenCV ya está listo (para no mostrar "cargando" de más). */
  listo = false;

  cargar(): Promise<OpenCV> {
    if (!this.pendiente) {
      this.pendiente = this.inyectar().then((cv) => {
        this.listo = true;
        return cv;
      });
    }
    return this.pendiente;
  }

  private inyectar(): Promise<OpenCV> {
    return new Promise<OpenCV>((resolve, reject) => {
      const script = document.createElement('script');
      // Absoluta por la misma razón que los .wasm de ONNX: dentro del WebView
      // de Capacitor el origen no es el mismo que el de la página.
      script.src = new URL('assets/opencv/opencv.js', document.baseURI).href;
      script.async = true;

      script.onerror = () =>
        reject(new Error('No se pudo cargar assets/opencv/opencv.js'));

      script.onload = () => {
        const global = window as unknown as { cv?: unknown };
        const cv = global.cv;
        if (!cv) {
          reject(new Error('opencv.js cargó pero no dejó nada en window.cv'));
          return;
        }
        // Build nuevo: window.cv es una promesa del módulo.
        if (typeof (cv as PromiseLike<unknown>).then === 'function') {
          (cv as PromiseLike<OpenCV>).then(resolve, reject);
          return;
        }
        // Build viejo: ya está inicializado, o avisa por onRuntimeInitialized.
        const modulo = cv as OpenCV & { onRuntimeInitialized?: () => void };
        if (modulo.Mat) {
          resolve(modulo);
        } else {
          modulo.onRuntimeInitialized = () => resolve(modulo);
        }
      };

      document.head.appendChild(script);
    });
  }
}

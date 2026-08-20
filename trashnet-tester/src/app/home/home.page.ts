import { Component, OnInit } from '@angular/core';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';
import { DefinicionModelo, MODELOS, Resultado } from '../modelos/modelo';
import { InferenciaService } from '../servicios/inferencia.service';
import { ModoPreproceso, MODOS } from '../servicios/opencv-pipeline';
import { OpencvService } from '../servicios/opencv.service';
import { PreprocesamientoService } from '../servicios/preprocesamiento.service';

interface FilaResultado {
  modelo: DefinicionModelo;
  resultado: Resultado;
}

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
  standalone: false,
})
export class HomePage implements OnInit {
  readonly modelos = MODELOS;
  readonly modos = MODOS;
  modeloSeleccionado: DefinicionModelo = MODELOS[0];

  imagenUrl: string | null = null;
  private imagen: HTMLImageElement | null = null;

  filas: FilaResultado[] = [];
  cargando = false;
  estado = '';
  error: string | null = null;

  constructor(
    private inferencia: InferenciaService,
    private preproceso: PreprocesamientoService,
    public opencv: OpencvService,
  ) {}

  get modoSeleccionado(): ModoPreproceso {
    return this.preproceso.modo;
  }

  /** Cambiar de pipeline invalida los resultados que hay en pantalla. */
  set modoSeleccionado(modo: ModoPreproceso) {
    if (modo === this.preproceso.modo) {
      return;
    }
    this.preproceso.modo = modo;
    this.filas = [];
    this.preproceso.precargar().catch(() => {
      /* si falla, el error real se muestra al predecir */
    });
  }

  get detalleModo(): string {
    return this.modos.find((m) => m.id === this.modoSeleccionado)?.detalle ?? '';
  }

  /** Nombre corto del pipeline con que se obtuvo un resultado. */
  nombreModo(modo: ModoPreproceso): string {
    return this.modos.find((m) => m.id === modo)?.corto ?? modo;
  }

  ngOnInit(): void {
    // Crear la sesión ONNX y arrancar los dos WASM (ONNX Runtime y OpenCV)
    // tarda unos segundos. Hacerlo ahora evita que ese costo se sume a la
    // primera predicción.
    this.inferencia.precargar(this.modeloSeleccionado).catch(() => {
      /* si falla, el error real se muestra al predecir */
    });
  }

  compararIds(a: DefinicionModelo, b: DefinicionModelo): boolean {
    return a?.id === b?.id;
  }

  async tomarFoto(): Promise<void> {
    await this.obtenerImagen(CameraSource.Camera);
  }

  async elegirDeGaleria(): Promise<void> {
    await this.obtenerImagen(CameraSource.Photos);
  }

  private async obtenerImagen(source: CameraSource): Promise<void> {
    try {
      const foto = await Camera.getPhoto({
        resultType: CameraResultType.DataUrl,
        source,
        quality: 90,
        // Sin edición: cualquier recorte del usuario cambiaría el encuadre
        // respecto del Resize directo con que se entrenó.
        allowEditing: false,
      });
      if (!foto.dataUrl) {
        return;
      }
      this.error = null;
      this.filas = [];
      this.imagenUrl = foto.dataUrl;
      this.imagen = await this.preproceso.cargarImagen(foto.dataUrl);
    } catch (e) {
      // El usuario cancelando el diálogo también entra acá; no es un error real.
      const msg = e instanceof Error ? e.message : String(e);
      if (!/cancel/i.test(msg)) {
        this.error = msg;
      }
    }
  }

  async predecir(): Promise<void> {
    await this.correr([this.modeloSeleccionado]);
  }

  async predecirTodos(): Promise<void> {
    await this.correr(this.modelos);
  }

  private async correr(modelos: DefinicionModelo[]): Promise<void> {
    if (!this.imagen) {
      return;
    }
    this.cargando = true;
    this.error = null;
    this.filas = [];
    try {
      for (const modelo of modelos) {
        this.estado = `Corriendo ${modelo.titulo}…`;
        const resultado = await this.inferencia.predecir(modelo, this.imagen);
        this.filas = [...this.filas, { modelo, resultado }];
      }
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.cargando = false;
      this.estado = '';
    }
  }

  porcentaje(prob: number): string {
    return `${(prob * 100).toFixed(1)}%`;
  }
}

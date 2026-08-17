import { Component, OnInit } from '@angular/core';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';
import { DefinicionModelo, MODELOS, Resultado } from '../modelos/modelo';
import { AutotestService, ResultadoAutotest } from '../servicios/autotest.service';
import { InferenciaService } from '../servicios/inferencia.service';
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
  modeloSeleccionado: DefinicionModelo = MODELOS[0];

  imagenUrl: string | null = null;
  private imagen: HTMLImageElement | null = null;

  filas: FilaResultado[] = [];
  cargando = false;
  estado = '';
  error: string | null = null;

  autotest: ResultadoAutotest[] = [];

  constructor(
    private inferencia: InferenciaService,
    private preproceso: PreprocesamientoService,
    private autotestService: AutotestService,
  ) {}

  ngOnInit(): void {
    // Crear la sesión ONNX y arrancar el WASM de OpenCV tarda ~1-2 s. Hacerlo
    // ahora evita que ese costo se sume a la primera predicción.
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

  async correrAutotest(): Promise<void> {
    this.cargando = true;
    this.error = null;
    this.autotest = [];
    this.estado = 'Comparando contra las predicciones de PyTorch…';
    try {
      this.autotest = await this.autotestService.correr();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.cargando = false;
      this.estado = '';
    }
  }

  get autotestOk(): number {
    return this.autotest.filter((r) => r.ok).length;
  }

  porcentaje(prob: number): string {
    return `${(prob * 100).toFixed(1)}%`;
  }
}

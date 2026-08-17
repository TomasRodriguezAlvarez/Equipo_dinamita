/**
 * Definición de los 3 modelos TrashNet exportados a ONNX y de la metadata que
 * acompaña a cada uno.
 *
 * Todo lo numérico (tamaño de entrada, mean/std, umbral del binario) se lee del
 * `.labels.json` que genera `scripts/export_onnx.py`, nunca se hardcodea acá: si
 * se reentrena un modelo y cambia alguno de esos valores, la app lo toma solo.
 */

/** Contenido de un archivo `*.labels.json` generado por `scripts/export_onnx.py`. */
export interface EtiquetasModelo {
  clases: string[];
  input_size: number;
  mean: number[];
  std: number[];
  nota?: string;
  /** Solo el binario: umbral calibrado sobre validación. Ver PyTorch/CONTEXT.md §8. */
  umbral_basura?: number;
  /** Solo el binario: índice de la clase `basura` dentro de `clases`. */
  idx_basura?: number;
  nota_umbral?: string;
}

export interface DefinicionModelo {
  id: string;
  titulo: string;
  descripcion: string;
  rutaOnnx: string;
  rutaEtiquetas: string;
}

export const MODELOS: DefinicionModelo[] = [
  {
    id: 'trashnet_mobilenetv3',
    titulo: 'Material (6 clases)',
    descripcion: 'cardboard · glass · metal · paper · plastic · trash',
    rutaOnnx: 'assets/models/trashnet_mobilenetv3.onnx',
    rutaEtiquetas: 'assets/models/trashnet_mobilenetv3.labels.json',
  },
  {
    id: 'trashnet_jerarquico',
    titulo: 'Contenedor (4 clases)',
    descripcion: 'amarillo · azul · gris · verde',
    rutaOnnx: 'assets/models/trashnet_jerarquico.onnx',
    rutaEtiquetas: 'assets/models/trashnet_jerarquico.labels.json',
  },
  {
    id: 'trashnet_binario',
    titulo: 'Binario (2 clases)',
    descripcion: 'basura · reciclable — decide por umbral, no por argmax',
    rutaOnnx: 'assets/models/trashnet_binario.onnx',
    rutaEtiquetas: 'assets/models/trashnet_binario.labels.json',
  },
];

export interface ProbabilidadClase {
  clase: string;
  prob: number;
}

export interface Resultado {
  clase: string;
  confianza: number;
  probabilidades: ProbabilidadClase[];
  /** Cómo se decidió la clase, para que se vea en pantalla qué regla se aplicó. */
  regla: string;
  msPreproceso: number;
  msInferencia: number;
}

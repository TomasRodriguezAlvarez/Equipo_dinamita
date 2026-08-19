# scriptsPython — segunda entrega (TensorFlow)

Versión script de los tres notebooks de esta carpeta (`TrashNet_TensorFlow_
MobileNetV3_DEFINITIVO.ipynb`, `..._Jerarquico_DEFINITIVO.ipynb`,
`..._Binario_DEFINITIVO.ipynb`), siguiendo la misma estructura que
`PyTorch/scriptsPython/` (ver `PyTorch/CONTEXT.md` §13 para el razonamiento
completo detrás del diseño).

Los notebooks originales no se tocaron: quedan como referencia. El entregable de la
pauta (código/script, no Jupyter) es el contenido de esta carpeta.

## Estructura

```
scriptsPython/
├── comun/utils.py              arquitectura MobileNetV3, callbacks, gráficos
├── dataset/preparar_dataset.py separación train/validation/test
├── modelos/
│   ├── mobilenetv3/entrenar.py     6 clases de material
│   ├── jerarquico/entrenar.py      4 clases de contenedor (remapeo con tf.gather)
│   └── binario/entrenar.py         2 clases, reciclable/basura (desequilibrado)
├── consumo/predecir.py         script de consumo independiente
├── camara/capturar_camara.py   clasificación en vivo con OpenCV
├── pesos/                      .keras generados por los entrenar.py
└── resultados/graficos, matriz_de_confusion, reporte/
```

## Diferencias fieles a los notebooks originales (no se "corrigieron")

- **Optuna con 10-15 trials**, sin el baseline encolado como trial 0 ni el
  reentrenamiento del top-K que sí tiene la rama PyTorch
  (`PyTorch/CONTEXT.md` §9). Con solo 10 trials y `n_startup_trials=10` por
  defecto, `TPESampler` nunca sale de su fase de arranque aleatorio — es una
  limitación conocida del notebook original, documentada ahí, no algo que se
  introdujo al convertir a script.
- **El modelo binario incorpora el Garbage Dataset al entrenamiento** (70/15/15),
  a diferencia del binario de PyTorch que lo reserva solo para evaluación fuera
  de distribución. Es una decisión de diseño distinta entre las dos ramas, no un
  error — ver el docstring de `modelos/binario/entrenar.py`.

## Uso

```bash
# entorno con TensorFlow, Optuna y scikit-learn instalados
pip install opencv-python   # solo si se va a usar camara/capturar_camara.py

cd dataset && python preparar_dataset.py && cd ..

cd modelos/mobilenetv3 && python entrenar.py && cd ../..
cd modelos/jerarquico  && python entrenar.py && cd ../..
cd modelos/binario     && python entrenar.py && cd ../..
# --sin-optuna entrena solo el baseline, sin búsqueda

cd consumo
python predecir.py --modelo ../pesos/binario_optuna.keras --imagen foto.jpg --binario

cd ../camara
python capturar_camara.py --modelo ../pesos/mobilenetv3_optuna.keras
```

## Verificación hecha

Desde la sesión no había TensorFlow ni OpenCV instalados, así que **no se ejecutó
ningún entrenamiento**. Lo que sí se comprobó: `pyflakes` limpio, resolución
correcta de `sys.path` en los 5 scripts que importan `comun/utils.py`, y chequeo
por AST de que toda llamada a `u.*` y a funciones locales existe y respeta su
firma. El mapeo del jerárquico (`[1,3,0,1,0,2]`) coincide con el del notebook.

## Pendiente

- Falta correr al menos `entrenar.py --sin-optuna` de uno de los tres modelos
  para confirmar que todo encaja antes de lanzar una búsqueda larga, y probar
  `predecir.py` cargando un `.keras` real.
- A diferencia de la rama PyTorch, aquí los conteos por clase ya filtran por
  extensión, así que el `hola.md` que hay en `dataset/train/metal/` no afecta a
  los pesos de clase.
- Esta carpeta la generó una sesión de Claude Code a pedido de Benjamín, sobre la
  parte de TensorFlow del equipo. Si el dueño de esta rama (ver
  `PyTorch/CONTEXT.md` §7, quien trabaja `tensorflow/`) no estaba al tanto,
  vale la pena avisarle antes de dar esto por definitivo.

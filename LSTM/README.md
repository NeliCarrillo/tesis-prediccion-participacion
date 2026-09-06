# Modelo de contraste LSTM

Contiene el modelo no explicable que sirve de referencia frente a la red bayesiana, en sus
dos estados: como fue construido originalmente y como queda adaptado a los datos de esta
investigación.

## `viejo/`

`lstm_computacion_emergente.ipynb` es el cuaderno desarrollado en la asignatura Computación
Emergente, tal como se entregó y con las salidas de su última ejecución. Clasifica el
resultado final de un curso en cuatro categorías a partir de la actividad semanal
registrada en un entorno virtual, sobre un conjunto público de analítica del aprendizaje
ajeno a la Universidad Metropolitana. Se conserva sin modificar: es el punto de partida
contra el que se contrasta la adaptación.

## `nuevo/`

`adaptacion_lstm_participaciones.ipynb` adapta esa red para que prediga la cantidad de
participaciones de un estudiante en el trimestre, a partir de los `.csv` que produce
`preprocesamiento/generar_csv.py`. Las tres adaptaciones son:

- **Capa de entrada.** De ocho variables sociodemográficas y veintiuna semanas de clics, a
  dos entradas paralelas: los rasgos numéricos de cada una de las doce semanas y el código
  de tema de sus dos sesiones, que pasa por una capa de *embedding* de dos dimensiones. La
  codificación numérica de los campos cualitativos se aplica aquí, no en el `.csv`.
- **Capa de salida.** De una densa de cuatro unidades con activación *softmax* y entropía
  cruzada, a una densa de una unidad con activación lineal y error cuadrático medio. La red
  predice las participaciones que faltan desde el hito hasta el final del trimestre, sobre
  el objetivo estandarizado; el total es el acumulado observado más esa predicción. Las
  métricas, calculadas sobre el total, son RMSE, MAE y coeficiente de determinación.
- **Evaluación.** De una partición temporal única, a una validación cruzada que deja un
  trimestre fuera en cada iteración, y con el entrenamiento separado por asignatura.

El cuaderno compara el modelo contra dos referencias sencillas —predecir la media y
extrapolar proporcionalmente lo observado— porque la tarea admite en buena medida una
solución trivial y sin ellas el coeficiente de determinación no es interpretable.

## Ejecución

```bash
pip install tensorflow scikit-learn matplotlib pandas numpy
jupyter notebook LSTM/nuevo/adaptacion_lstm_participaciones.ipynb
```

El cuaderno localiza por sí mismo la carpeta `Datos Tesis Downstream` subiendo desde su
ubicación, de modo que no depende del directorio desde el que se lance Jupyter. Al
terminar deja las métricas en `metricas_adaptacion.csv`, junto al propio cuaderno.

# tesis-prediccion-participacion

Sistema de contraste de Redes Bayesianas y LSTM para la predicción explicable de la
participación en el aula (Universidad Metropolitana).

Este repositorio contiene los datos anonimizados, el código y los resultados que
respaldan el informe de la tesis. Las carpetas están numeradas en el orden en que se
desarrolló el trabajo, que coincide con las fases (*Sprints*) del Método.

## Mapa del repositorio

| Carpeta | Qué contiene | Fase del informe |
|---|---|---|
| `1_datos/originales/` | Planillas de participación y cronogramas de cada sección, tal como las entregaron los profesores, con las hojas "Estandar" y "Cronograma" añadidas | Levantamiento de información (*Sprint* 1) |
| `1_datos/estandarizados/` | Un CSV anonimizado por sección (una fila por estudiante y sesión); es la entrada de ambos modelos | Levantamiento de información (*Sprint* 1) |
| `1_datos/plantillas/` | Plantilla estándar de participaciones y catálogo de temas por asignatura | Levantamiento de información (*Sprint* 1) |
| `2_preprocesamiento/` | Script que convierte las planillas originales en los CSV estandarizados y anonimiza a los estudiantes | Levantamiento de información (*Sprint* 1) |
| `3_lstm/` | Modelo de contraste: el cuaderno original (`modelo_original/`), su adaptación con validación cruzada (`adaptacion/`) y sus métricas y predicciones (`resultados/`) | Adaptación del modelo *baseline* (*Sprint* 2) |
| `4_red_bayesiana/` | Modelo explicable: exploración de variables, código de la red, notebooks y resultados oficiales | Diseño y construcción de la red (*Sprints* 3 y 4) |
| `5_prototipo/` | Aplicación que ejecuta ambos modelos sobre un mismo caso, con los 12 modelos LSTM finales | Implementación del prototipo (*Sprint* 5) |
| `6_evaluacion/` | Comparación de los tres métodos y análisis complementarios de Resultados y Discusión | Validación y evaluación (*Sprint* 6) |
| `figuras/` | Todas las figuras del informe, agrupadas por capítulo, con su índice | Todo el informe |

**Para encontrar una figura del informe,** abra `figuras/INDICE.md`: indica el archivo de
cada figura y el script que la genera. Cada carpeta tiene su propio `README.md` con el
detalle de sus archivos.

## Cómo reproducir los resultados

Requiere Python 3.11 con las dependencias de `5_prototipo/requirements-dev.txt`.

1. `python3 2_preprocesamiento/generar_csv.py` regenera los CSV estandarizados, que salen
   idénticos byte a byte.
2. `3_lstm/adaptacion/adaptacion_lstm_participaciones.ipynb` entrena y valida la LSTM
   (cinco semillas) y guarda sus métricas en `3_lstm/resultados/`.
3. `4_red_bayesiana/notebooks/sprint4_inferencia.ipynb` (o
   `python3 4_red_bayesiana/codigo/inferencia.py`) ejecuta la validación cruzada de la red
   bayesiana y guarda sus predicciones en `4_red_bayesiana/resultados/`.
4. Los scripts de `6_evaluacion/codigo/` generan las tablas y figuras de Resultados y
   Discusión; cada uno comprueba sus cifras contra el informe.
5. `cd 5_prototipo && python3 app.py` abre el prototipo en `http://localhost:8080`.

## Datos personales

Los CSV estandarizados no contienen nombres ni cédulas: cada estudiante tiene un
identificador anónimo (`anon_###`). La tabla que relaciona cédulas e identificadores se
genera en `1_datos/originales/_procesado/_confidencial/` y está excluida del control de
versiones (`.gitignore`), por lo que nunca se publica.

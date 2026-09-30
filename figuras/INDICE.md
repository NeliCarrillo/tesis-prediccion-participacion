# Índice de figuras

Todas las figuras del proyecto están en esta carpeta, agrupadas según la parte del
informe en la que aparecen. Esta tabla indica, para cada figura del informe, el archivo
y el script que la genera, para poder auditarla o regenerarla.

La numeración corresponde a la versión actual del informe, con "Resultados de los Datos
Atípicos" antes de "Comparación LSTM y Bayes". Si el informe cambia de numeración,
actualice esta tabla; los nombres de los archivos no llevan número para que no queden
desactualizados.

## Método

| Figura | Título abreviado | Archivo | Generada por |
|---|---|---|---|
| 1 | Hoja "Estandar" de participaciones | No está en el repositorio: captura de la plantilla | `1_datos/plantillas/ESTANDAR_Participaciones.xlsx` |
| 2 | Hoja "Cronograma" de participaciones | No está en el repositorio: captura de la plantilla | `1_datos/plantillas/ESTANDAR_Participaciones.xlsx` |
| 3 | Tamaño de las secciones | `metodo_datos_y_exploracion/figura_tamano_grupo_exploracion.png` | `4_red_bayesiana/exploracion/figura_tamano_grupo_exploracion.py` |
| 4 | Total de participaciones por asignatura | `metodo_datos_y_exploracion/figura_distribucion_objetivo.png` | `3_lstm/adaptacion/adaptacion_lstm_participaciones.ipynb` |
| 5 | Participación semanal según año académico | `metodo_datos_y_exploracion/figura_participacion_semanal_anio_academico.png` | Sin script en el repositorio |
| 6 | Acumulado frente al total del trimestre | `metodo_datos_y_exploracion/figura_correlacion_acumulado_total.png` | `3_lstm/adaptacion/adaptacion_lstm_participaciones.ipynb` |
| 7 | Trayectoria de los tres casos atípicos | `metodo_datos_y_exploracion/figura_atipicos_dispersion.png` | `3_lstm/adaptacion/adaptacion_lstm_participaciones.ipynb` |
| 8 | Posición relativa en la lista | `metodo_datos_y_exploracion/figura_posicion_lista.png` | `4_red_bayesiana/exploracion/figura_posicion_lista.py` |
| 9 | Arquitectura de la LSTM adaptada | `metodo_modelos/figura_arquitectura.png` (y `.pdf`) | `3_lstm/adaptacion/figura_3_arquitectura.py` |
| 10 | Participación acumulada por tema | `metodo_datos_y_exploracion/figura_tema.png` | `4_red_bayesiana/exploracion/figura_tema.py` |
| 11 | Estructuras aprendidas automáticamente | `metodo_modelos/estructura_comparada_todas_asignaturas.png` | `4_red_bayesiana/codigo/estructura_aprendida.py` |
| 12 | Estructura de dependencias del grafo | `metodo_modelos/figura_grafo_bayesiano.png` | `4_red_bayesiana/codigo/figura_grafo_bayesiano.py` |
| 13 | Flujo del prototipo | `metodo_prototipo/figura_flujo_prototipo.png` | `5_prototipo/figura_flujo_prototipo.py` (versión de referencia; la del informe se ajustó a mano) |

En `metodo_modelos/` están también las estructuras aprendidas por asignatura
(`estructura_comparada_<asignatura>.png`) y la versión solo con nodos, que complementan
la Figura 11.

## Resultados y Discusión de Resultados

| Figura | Título abreviado | Archivo | Generada por |
|---|---|---|---|
| 14 | RMSE y R² de la LSTM y la extrapolación | `resultados/figura_lstm_vs_extrapolacion_tabla16.png` (y `.svg`) | `6_evaluacion/codigo/figura_lstm_vs_extrapolacion_tabla16.py` |
| 15 | Puntaje de Brier frente a referencias | No está en el repositorio | Elaborada fuera del código versionado |
| 16 | RMSE según el respaldo de la evidencia | No está en el repositorio | Elaborada fuera del código versionado; sus valores salen de `6_evaluacion/codigo/respaldo_evidencia.py` |
| 17 | Sensibilidad por variable y asignatura | `resultados/figura_sensibilidad_por_asignatura.png` | `6_evaluacion/codigo/sensibilidad_evidencia.py` |
| 18 | Predicciones de los tres casos atípicos | `resultados/figura_atipicos_predicciones.png` | `6_evaluacion/codigo/casos_atipicos.py` |
| 19 | RMSE de los tres métodos | `resultados/figura_comparacion_rmse_modelos.png` | `6_evaluacion/codigo/figura_comparacion_modelos.py` |
| 20 | Calendario de participación y acontecimientos | `resultados/figura_calendario_participacion.png` | `6_evaluacion/codigo/anomalias_calendario.py` |
| 21 | Participación acumulada y casos sin participación | `resultados/figura_mecanismo_anomalias.png` | `6_evaluacion/codigo/anomalias_calendario.py` |

## Apéndice I (prototipo)

Capturas reales de la interfaz, en `apendice_I_prototipo/`:

| Figura | Archivo |
|---|---|
| I1 | `historico_1_seleccion_a.png` |
| I2 | `historico_1_seleccion_b.png` |
| I3 | `historico_2_lstm.png` |
| I4 | `historico_3_bayes_a.png` |
| I5 | `historico_3_bayes_b.png` |
| I6 | `nuevo_1_formulario_a.png` |
| I7 | `nuevo_1_formulario_b.png` |
| I8 | `nuevo_1_formulario_c.png` |
| I9 | `nuevo_2_lstm.png` |
| I10 | `nuevo_3_bayes_a.png` |
| I11 | `nuevo_3_bayes_b.png` |

## Apéndice K (explicabilidad de la LSTM)

| Figura | Título abreviado | Archivo | Generada por |
|---|---|---|---|
| K1 | Predicciones con la misma evidencia y pesos de la primera capa LSTM | `apendice_K_explicabilidad_lstm/figura_explicabilidad_lstm.png` | `3_lstm/adaptacion/explicabilidad_lstm.py` |

## `no_incluidas_en_informe/`

Figuras que se generaron durante el trabajo pero que no aparecen en la versión actual del
informe: versiones anteriores (por ejemplo, las de tres asignaturas, antes de incorporar
Matemáticas Discretas, o `figura_discretizacion.png`, cuyo script comprueba los 437
registros de esa etapa), alternativas descartadas (`figura_grafo.png`,
`estructura_comparada_todas_asignaturas_prueba.png`) y análisis complementarios
(`figura_posterior_atipicos.png`). Se conservan como
registro; ninguna sustenta un resultado del informe.
